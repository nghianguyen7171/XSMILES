# model_server/main.py
"""
ToxAgent Model Server - FastAPI wrapper around SMILESGNN (XSmiles)

Endpoints: 
    GET /health         -> health check
    POST /predict       -> single molecule toxicity prediction
    POST /predict/batch -> batch prediction
    POST /explain       -> GNNExplainer atom/bond attribution

Deployment: Docker container -> Google Cloud Run
"""

import sys
import base64
import io
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import torch 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from rdkit import Chem

# Important: Add project root to sys.path so XSmiles src/ modules are importatble
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import load_model, predict_batch
from src.graph_data import smiles_to_pyg_data
from src.gnn_explainer import explain_molecule, visualize_explanation
from model_server.schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    ExplainRequest, ExplainResponse, AtomImportance, BondImportance
)

# Configuration
MODEL_DIR = PROJECT_ROOT / "models" / "smilesgnn_model"
CONFIG_PATH = PROJECT_ROOT / "config" / "smilesgnn_config.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_server")

# Lifespan: load model once at startup
model_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Loading SMILESGNN model on {DEVICE}...")
    try:
        model, tokenizer, wrapped_model = load_model(MODEL_DIR, CONFIG_PATH, DEVICE)
        model_state["model"] = model
        model_state["tokenizer"] = tokenizer
        model_state["wrapped"] = wrapped_model
        logger.info("Model loaded succesfully.")
    except Exception as e:
        logger.info(f"Model load FAILED: {e}")
        raise
    yield
    model_state.clear()

# FastAPI App
app = FastAPI(
    title="ToxAgent Model Server",
    description="SMILESGNN toxicity prediction API for ToxAgent agentic system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health Check
@app.get("/health")
async def health():
    import torch 
    model_ready = "model" in model_state
    return {
        "status": "ok" if model_ready else "degraded",
        "model_loaded": model_ready,
        "device": DEVICE,
        "model_dir_exists": MODEL_DIR.exists(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else None,
    }

# Single Prediction
@app.post("predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Predict clinical toxicity for a single SMILES molecule"""
    if "model" not in model_state:
        raise HTTPException(503, "Model not loaded")
    
    smiles = req.smiles.strip()

    # Validate and canonicalize SMILES
    mol = Chem.MolFromSmiles(smiles)
    canonical = Chem.MolToSmiles(mol) if mol else None

    try: 
        results_df = predict_batch(
            smiles_list=[smiles],
            tokenizer=model_state["tokenizer"],
            wrapped_model=model_state["wrapped"],
            device=DEVICE,
            threshold=req.threshold,
        )
    except Exception as e:
        raise HTTPException(500, f"Inference error: {str(e)}")

    row = results_df.iloc[0]
    p_toxic = float(row["P(toxic)"]) if row["P(toxic)"] is not None else 0.0
    predicted = str(row["Predicted"])

    # Map predicted label to uppercase for consistency
    label_map = {"Toxic": "TOXIC", "Non-toxic": "NON_TOXIC", "Parse error": "PARSE_ERROR"}
    label = label_map.get(predicted, "UNKNOWN")

    # Confidence: distance from threshold
    confidence = abs(p_toxic - req.threshold) / max(req.threshold, 1 - req.threshold)

    return PredictResponse(
        smiles=smiles,
        canonical_smiles=canonical,
        p_toxic=p_toxic,
        label=label,
        confidence=min(confidence, 1.0),
        threshold_used=req.threshold,
    )

# Batch Prediction
@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch_endpoint(req: BatchPredictRequest):
    """Predict clinical toxicity for a list of SMILES molecules."""
    if "model" not in model_state:
        raise HTTPException(503, "Model not loaded")
    
    if len(req.smile_list) > 500:
        raise HTTPException(400, "Batch size limited to 500 molecules")
    
    try:
        results_df = predict_batch(
            smiles_list=req.smile_list,
            tokenizer=model_state["tokenizer"],
            wrapped_model=model_state["wrapped"],
            device=DEVICE,
            threshold=req.threshold,
        )

    except Exception as e:
        raise HTTPException(500, f"Batch inference error: {str(e)}")
    
    results = []
    for _, row in results_df.iterrows():
        p = float(row["P(toxic)"]) if row["P(toxic)"] is not None else 0.0
        predicted = str(row["Predicted"])
        label_map = {"Toxic": "TOXIC", "Non-toxic": "NON_TOXIC", "Parse error": "PARSE_ERROR"}
        label = label_map.get(predicted, "UNKNOWN")
        confidence = abs(p - req.threshold) / max(req.threshold, 1 - req.threshold)
        results.append(PredictResponse(
            smiles=row.get("SMILES", ""),
            canonical_smiles=None,
            p_toxic=p,
            label=label,
            confidence=min(confidence, 1.0),
            threshold_used=req.threshold,
        ))

    n_toxic = sum(1 for r in results if r.label == "TOXIC")
    n_non_toxic = sum(1 for r in results if r.label == "NON_TOXIC")
    n_errors = sum(1 for r in results if r.label == "PARSE_ERROR")

    return BatchPredictResponse(
        results=results,
        total=len(results),
        n_toxic=n_toxic,
        n_non_toxic=n_non_toxic,
        n_errors=n_errors,
    )

# GNNExplainer
@app.post("/explain", response_model=ExplainResponse)
async def explain(req: ExplainRequest):
    """
    Generate GNNExplainer atom/bond attribution for a molecule.

    NOTE: GNNExplainer atom/bond attribution for a molecule.
    (P(toxic) from explainer ≠ P(toxic) from /predict). 
    This is a known limitation documented in XSmiles README. 
    Use /predict for final label; use /explain only for structural attribution.
    """
    if "model" not in model_state:
        raise HTTPException(503, "Model not loaded")
    
    smiles = req.smiles.strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        raise HTTPException(400, f"Invalid SMILES: {smiles}")
    # First get real prediction for targert_class determination
    pred_df = predict_batch(
        smiles_list=[smiles],
        tokenizer=model_state["tokenizer"],
        wrapped_model=model_state["wrapped"],
        device=DEVICE,
    )
    p_toxic = float(pred_df.iloc[0]["P(toxic)"])
    label_str = str(pred_df.iloc[0]["Predicted"])
    target_class = req.target_class if req.target_class is not None else (1 if p_toxic >= 0.5 else 0)

    # Run GNNExplainer
    try:
        pyg = smiles_to_pyg_data(smiles, label=target_class)
        result = explain_molecule(
            smiles=smiles,
            model=model_state["model"],
            tokenizer=model_state["tokenizer"],
            pyg_data=pyg,
            device=DEVICE,
            epochs=req.epochs,
            target_class=target_class,
        )
    except Exception as e:
        raise HTTPException(500, f"GNNExplainer error: {str(e)}")
    
    # Render heatmap to base64 PNG
    buf = io.BytesIO()
    visualize_explanation(result, figsize=(13, 5), save_path=buf)
    buf.seek(0)
    heatmap_b64 = base64.b64decode(buf.read()).decode("utf-8")

    # Build atom/bond lists
    atom_importance = result["atom_importance"]
    bond_importance = result["bond_importance"]

    top_atoms = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        imp = float(atom_importance[idx])
        top_atoms.append(AtomImportance(
            atom_idx=idx,
            element=atom.GetSymbol(),
            importance=round(imp, 4),
            is_in_ring=atom.IsInRing(),
            is_aromatic=atom.GetIsAromatic(),
        ))
    top_atoms.sort(key=lambda x: x.importance, reverse=True)

    top_bonds = []
    for bond in mol.GetBonds():
        k = bond.GetIdx()
        imp = float(bond_importance[k]) if k < len(bond_importance) else 0.0
        a1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetSymbol()
        a2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetSymbol()
        top_bonds.append(BondImportance(
            bond_idx=k,
            atom_pair=f"{a1}({bond.GetBeginAtomIdx()}) - {a2}({bond.GetEndAtomIdx()})",
            bond_type=str(bond.GetBondTypeAsDouble()),
            importance=round(imp, 4),
        ))
    top_bonds.sort(key=lambda x: x.importance, reverse=True)
    top_bonds = top_bonds[:10]
    
    label_map = {"Toxic": "TOXIC", "Non-toxic": "NON_TOXIC"}
    return ExplainResponse(
        smiles=smiles,
        p_toxic=p_toxic,
        label=label_map.get(label_str, "UNKNOWN"),
        top_atoms=top_atoms[:10],
        top_bonds=top_bonds,
        heatmap_base64=heatmap_b64,
        chemical_interpretation=f"Top contributing atom: {top_atoms[0].element}",
        explainer_note="GNNExplainer optimizes only the GATv2 graph pathway.",
    )