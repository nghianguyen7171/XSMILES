# model_server/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List

class PredictRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string")
    threshold: float = Field(0.5, ge=0.0, le=1.0)

class PredictResponse(BaseModel):
    smiles: str
    canonical_smiles: Optional[str]
    p_toxic: float
    label: str 
    confidence: float
    threshold_used: float

class BatchPredictRequest(BaseModel):
    smile_list: List[str]
    threshold: float = 0.5

class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total: int
    n_toxic: int
    n_non_toxic: int
    n_errors: int

class ExplainRequest(BaseModel):
    smiles: str
    epochs: int = Field(200, ge=50, le=500)
    target_class: Optional[int] = None  # None = auto-detect from prediction

class AtomImportance(BaseModel):
    atom_idx: int
    element: str
    importance: float
    is_in_ring: bool

class BondImportance(BaseModel):
    bond_idx: int
    atom_pair: str
    bond_type: str
    importance: float

class ExplainResponse(BaseModel):
    smiles: str
    p_toxic: float
    label: str
    top_atoms: List[AtomImportance]
    top_bonds: List[BondImportance]
    heatmap_base64: str     # PNG image encoded as base64
    chemical_interpretation: str
    explainer_note: str     # Document known limitations

