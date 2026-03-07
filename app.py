"""
SMILESGNN Clinical Toxicity Predictor — Streamlit application

Two-tab interface:
  Tab 1 — Batch Screening  : upload a CSV/TXT file or paste SMILES, batch
                             predict, visualise distribution, download CSV.
  Tab 2 — Deep Dive        : single-molecule GNNExplainer attribution with
                             atom/bond heatmap and ranked importance tables.

Usage
-----
  conda activate drug-tox-env
  cd /path/to/molecule
  streamlit run app.py
"""

import io
import sys
import warnings
from pathlib import Path

# Set non-interactive backend BEFORE any matplotlib import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd           # noqa: E402
import streamlit as st        # noqa: E402
import torch                  # noqa: E402
from rdkit import Chem        # noqa: E402

from src.inference import load_model, predict_batch          # noqa: E402
from src.graph_data import smiles_to_pyg_data                # noqa: E402
from src.gnn_explainer import explain_molecule, visualize_explanation  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DIR   = PROJECT_ROOT / "models" / "smilesgnn_model"
CONFIG_PATH = PROJECT_ROOT / "config" / "smilesgnn_config.yaml"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

EXAMPLES = {
    "Thalidomide (toxic)":    "O=C1CCC(=O)N1c1ccc2c(c1)C(=O)N(C2=O)",
    "5-Fluorouracil (toxic)": "O=c1[nH]cc(F)c(=O)[nH]1",
    "Aspirin (safe)":         "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine (safe)":        "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
}


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SMILESGNN Toxicity Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Cached model loading (once per session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading SMILESGNN model …")
def get_model():
    return load_model(MODEL_DIR, CONFIG_PATH, DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    threshold = st.slider(
        "Decision threshold",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="P(toxic) ≥ threshold → flagged as Toxic.",
    )

    gnn_epochs = st.slider(
        "GNNExplainer epochs (Deep Dive)",
        min_value=50, max_value=500, value=200, step=50,
        help="Higher = more stable atom masks, but slower. 200 works well.",
    )

    st.markdown("---")
    model_ok = (MODEL_DIR / "best_model.pt").exists()
    if model_ok:
        st.success("Model ready ✓")
    else:
        st.error(
            "**Model not found.**\n\n"
            "Run:\n```\npython scripts/train_hybrid.py\n```"
        )

    st.markdown("---")
    st.markdown("**SMILESGNN** · CITA 2026")
    st.markdown("AUC-ROC **0.997** · AUPRC **0.967**")
    st.caption("ClinTox · scaffold split · 11.5:1 class imbalance")
    st.caption(f"Device: `{DEVICE}`")


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.title("🧪 SMILESGNN Clinical Toxicity Predictor")
st.markdown(
    "Multimodal deep learning for drug toxicity screening. "
    "Combines **SMILES Transformer** + **GATv2 graph encoder** via attention fusion."
)

if not model_ok:
    st.warning("Model checkpoint not found. See sidebar for setup instructions.")
    st.stop()

model, tokenizer, wrapped_model = get_model()


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_screen, tab_deep = st.tabs(["🔬 Batch Screening", "🔍 Deep Dive (Explain)"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Batch Screening
# ═════════════════════════════════════════════════════════════════════════════

with tab_screen:
    st.header("Virtual Screening — Batch Prediction")
    st.markdown(
        "Upload a compound library (CSV, XLSX, or TXT) or paste SMILES directly. "
        "The model scores every compound and ranks them by P(toxic)."
    )

    # ── Input method ──────────────────────────────────────────────────────────
    input_method = st.radio(
        "Input method",
        ["Upload file (CSV / XLSX / TXT)", "Paste SMILES"],
        horizontal=True,
    )

    smiles_input: list = []
    names_input:  list = []
    labels_input        = None

    # ── File upload ───────────────────────────────────────────────────────────
    if input_method == "Upload file (CSV / XLSX / TXT)":
        uploaded = st.file_uploader(
            "Drag and drop your compound library here",
            type=["csv", "xlsx", "txt"],
            help=(
                "**CSV / XLSX**: needs a column named `smiles`, `SMILES`, or `Smiles`.  \n"
                "Optional: `name` (compound ID) and `label` / `CT_TOX` (0=non-toxic, 1=toxic).  \n\n"
                "**TXT**: one SMILES per line. Optional: `SMILES<TAB>name` per line."
            ),
        )

        if uploaded is not None:
            try:
                fname = uploaded.name.lower()
                if fname.endswith(".txt"):
                    lines = uploaded.read().decode().strip().splitlines()
                    for i, line in enumerate(lines):
                        parts = line.strip().split("\t")
                        smiles_input.append(parts[0].strip())
                        names_input.append(
                            parts[1].strip() if len(parts) > 1 else f"Mol-{i:03d}"
                        )
                else:
                    df_in = (
                        pd.read_excel(uploaded)
                        if fname.endswith(".xlsx")
                        else pd.read_csv(uploaded)
                    )
                    # Auto-detect SMILES column
                    smi_col = next(
                        (c for c in df_in.columns if c.lower() == "smiles"), None
                    )
                    if smi_col is None:
                        st.error(
                            "Could not find a column named `smiles` / `SMILES`. "
                            "Please rename the column and re-upload."
                        )
                        st.stop()

                    smiles_input = df_in[smi_col].astype(str).tolist()

                    name_col = next(
                        (c for c in df_in.columns
                         if c.lower() in ("name", "compound", "id", "compound_id")),
                        None,
                    )
                    names_input = (
                        df_in[name_col].astype(str).tolist()
                        if name_col
                        else [f"Mol-{i:03d}" for i in range(len(smiles_input))]
                    )

                    lbl_col = next(
                        (c for c in df_in.columns
                         if c.lower() in ("label", "ct_tox", "toxic", "toxicity")),
                        None,
                    )
                    if lbl_col:
                        labels_input = df_in[lbl_col].astype(int).tolist()

                st.success(
                    f"Loaded **{len(smiles_input)}** compounds from `{uploaded.name}`"
                    + (f" · labels found" if labels_input else "")
                )

            except Exception as e:
                st.error(f"Failed to parse file: {e}")

    # ── Paste SMILES ──────────────────────────────────────────────────────────
    else:
        paste = st.text_area(
            "Paste SMILES (one per line, optionally `SMILES<TAB>name`)",
            height=180,
            placeholder=(
                "CC(=O)Oc1ccccc1C(=O)O\t Aspirin\n"
                "Cn1cnc2c1c(=O)n(C)c(=O)n2C\t Caffeine\n"
                "O=C1CCC(=O)N1c1ccc2c(c1)C(=O)N(C2=O)\t Thalidomide"
            ),
        )
        if paste.strip():
            for i, line in enumerate(paste.strip().splitlines()):
                parts = line.strip().split("\t")
                smiles_input.append(parts[0].strip())
                names_input.append(
                    parts[1].strip() if len(parts) > 1 else f"Mol-{i:03d}"
                )
            st.info(f"**{len(smiles_input)}** SMILES parsed.")

    # ── Run ───────────────────────────────────────────────────────────────────
    if smiles_input:
        if st.button("▶  Run Screening", type="primary", use_container_width=True):
            with st.spinner(f"Scoring {len(smiles_input)} compounds …"):
                results_df = predict_batch(
                    smiles_list   = smiles_input,
                    tokenizer     = tokenizer,
                    wrapped_model = wrapped_model,
                    device        = DEVICE,
                    names         = names_input or None,
                    true_labels   = labels_input,
                    threshold     = threshold,
                    batch_size    = 32,
                )
            st.session_state["screen_results"] = results_df

    # ── Display ───────────────────────────────────────────────────────────────
    if "screen_results" in st.session_state:
        df = st.session_state["screen_results"]
        n_toxic    = (df["Predicted"] == "Toxic").sum()
        n_nontoxic = (df["Predicted"] == "Non-toxic").sum()
        n_error    = (df["Predicted"] == "Parse error").sum()

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Screened",      len(df))
        c2.metric("Flagged toxic", int(n_toxic))
        c3.metric("Passed",        int(n_nontoxic))
        if n_error:
            c4.metric("Parse errors", int(n_error))

        st.markdown("---")

        # Charts
        ch1, ch2 = st.columns([3, 2])
        valid_df  = df[df["P(toxic)"].notna()]

        with ch1:
            st.subheader("Probability distribution")
            fig, ax = plt.subplots(figsize=(7, 3.5))
            has_labels = (
                "True label" in df.columns
                and set(df["True label"].dropna().unique()) >= {"Toxic", "Non-toxic"}
            )
            if has_labels:
                t_probs = valid_df[valid_df["True label"] == "Toxic"]["P(toxic)"]
                n_probs = valid_df[valid_df["True label"] == "Non-toxic"]["P(toxic)"]
                ax.hist(n_probs, bins=20, color="steelblue", alpha=0.7,
                        label="Non-toxic (true)", edgecolor="white")
                ax.hist(t_probs, bins=20, color="salmon",    alpha=0.9,
                        label="Toxic (true)",     edgecolor="white")
                ax.legend(fontsize=9)
            else:
                ax.hist(valid_df["P(toxic)"], bins=20, color="steelblue", edgecolor="white")
            ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2,
                       label=f"Threshold = {threshold}")
            ax.set_xlabel("P(toxic)", fontsize=11)
            ax.set_ylabel("Count",    fontsize=11)
            ax.legend(fontsize=9)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
            plt.close()
            st.image(buf)

        with ch2:
            st.subheader("Decision split")
            if n_toxic + n_nontoxic > 0:
                fig2, ax2 = plt.subplots(figsize=(4, 4))
                sizes  = [s for s in [n_toxic, n_nontoxic, n_error] if s > 0]
                labels = [
                    l for l, s in zip(
                        [f"Flagged ({n_toxic})", f"Passed ({n_nontoxic})",
                         f"Error ({n_error})"],
                        [n_toxic, n_nontoxic, n_error],
                    ) if s > 0
                ]
                colors = ["salmon", "steelblue", "lightgrey"][: len(sizes)]
                ax2.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%",
                        startangle=90, textprops={"fontsize": 10})
                ax2.set_title(f"threshold = {threshold}", fontsize=10)
                plt.tight_layout()
                buf2 = io.BytesIO()
                plt.savefig(buf2, format="png", dpi=130, bbox_inches="tight")
                plt.close()
                st.image(buf2)

        st.markdown("---")
        st.subheader("Ranked results")

        def _row_style(row):
            if row["Predicted"] == "Toxic":
                return ["background-color: #ffe0e0"] * len(row)
            if row["Predicted"] == "Parse error":
                return ["background-color: #f5f5f5; color: grey"] * len(row)
            return [""] * len(row)

        show_cols = [
            c for c in ["Name", "SMILES", "P(toxic)", "Predicted", "True label", "Correct"]
            if c in df.columns
        ]
        styled = (
            df[show_cols]
            .style.apply(_row_style, axis=1)
            .format({"P(toxic)": lambda v: f"{v:.4f}" if v is not None else "—"})
        )
        st.dataframe(styled, use_container_width=True, height=420)

        st.download_button(
            "⬇  Download results as CSV",
            data=df.to_csv(index=False).encode(),
            file_name="smilesgnn_screening_results.csv",
            mime="text/csv",
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Deep Dive (Explain)
# ═════════════════════════════════════════════════════════════════════════════

with tab_deep:
    st.header("Lead Optimization — Structural Explanation")
    st.markdown(
        "Enter a single SMILES string to get atom- and bond-level importance scores "
        "that explain **why** the model flags this compound."
    )
    st.info(
        "GNNExplainer explains the **GATv2 graph pathway** only. "
        "The SMILES Transformer embedding is frozen per molecule. "
        "Treat atom/bond scores as structural hypotheses, not definitive attributions.",
        icon="ℹ️",
    )

    # ── Quick-fill buttons ────────────────────────────────────────────────────
    st.markdown("**Try a known compound:**")
    ex_cols = st.columns(len(EXAMPLES))
    for col, (name, smi) in zip(ex_cols, EXAMPLES.items()):
        if col.button(name, use_container_width=True):
            st.session_state["deep_smiles"] = smi

    # ── SMILES input ──────────────────────────────────────────────────────────
    default_smi = st.session_state.get("deep_smiles", "")
    deep_smiles = st.text_input(
        "SMILES string",
        value=default_smi,
        placeholder="e.g.  O=C1CCC(=O)N1c1ccc2c(c1)C(=O)N(C2=O)  (Thalidomide)",
    )

    run_explain = st.button(
        "▶  Predict + Explain",
        type="primary",
        use_container_width=True,
        disabled=not deep_smiles.strip(),
    )

    if run_explain and deep_smiles.strip():
        smi = deep_smiles.strip()

        # ── Stage A: quick prediction ─────────────────────────────────────────
        with st.spinner("Predicting …"):
            quick_df = predict_batch(
                [smi],
                tokenizer     = tokenizer,
                wrapped_model = wrapped_model,
                device        = DEVICE,
                threshold     = threshold,
            )

        prob = quick_df.iloc[0]["P(toxic)"]
        pred = quick_df.iloc[0]["Predicted"]

        if pred == "Parse error":
            st.error(
                f"RDKit could not parse `{smi}`.  \n"
                "This may be an organometallic or unusual coordination compound "
                "(Pt, Pd, Ir, etc.). The model is trained on organic SMILES only."
            )
            st.stop()

        # Prediction banner
        m1, m2, m3 = st.columns(3)
        m1.metric("P(toxic)", f"{prob:.4f}")
        m2.metric("Prediction", pred)
        m3.metric("Threshold", threshold)

        if pred == "Toxic":
            st.error(f"⚠️  Flagged as **TOXIC** — P = {prob:.4f}")
        else:
            st.success(f"✓  Predicted **NON-TOXIC** — P = {prob:.4f}")

        target_class = 1 if prob >= threshold else 0

        # ── Stage B: GNNExplainer ─────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Structural explanation (GNNExplainer)")
        st.caption(
            f"Explaining the {'**toxic**' if target_class == 1 else '**non-toxic**'} "
            f"prediction using {gnn_epochs} optimisation epochs.  "
            f"Red = high importance · Green = low importance."
        )

        with st.spinner(f"Running GNNExplainer ({gnn_epochs} epochs) …"):
            try:
                pyg = smiles_to_pyg_data(smi, label=target_class)
            except Exception as e:
                st.error(f"Graph featurisation failed: {e}")
                st.stop()

            result = explain_molecule(
                smiles       = smi,
                model        = model,
                tokenizer    = tokenizer,
                pyg_data     = pyg,
                device       = DEVICE,
                epochs       = gnn_epochs,
                target_class = target_class,
            )

        # Render heatmap into BytesIO (save_path accepts file-like objects)
        buf_exp = io.BytesIO()
        visualize_explanation(
            result,
            figsize        = (13, 5),
            atom_threshold = 0.4,
            bond_threshold = 0.4,
            save_path      = buf_exp,
        )
        buf_exp.seek(0)
        st.image(buf_exp, use_container_width=True)

        # ── Atom table ────────────────────────────────────────────────────────
        mol      = Chem.MolFromSmiles(smi)
        atom_imp = result["atom_importance"]
        bond_imp = result["bond_importance"]

        atom_rows = []
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            imp = float(atom_imp[idx])
            atom_rows.append({
                "Atom":          idx,
                "Element":       atom.GetSymbol(),
                "Importance":    round(imp, 4),
                "Bar":           "█" * int(imp * 10),
                "Hybridization": str(atom.GetHybridization()).split(".")[-1],
                "In ring":       "✓" if atom.IsInRing()      else "",
                "Aromatic":      "✓" if atom.GetIsAromatic() else "",
            })

        atom_df = (
            pd.DataFrame(atom_rows)
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )

        st.markdown("#### Atom importance (ranked)")
        st.dataframe(atom_df, use_container_width=True, height=320)

        # ── Bond table ────────────────────────────────────────────────────────
        bond_rows = []
        for bond in mol.GetBonds():
            k   = bond.GetIdx()
            imp = float(bond_imp[k]) if k < len(bond_imp) else 0.0
            a1  = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetSymbol()
            a2  = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetSymbol()
            bond_rows.append({
                "Bond":       k,
                "Atoms":      f"{a1}({bond.GetBeginAtomIdx()})–{a2}({bond.GetEndAtomIdx()})",
                "Type":       str(bond.GetBondTypeAsDouble()),
                "Importance": round(imp, 4),
            })

        bond_df = (
            pd.DataFrame(bond_rows)
            .sort_values("Importance", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )

        st.markdown("#### Top-10 bond importance")
        st.dataframe(bond_df, use_container_width=True)

        # ── Downloads ─────────────────────────────────────────────────────────
        dl1, dl2 = st.columns(2)
        dl1.download_button(
            "⬇  Atom importance (CSV)",
            data      = atom_df.to_csv(index=False).encode(),
            file_name = "atom_importance.csv",
            mime      = "text/csv",
        )
        dl2.download_button(
            "⬇  Bond importance (CSV)",
            data      = bond_df.to_csv(index=False).encode(),
            file_name = "bond_importance.csv",
            mime      = "text/csv",
        )
