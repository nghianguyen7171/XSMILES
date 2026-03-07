# Test Data

Test files for the SMILESGNN Streamlit app (`app.py`).
All organic compounds are drawn from the **ClinTox scaffold-split test set** (148 molecules).

---

## Files

| File | Format | Rows | Labels | Purpose |
|---|---|---|---|---|
| `full_test_set.csv` | CSV | 148 | yes | Full ClinTox test set (10 toxic, 138 non-toxic) |
| `screening_library.csv` | CSV | 30 | yes | Balanced library: 10 toxic + 20 non-toxic (shuffled) |
| `screening_library.xlsx` | XLSX | 30 | yes | Same as above, Excel format |
| `toxic_compounds.csv` | CSV | 10 | yes | All 10 confirmed toxic test compounds |
| `smiles_only.csv` | CSV | 20 | no | SMILES column only — tests minimal input (no names/labels) |
| `compounds.txt` | TXT | 15 | no | One SMILES per line — tests bare TXT input |
| `named_compounds.txt` | TXT | 15 | no | `SMILES<TAB>name` per line — tests named TXT input |
| `with_parse_errors.csv` | CSV | 8 | yes | Mix of organic + organometallic compounds |
| `reference_panel.csv` | CSV | 8 | yes | 8 well-known drugs (external sanity check) |

---

## Column conventions

All CSV/XLSX files use these column names, which the app auto-detects:

| Column | Required | Description |
|---|---|---|
| `smiles` | yes | SMILES string |
| `name` | optional | Compound identifier (shown in results table) |
| `label` | optional | Ground-truth: `0` = non-toxic, `1` = toxic |

---

## Expected app behaviour per file

### `full_test_set.csv`
Upload to **Batch Screening** tab.
Expected: ~97% accuracy, histogram shows clear separation, ~10 flagged toxic.

### `screening_library.csv` / `screening_library.xlsx`
Upload to **Batch Screening** tab.
Expected: 29/30 correct (one Pt-complex scored as non-toxic), 9 flagged toxic.
Both CSV and XLSX are identical content — use to test both upload formats.

### `toxic_compounds.csv`
Upload to **Batch Screening** tab.
Expected: all rows predicted Toxic (except TOX-009 = Carboplatin, a Pt complex outside training distribution).

### `smiles_only.csv`
Upload to **Batch Screening** tab.
Expected: model scores all 20 compounds; names auto-assigned as `Mol-000`, `Mol-001`, etc.
True-label column absent so `Correct` column shows `—`.

### `compounds.txt`
Upload to **Batch Screening** tab (TXT format).
Expected: 15 compounds scored, names auto-assigned as `Mol-00`, `Mol-01`, etc.

### `named_compounds.txt`
Upload to **Batch Screening** tab (TXT format).
Expected: names parsed from TAB-separated second column.

### `with_parse_errors.csv`
Upload to **Batch Screening** tab.
Contains 3 organometallic SMILES (`Cisplatin-Pt`, `Carboplatin-Pt`, `Bi-subgallate`).
Behaviour depends on RDKit version:
- RDKit ≥ 2022.09: organometallics may be silently processed with low/uncertain P(toxic)
- Older RDKit: may produce `Parse error` rows in the results table
Use this file to observe how the model handles out-of-distribution chemistry.

### `reference_panel.csv`
Upload to **Batch Screening** tab.
**Expected predictions** (model is trained on organic ClinTox molecules):

| Compound | True | Expected P(toxic) | Notes |
|---|---|---|---|
| TOX-ref-004 | Toxic | ~0.94 | In ClinTox test set |
| 5-Fluorouracil | Toxic | moderate | Antimetabolite |
| Methotrexate | Toxic | variable | Antifolate |
| Benzo[a]pyrene | Toxic | variable | PAH |
| Aspirin | Non-toxic | ~0.05 | Common safe drug |
| Caffeine | Non-toxic | ~0.05 | Common safe drug |
| Warfarin | Non-toxic | variable | Narrow therapeutic window |
| Acetaminophen | Non-toxic | ~0.05 | Safe at normal dose |

> **Note**: Thalidomide, Methotrexate, Benzo[a]pyrene are NOT in the ClinTox dataset.
> The model's predictions for these are extrapolations, not validated outputs.
> Aspirin, Caffeine, and Acetaminophen may or may not be in the training set.

---

## Deep Dive tab — suggested molecules

Copy any of these SMILES into the **Deep Dive** tab to test GNNExplainer:

| Compound | SMILES | In ClinTox test? |
|---|---|---|
| TOX-001 (highest P) | `C1CN(CCN1CC2=CC3=C(C=C2)OC(O3)(F)F)C(=O)NC4=C(C=CN=C4)Cl` | yes |
| TOX-004 | `C1=CC=C2C(=C1)C(=NN=C2NC3=CC=C(C=C3)Cl)CC4=CC=NC=C4` | yes |
| TOX-010 | `C1=CC(=C(C=C1Cl)Cl)C(=O)NS(=O)(=O)C2=CC=C(S2)Br` | yes |
| Aspirin (safe) | `CC(=O)Oc1ccccc1C(=O)O` | no |
| Caffeine (safe) | `Cn1cnc2c1c(=O)n(C)c(=O)n2C` | no |
