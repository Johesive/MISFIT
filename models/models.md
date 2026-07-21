# Required Models & Data

This directory contains model weights and large data files that are **NOT tracked in git**.
Download the following before running the notebooks.

---

## 1. ESM2-650M

- **Source**: https://huggingface.co/facebook/esm2_t33_650M_UR50D
- **Files needed**: `pytorch_model.bin`, `config.json`, `tokenizer.json`, `tokenizer_config.json`
- **Place in**: `models/esm2_650M/`
- **Used by**: `data_preparation/3_baseline.ipynb`, `data_preparation/4.0_esm2_local_delta.ipynb`

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/esm2_t33_650M_UR50D', local_dir='models/esm2_650M')"
```

---

## 2. TabPFN v2 Classifier

- **Source**: https://huggingface.co/Prior-Labs/tabpfn_3 (requires accepting terms of use)
- **File**: `tabpfn-v2-classifier.ckpt` (~28 MB)
- **Place at**: `models/tabpfn-v2-classifier.ckpt`
- **Used by**: `xgboost_trial/task16_tabpfn.ipynb`

```bash
# 1. Go to https://huggingface.co/Prior-Labs/tabpfn_3 and accept the terms
# 2. Login with your HF token
huggingface-cli login
# 3. Download
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Prior-Labs/tabpfn_3', 'tabpfn-v2-classifier.ckpt', local_dir='models')"
```

---

## 3. AlphaFold PDB Files

- **Source**: https://alphafold.ebi.ac.uk/
- **Files**: ~910 PDB files, one per UniProt accession in the dataset
- **Place in**: `models/alphafold_pdb/` (raw PDBs) and `models/alphafold_pdb_clean/` (pdbfixer-cleaned)
- **Used by**: `data_preparation/3.5.2_alphafold_download.ipynb`, `xgboost_trial/task8_struct_ddg.ipynb` to `task10_foldx_ddg.ipynb`

```bash
# The notebook data_preparation/3.5.2_alphafold_download.ipynb handles downloading
# For manual download of a single PDB:
wget https://alphafold.ebi.ac.uk/files/AF-{UNIPROT}-F1-model_v4.pdb -O models/alphafold_pdb/{UNIPROT}.pdb
```

---

## 4. PUPS Model (optional — baseline only)

- **Source**: The PUPS model repository (check `models/PUPS/README.md` for details)
- **Checkpoints**: ~23 `.ckpt` files in `models/PUPS/checkpoints/` (~1.5 GB total)
- **Used by**: `pups_trial/` notebooks
- **Note**: PUPS is only used as a baseline comparison; the main XGBoost/TabPFN pipeline does not depend on it.

---

## 5. RaSP / ML-ddG-Blaabjerg (optional — for ddg_rasp)

- **Source**: https://github.com/KULL-Centre/ML-ddG-Blaabjerg
- **Pretrained models**: `models/_2022_ML-ddG-Blaabjerg-main/pretrained_models/` (cavity + ds models, ~20 `.pt` files)
- **Used by**: `xgboost_trial/task9_rasp_ddg.ipynb`

```bash
git clone https://github.com/KULL-Centre/ML-ddG-Blaabjerg.git
# Copy pretrained_models/ into models/_2022_ML-ddG-Blaabjerg-main/pretrained_models/
```

---

## 6. FoldX (optional — for ddg_foldx)

- **Source**: https://foldxsuite.crg.eu/ (requires academic license)
- **Binary**: `foldx` (Linux x86_64, ~83 MB)
- **Place at**: `models/FoldX/foldx`
- **Used by**: `xgboost_trial/task10_foldx_ddg.ipynb`

After downloading from the FoldX website, place the binary at `models/FoldX/foldx` and ensure it's executable:
```bash
chmod +x models/FoldX/foldx
```
