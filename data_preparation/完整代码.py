# 1_data_preprocessing.py
import pandas as pd
from pathlib import Path

base_path = Path("/mnt/volume6/czj/labLGN/LabLZ")
file_path = base_path / "mmc1.xlsx"

import re

aa3to1 = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V'
}


# Input Format Example: LDHA K222E
def extract_mutation(variant_str, gene):
    if not isinstance(variant_str, str):
        return None
    
    # Remove the gene prefix if present
    prefix = str(gene) + " "
    mut = variant_str[len(prefix):] if variant_str.startswith(prefix) else variant_str.strip()
    
    # K222E
    if re.match(r'^[A-Z]\d+[A-Z]$', mut):
        return mut
    
    # Tyr418H
    m = re.match(r'^([A-Z][a-z]{2})(\d+)([A-Z])$', mut)
    if m:
        aa3, pos, new_aa = m.group(1), m.group(2), m.group(3)
        if aa3 in aa3to1:
            return f"{aa3to1[aa3]}{pos}{new_aa}"
    
    return None

def make_key(df, variant_col):
    # use gene and mutation to create a unique key for each row
    mut = df.apply(lambda r: extract_mutation(r[variant_col], r["Gene"]), axis=1)
    return df["Gene"].astype(str) + " " + mut.astype(str)

df = pd.read_excel(file_path, sheet_name="Variant annotation")
print(f"Number of rows: {len(df)}")
print(f"Column names: {df.columns.tolist()}")

df_clean = df[df["Mislocalized"].isin([0, 1])].copy()
print(f"\nAfter filtering NAs: {len(df_clean)} rows")
print(df_clean["Mislocalized"].value_counts())

cols = ["Gene", "Variant", "Variant (alternative)", "Uniprot",
        "Mislocalized", "Mislocalization phenotype",
        "AlphaMissense score", "AlphaMissense class",
        "ClinVar class", "HeLa expression (TPM)"]
df_model = df_clean[cols].copy()
df_model["source"] = "main"
print(f"\ndf_model: {len(df_model)} rows, UniProt missing: {df_model['Uniprot'].isna().sum()}")


df_benign = pd.read_excel(file_path, sheet_name="Additional benign variants")

# filter NAs 
df_benign_clean = df_benign[df_benign["Mislocalized?"].isin([0, 1])].copy()
print(f"\nAdditional benign: {len(df_benign_clean)} rows")
print(df_benign_clean["Mislocalized?"].value_counts())

# check for overlap between the two datasets
va_keys  = set(make_key(df_model, "Variant"))
abv_keys = set(make_key(df_benign_clean, "Variant"))
overlap  = va_keys & abv_keys
print(f"Overlap rows: {len(overlap)}")  # is 0

# align column formats
df_benign_aligned = pd.DataFrame({
    "Gene":                      df_benign_clean["Gene"].values,
    "Variant":                   df_benign_clean["Variant"].values,
    "Variant (alternative)":     None,
    "Uniprot":                   None,
    "Mislocalized":              df_benign_clean["Mislocalized?"].astype(int).values,
    "Mislocalization phenotype": None,
    "AlphaMissense score":       None,
    "AlphaMissense class":       None,
    "ClinVar class":             None,
    "HeLa expression (TPM)":     None,
    "source":                    "additional_benign"
})

df_combined = pd.concat([df_model, df_benign_aligned], ignore_index=True)
print(f"\nTotal rows after combining: {len(df_combined)}")
print(df_combined["Mislocalized"].value_counts())

df_combined["Mutation"] = df_combined.apply(
    lambda row: extract_mutation(row["Variant"], row["Gene"]), axis=1)

df_combined["Mutation (alternative)"] = df_combined.apply(
    lambda row: extract_mutation(row["Variant (alternative)"], row["Gene"]), axis=1)

print(f"Mutation extraction successful: {df_combined['Mutation'].notna().sum()}/{len(df_combined)}")
print(f"Mutation (alternative) extraction successful: {df_combined['Mutation (alternative)'].notna().sum()}/{len(df_combined)}")

df_screen = pd.read_excel(file_path, sheet_name="Localization screen results")

# Mutation == "reference"
step1 = df_screen[df_screen["Mutation"] == "reference"]

# Drop duplicates based on the "Gene" column
step2 = step1.drop_duplicates(subset="Gene")

# Only keep the "Gene" and "Primary location" columns
step3 = step2[["Gene", "Primary location"]]

# Rename the "Primary location" column to "wt_primary" 
df_wt = step3.rename(columns={"Primary location": "wt_primary"})

df_combined = df_combined.merge(df_wt, on="Gene", how="left")
print(f"\n{df_combined['wt_primary'].isna().sum()} rows have missing wt_primary values")

df_combined.to_csv(base_path / "cell2024_combined.csv", index=False)
print(f"Saved: {len(df_combined)} rows and {len(df_combined.columns)} columns")



# 2_uniprot.py

import requests
import pandas as pd
import time

base_path = "/mnt/volume6/czj/labLGN/LabLZ/"
df = pd.read_csv(base_path + "cell2024_combined.csv")

# Rows with non-null Uniprot IDs
df_with_uniprot = df[df["Uniprot"].notna()].copy()
uniprot_ids = df_with_uniprot["Uniprot"].unique()
print(f"Unique UniProt ID count: {len(uniprot_ids)}")

def fetch_sequence(uid):
    url = f"https://rest.uniprot.org/uniprotkb/{uid}.fasta"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            lines = r.text.strip().split("\n")
            return "".join(lines[1:])
        return None
    except:
        return None

sequences = {}
failed = []

for i, uid in enumerate(uniprot_ids):
    seq = fetch_sequence(uid)
    if seq:
        sequences[uid] = seq
    else:
        failed.append(uid)
    if i % 100 == 0:
        print(f"Progress: {i}/{len(uniprot_ids)}, Failed: {len(failed)}")
    time.sleep(0.1)

print(f"\nComplete: Successful {len(sequences)}, Failed {len(failed)}")

seq_df = pd.DataFrame(list(sequences.items()), columns=["Uniprot", "sequence"])
seq_df.to_csv(base_path + "uniprot_seq.csv", index=False)
print("Saved")

# Rows without

no_uniprot = df[df["Uniprot"].isna()]
genes_to_map = no_uniprot["Gene"].unique()
print(f" Without Uniprot IDs: {len(genes_to_map)} genes")

# Gene name → UniProt ID
def fetch_uniprot_id_by_gene(gene):
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true",
        "format": "tsv",
        "fields": "accession",
        "size": 1,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            lines = r.text.strip().split("\n")
            if len(lines) > 1:
                return lines[1].strip()
        return None
    except:
        return None

gene2id = {}
gene_id_failed = []
for i, gene in enumerate(genes_to_map):
    uid = fetch_uniprot_id_by_gene(gene)
    if uid:
        gene2id[gene] = uid
    else:
        gene_id_failed.append(gene)
    if i % 100 == 0:
        print(f"Gene→ID progress: {i}/{len(genes_to_map)}, Failed: {len(gene_id_failed)}")
    time.sleep(0.1)

print(f"\nGene→ID success: {len(gene2id)}, faliure: {len(gene_id_failed)}")

# Fetch
new_ids = [uid for uid in gene2id.values() if uid not in sequences]
print(f"IDs to fetch: {len(new_ids)}")

for i, uid in enumerate(new_ids):
    seq = fetch_sequence(uid)
    if seq:
        sequences[uid] = seq
    else:
        failed.append(uid)
    if i % 100 == 0:
        print(f"Seq progress: {i}/{len(new_ids)}, Failed total: {len(failed)}")
    time.sleep(0.1)

# Fill back UniProt ID
df["Uniprot"] = df["Uniprot"].fillna(df["Gene"].map(gene2id))
print(f"After filling, Uniprot missing: {df['Uniprot'].isna().sum()}")
df.to_csv(base_path + "cell2024_combined_filled.csv", index=False)

seq_df = pd.DataFrame(list(sequences.items()), columns=["Uniprot", "sequence"])
seq_df.to_csv(base_path + "uniprot_seq_2.csv", index=False)

print(f"Saved, All: {len(seq_df)} sequences")

still_na = df[df["Uniprot"].isna()]
print("Missing:", len(still_na))

is_fusion = still_na["Gene"].str.contains("-", na=False)
print("Fusion genes:", is_fusion.sum())
print("Non-fusion genes that failed:", sorted(still_na.loc[~is_fusion, "Gene"].unique()))
print(still_na["source"].value_counts())

# Merge
df = pd.read_csv(base_path + "cell2024_combined_filled.csv")
seq_df = pd.read_csv(base_path + "uniprot_seq_2.csv")

df = df.merge(seq_df, on="Uniprot", how="left")


print(f"With sequences: {df['sequence'].notna().sum()}")
print(f"Without sequences: {df['sequence'].isna().sum()}")

def parse_variant(variant_str):
    # K222E
    if not isinstance(variant_str, str):
        return None
    import re
    m = re.match(r'^([A-Z])(\d+)([A-Z])$', variant_str.strip())
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return None

def make_mutant_sequence(wt_seq, variant_str):
    # Substitute the amino acid at the specified position
    parsed = parse_variant(variant_str)
    if parsed is None:
        return None
    wt_aa, pos, mt_aa = parsed
    if pos < 1 or pos > len(wt_seq):
        return None  # out of bounds
    if wt_seq[pos - 1] != wt_aa:
        return None  # wild-type amino acid does not match
    return wt_seq[:pos - 1] + mt_aa + wt_seq[pos:]

df_seq = df[df['sequence'].notna()].copy()
df_seq['Mutation_final'] = df_seq["Mutation (alternative)"].fillna(df_seq["Mutation"])

# Important: only keep rows with a single mutation
df_seq['n_mut'] = df_seq['Mutation'].str.count(r'[A-Z]\d+[A-Z]')
n_before = len(df_seq)
df_seq = df_seq[df_seq['n_mut'] == 1].copy()
df_seq = df_seq.drop(columns=['n_mut'])
print(f"Exclude multi-mutation rows: {n_before - len(df_seq)} → Remaining: {len(df_seq)} (single mutation)")

def build_with_fallback(row):
    # try mutation and alternative mutation; use the one that passes validation
    for mut in [row['Mutation'], row['Mutation (alternative)']]:
        seq = make_mutant_sequence(row['sequence'], mut)
        if seq is not None:
            return pd.Series([seq, mut])
    return pd.Series([None, None])

df_seq[['mutant_sequence', 'Mutation_used']] = df_seq.apply(build_with_fallback, axis=1)

total = len(df_seq)
success = df_seq['mutant_sequence'].notna().sum()
failed_parse = df_seq['mutant_sequence'].isna().sum()

print(f"Success: {success}/{total}")
print(f"Failed rows: {failed_parse}")

failed_rows = df_seq[df_seq['mutant_sequence'].isna()][
    ['Gene', 'Mutation_final', 'sequence']].head(10)
print(failed_rows)


df = df.merge(
    df_seq[["Gene", "Variant", "Mutation_used", "mutant_sequence"]],
    on=["Gene", "Variant"], how="left")

print(f"All rows: {len(df)}")
print(f"Include: (with mutant_sequence): {df['mutant_sequence'].notna().sum()}")
print(f"Exclude rows (no mutant_sequence): {df['mutant_sequence'].isna().sum()}")
print(f"\nDistribution of Mislocalized for rows with mutant_sequence:")
print(df[df['mutant_sequence'].notna()]['Mislocalized'].value_counts())

df.to_csv(base_path + "cell2024_final.csv", index=False)
print(f"\nSaved cell2024_final.csv")




# 3_baseline.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import re, pickle
import torch, numpy as np, pandas as pd
from tqdm import tqdm
from transformers import EsmModel, EsmTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score

BASE_PATH  = "/mnt/volume6/czj/labLGN/LabLZ/"
MODEL_DIR  = BASE_PATH + "esm2_650M"
INPUT_CSV  = BASE_PATH + "cell2024_final.csv"
SCORE_CSV  = BASE_PATH + "phase3_esm2_scores.csv"
CACHE_PKL  = BASE_PATH + "esm2_emb_cache.pkl"           # embedding cache
MAX_LEN    = 1022
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
tokenizer = EsmTokenizer.from_pretrained(MODEL_DIR)
model = EsmModel.from_pretrained(MODEL_DIR).eval().to(device)
if device.type == "cuda":
    model = model.half()
print("Model loaded.")

# Get mutation position from row
def get_pos(row):
    for col in ["Mutation_used", "Mutation"]:
        m = re.match(r'^A-Z[A-Z]$', str(row.get(col, "")))
        if m:
            return int(m.group(1))
    return None

def window(seq, pos):
    # long protein sequences may exceed the model's max length, so we take a window around the mutation position
    if len(seq) <= MAX_LEN:
        return seq
    if pos is None:
        return seq[:MAX_LEN]  # if no mutation position, just take the first MAX_LEN characters
    end   = min(len(seq), pos + MAX_LEN // 2)
    start = max(0, end - MAX_LEN)
    return seq[start:start + MAX_LEN]

df = pd.read_csv(INPUT_CSV)
df_eval = df[df["mutant_sequence"].notna() & df["Mislocalized"].notna()].reset_index(drop=True)
df_eval["_pos"]   = df_eval.apply(get_pos, axis=1)
df_eval["_wtwin"] = df_eval.apply(lambda r: window(r["sequence"],        r["_pos"]), axis=1)
df_eval["_mtwin"] = df_eval.apply(lambda r: window(r["mutant_sequence"], r["_pos"]), axis=1)

n_long = (df_eval["sequence"].str.len() > MAX_LEN).sum()
print(f"Rows to be handled: {len(df_eval)}, from which sequences longer than {MAX_LEN} need a window: {n_long}")

# Get unique sequences to embed, so we don't embed the same sequence multiple times
unique_seqs = sorted(set(df_eval["_wtwin"]) | set(df_eval["_mtwin"]), key=len)

emb = {}
if os.path.exists(CACHE_PKL):
    with open(CACHE_PKL, "rb") as f:
        emb = pickle.load(f)
    print(f"We already have {len(emb)} embeddings cached.")
todo = [s for s in unique_seqs if s not in emb]
print(f"This time: {len(todo)}")

@torch.inference_mode()
def embed_batch(seqs):
    enc  = tokenizer(seqs, return_tensors="pt", padding=True, add_special_tokens=True)
    ids  = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    out  = model(input_ids=ids, attention_mask=mask).last_hidden_state.float()  # [B,L,H]
    m = mask.clone()
    m[:, 0] = 0                                        # delete CLS
    lengths = mask.sum(dim=1)
    m[torch.arange(m.size(0)), lengths - 1] = 0        # delete EOS
    m = m.unsqueeze(-1).float()
    pooled = (out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
    return pooled.cpu().numpy()

for i in tqdm(range(0, len(todo), BATCH_SIZE), desc="ESM2 embedding"):
    batch = todo[i:i + BATCH_SIZE]
    for s, v in zip(batch, embed_batch(batch)):
        emb[s] = v
    if (i // BATCH_SIZE) % 20 == 0:
        with open(CACHE_PKL, "wb") as f:
            pickle.dump(emb, f)
with open(CACHE_PKL, "wb") as f:
    pickle.dump(emb, f)
print(f"embedding finished, altogether {len(emb)} sequences embedded and cached.")

def cosine_distance(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return np.nan if d == 0 else 1.0 - np.dot(a, b) / d

df_eval["esm2_delta_score"] = df_eval.apply(
    lambda r: cosine_distance(emb[r["_wtwin"]], emb[r["_mtwin"]]), axis=1)
df_eval[["Gene", "Variant", "Mislocalized", "esm2_delta_score"]].to_csv(SCORE_CSV, index=False)
print(f"Saved {SCORE_CSV}, valid scores {df_eval['esm2_delta_score'].notna().sum()}/{len(df_eval)}")

print("\n─── baseline assessment with the same n (no addtional_benign) ───")
scored = df.merge(
    df_eval[["Gene", "Variant", "esm2_delta_score"]],
    on=["Gene", "Variant"], how="left")

mask = (scored["mutant_sequence"].notna()
        & scored["Mislocalized"].notna()
        & scored["AlphaMissense score"].notna()
        & scored["esm2_delta_score"].notna())
df_A = scored[mask].copy()
y = df_A["Mislocalized"].astype(int)

def report(name, scores):
    auc, auprc = roc_auc_score(y, scores), average_precision_score(y, scores)
    print(f"{name:15s} AUROC {auc:.4f}  AUPRC {auprc:.4f}")

report("AlphaMissense", df_A["AlphaMissense score"])
report("ESM2 delta",    df_A["esm2_delta_score"])
rng  = np.random.default_rng(0)
aucs = [roc_auc_score(y, rng.random(len(df_A))) for _ in range(10)]
print(f"{'Random':15s} AUROC {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

df_A = scored[mask].copy()
y = df_A["Mislocalized"].astype(int)

print(f"n = {len(df_A)}, positive {int(y.sum())}, negative {int((y == 0).sum())}")


# 3.5.1_classifier_label.py
import pandas as pd
BASE_PATH = "/mnt/volume6/czj/labLGN/LabLZ/"
df = pd.read_csv(BASE_PATH + "cell2024_final.csv")
df = df[df["mutant_sequence"].notna()].copy()
df_eval = df[df["Mislocalized"].notna()].copy()
df_pos = df_eval[df_eval["Mislocalized"] == 1]
phenotypes = df_pos["Mislocalization phenotype"].value_counts()
print(phenotypes.to_string())

def assign_class(phenotype):
    if not isinstance(phenotype, str):
        return None
    parts = phenotype.split(">")
    if len(parts) != 2:
        return None
    wt = parts[0].strip()
    mt = parts[1].strip()

    # C1：no relocation
    if wt == mt:
        return "C1_no_reloc"
    # C2：aggregation
    if mt in {"Foci", "Rods & rings"}:
        return "C2_aggregation"
    # C3：secretory pathway relocation
    if mt in {"ER", "Golgi apparatus", "Vesicles", "Plasma membrane",
              "Nuclear membrane", "Nuclear periphery"}:
        return "C3_secretory"
    # C4：nuclear relocation
    if mt in {"Nucleus", "Nucleolus", "Cytoplasm, nucleus"}:
        return "C4_nuclear"
    # C5： cytoplasmic relocation / others
    if mt in {"Cytoplasm", "Cellular periphery", "Mitochondria"}:
        return "C5_cytoplasmic"

    return "C_unknown"

df["label_5class"] = df["Mislocalization phenotype"].apply(assign_class)

# 验证
df_pos = df[df["Mislocalized"] == 1]
print("=== Five Classes ===")
print(df_pos["label_5class"].value_counts())

print(f"\nNot covered (C_unknown): {(df_pos['label_5class'] == 'C_unknown').sum()}")
print(f"label is None: {df_pos['label_5class'].isna().sum()}")

df_multiclass = df[
    (df["Mislocalized"] == 0) | 
    (df["label_5class"].notna())
].copy()

print(f"Available rows: {len(df_multiclass)}")
print(f"Positive: {(df_multiclass['Mislocalized']==1).sum()}")
print(f"Negative: {(df_multiclass['Mislocalized']==0).sum()}")

df.to_csv(BASE_PATH + "cell2024_final_with_labels.csv", index=False)
print("\nSaved to cell2024_final_with_labels.csv")




# 3.5.2_fetch_alphafold_pdb.py
import os
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_PATH, PDB_DIR = "/mnt/volume6/czj/labLGN/LabLZ/", "/mnt/volume6/czj/labLGN/LabLZ/alphafold_pdb/"
os.makedirs(PDB_DIR, exist_ok=True)
uniprot_ids = pd.read_csv(BASE_PATH + "cell2024_final.csv")["Uniprot"].dropna().unique()

def fetch(uid):
    out = PDB_DIR + f"{uid}.pdb"
    if os.path.exists(out):
        return None
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v6.pdb"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(out, "w") as f: f.write(r.text)
            return None
        return (uid, r.status_code)
    except Exception as e:
        return (uid, str(e))

failed = []
with ThreadPoolExecutor(max_workers=12) as ex:
    futs = [ex.submit(fetch, u) for u in uniprot_ids]
    for i, fut in enumerate(as_completed(futs)):
        r = fut.result()
        if r: failed.append(r)
        if i % 50 == 0: print(f"Progress: {i}/{len(uniprot_ids)}, failed {len(failed)}")

pd.DataFrame(failed, columns=["Uniprot", "Status"]).to_csv(BASE_PATH + "alphafold_failed.csv", index=False)
print(f"Finished. Failed: {len(failed)}")

import os
import re
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

import biotite.structure.io.pdb as pdb_io
import biotite.structure as struc

warnings.filterwarnings("ignore")


BASE_PATH  = "/mnt/volume6/czj/labLGN/LabLZ/"
PDB_DIR    = BASE_PATH + "alphafold_pdb/"
INPUT_CSV  = BASE_PATH + "cell2024_final_with_labels.csv"
OUTPUT_CSV = BASE_PATH + "phase35_struct_features.csv"
SAVE_EVERY = 100

TEST_MODE = False    # Test mode: only handle small subset of data, write to separate file
TEST_N    = 20

if TEST_MODE:
    OUTPUT_CSV = OUTPUT_CSV.replace(".csv", "_TEST.csv")
    SAVE_EVERY = 5


# Kyte-Doolittle hydrophobicity scale
KD_SCALE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}

# Tien et al. 2013 maximum ASA
MAX_ASA = {
    'A': 129.0, 'R': 274.0, 'N': 195.0, 'D': 193.0, 'C': 167.0,
    'Q': 225.0, 'E': 223.0, 'G': 104.0, 'H': 224.0, 'I': 197.0,
    'L': 201.0, 'K': 236.0, 'M': 224.0, 'F': 240.0, 'P': 159.0,
    'S': 155.0, 'T': 172.0, 'W': 285.0, 'Y': 263.0, 'V': 174.0,
}

# SAFETY: three aa to one aa mapping, defensive
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'MSE': 'M',
}

def three_to_one(resname):
    return THREE_TO_ONE.get(resname.strip().upper(), "X")

def parse_mutation(m):
    if not isinstance(m, str): return None, None, None
    mt = re.match(r'^([A-Z])(\d+)([A-Z])$', m.strip())
    return (mt.group(1), int(mt.group(2)), mt.group(3)) if mt else (None, None, None)

def compute_protein(uid):
    # AF PDB → {pos: (res_aa, plddt, sasa, ss)}
    path = PDB_DIR + f"{uid}.pdb"
    if not os.path.exists(path): 
        return uid, None
    try:
        structure = PDBParser(QUIET=True).get_structure(uid, path)
        ShrakeRupley().compute(structure, level="R")
        chain = list(structure[0].get_chains())[0]
        # SSE
        try:
            arr = pdb_io.PDBFile.read(path).get_structure(model=1)
            arr = arr[struc.filter_amino_acids(arr)]
            arr = arr[arr.chain_id == arr.chain_id[0]]
            m = {"a":"H","b":"E","c":"C"}
            sse = {int(r): m.get(s,"C") for r,s in zip(struc.get_residues(arr)[0], struc.annotate_sse(arr))}
        except Exception:
            sse = {}
        out = {}
        for res in chain.get_residues():
            rid = res.get_id()
            if rid[0] != " ": continue
            ca = [a for a in res.get_atoms() if a.get_name()=="CA"]
            plddt = (ca[0] if ca else next(iter(res.get_atoms()))).get_bfactor()
            out[rid[1]] = (three_to_one(res.get_resname()), plddt, getattr(res,"sasa",np.nan), sse.get(rid[1]))
        return uid, out
    except Exception as e:
        return uid, f"error:{str(e)[:50]}"


df = pd.read_csv(INPUT_CSV)
df_eval = df[df["mutant_sequence"].notna() & df["Mislocalized"].notna()].copy()
uids = [u for u in df_eval["Uniprot"].dropna().unique()]
print(f"unique proteins: {len(uids)}, rows: {len(df_eval)}")

prot = {}
with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
    for uid, feats in tqdm(ex.map(compute_protein, uids, chunksize=4), total=len(uids), desc="Computing protein features"):
        prot[uid] = feats

# Assemble results
rows = []
for _, row in df_eval.iterrows():
    uid = row.get("Uniprot")
    wt, pos, mt = parse_mutation(row.get("Mutation_used"))
    r = {"plddt":np.nan,"sasa":np.nan,"rsa":np.nan,"delta_hydrophobicity":np.nan,
         "ss_type":"unknown","ss_helix":np.nan,"ss_strand":np.nan,"ss_coil":np.nan,"struct_status":"ok"}
    if wt in KD_SCALE and mt in KD_SCALE:
        r["delta_hydrophobicity"] = KD_SCALE[mt] - KD_SCALE[wt]
    pf = prot.get(uid)
    if wt is None or pd.isna(uid):      r["struct_status"]="no_mutation_or_uniprot"
    elif pf is None:                    r["struct_status"]="no_pdb"
    elif isinstance(pf, str):           r["struct_status"]=pf
    elif pos not in pf:                 r["struct_status"]="pos_not_found"
    else:
        res_aa, plddt, sasa, ss = pf[pos]
        if res_aa != wt:                # Safety check: AF structure residue mismatch with WT
            r["struct_status"]=f"wt_mismatch(struct={res_aa},label={wt})"
        else:
            r["plddt"], r["sasa"] = plddt, sasa
            if not np.isnan(sasa) and wt in MAX_ASA: r["rsa"]=min(sasa/MAX_ASA[wt],1.0)
            if ss:  r.update(ss_type=ss, ss_helix=int(ss=="H"), ss_strand=int(ss=="E"), ss_coil=int(ss=="C"))
    rows.append({"Variant":row.get("Variant"),"Gene":row.get("Gene"),"Uniprot":uid,
                 "Mutation_used":row.get("Mutation_used"),"Mislocalized":row.get("Mislocalized"), **r})

pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
print("done", pd.DataFrame(rows)["struct_status"].value_counts().to_dict())


df_out = pd.read_csv(OUTPUT_CSV)              # 正常读，NaN 保持 NaN
print(f"Rows: {len(df_out)}")

print("\nstruct_status distribution:")
print(df_out["struct_status"].value_counts())

print("\nss_type distribution (H/E/C/unknown):")
print(df_out["ss_type"].value_counts())

for col in ["plddt", "sasa", "rsa", "delta_hydrophobicity"]:
    print(f"{col} not null: {df_out[col].notna().sum()}")

print("\nRange of numerical features:")
for col in ["plddt", "sasa", "rsa", "delta_hydrophobicity"]:
    s = df_out[col]
    print(f"{col:22s} min={s.min():.2f}  median={s.median():.2f}  max={s.max():.2f}")

import pandas as pd

BASE_PATH = "/mnt/volume6/czj/labLGN/LabLZ/"

df_main = pd.read_csv(BASE_PATH + "cell2024_final_with_labels.csv")
df_feat = pd.read_csv(BASE_PATH + "phase35_struct_features.csv")

# 2. Use the full Variant string as the join key 
feature_cols = [
    "plddt", "sasa", "rsa",
    "ss_type", "ss_helix", "ss_strand", "ss_coil",
    "delta_hydrophobicity",
    "struct_status",
]
df_feat_slim = df_feat[["Gene", "Variant"] + feature_cols].copy()

# 3. Verify Variant is unique in the feature table
dup = df_feat_slim[["Gene", "Variant"]].duplicated().sum()
if dup:
    print(f"WARNING: {dup} duplicate Variant(s) in the feature table — please investigate (keeping first).")
    df_feat_slim = df_feat_slim.drop_duplicates(["Gene", "Variant"], keep="first")

# 4. Left join: enforces uniqueness on the feature side.
df_merged = df_main.merge(df_feat_slim, on=["Gene", "Variant"], how="left", validate="m:1")

# 5. Sanity-check the merge
print(f"Main rows: {len(df_main)}, after merge: {len(df_merged)}")
print(f"Rows with structural features: {df_merged['plddt'].notna().sum()}")
print(f"Rows without matched features: {df_merged['struct_status'].isna().sum()}")

# 6. Merge
df_model = df_merged[
    df_merged["Mislocalized"].notna()
    & df_merged["mutant_sequence"].notna()
].copy()

print(f"\nSingle-substitution training set: {len(df_model)} rows")
print(df_model["Mislocalized"].value_counts())
print(f"Rows with structural features: {df_model['plddt'].notna().sum()}")

pos = (df_model["Mislocalized"] == 1).sum()
neg = (df_model["Mislocalized"] == 0).sum()
print(f"scale_pos_weight ~ {neg / pos:.2f}")

df_merged.to_csv(BASE_PATH + "cell2024_with_struct_features.csv", index=False)
df_model.to_csv(BASE_PATH + "cell2024_model_single_subst.csv", index=False)

