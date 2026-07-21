import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys
import numpy as np
import torch
from transformers import EsmModel, EsmTokenizer

sys.path.insert(0, "/mnt/volume6/czj/labLGN/LabLZ/models/PUPS")
sys.path.insert(0, "/mnt/volume6/czj/labLGN/LabLZ/models/PUPS/src")
from src.model.full_model import SubCellProtModel

# config
ESM_LOCAL   = "/mnt/volume6/czj/labLGN/LabLZ/models/esm2_650M"
CKPT        = "/mnt/volume6/czj/labLGN/LabLZ/models/PUPS/checkpoints/" \
              "splice_isoform_dataset_cell_line_and_gene_split_full-epoch=01-val_combined_loss=0.18.ckpt"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LEN = 2000          # token maximum (including BOS/EOS)
ESM_LAYER   = 33            # fetch the 33rd layer hidden states from ESM2-650M, as used in PUPS training
EMB_DIM     = 1280          # dimension of ESM2-650M hidden states, as used in PUPS training

CLASSES = [
    "Nucleoplasm","Cytosol","Vesicles","Plasma membrane","Mitochondria","Golgi apparatus",
    "Endoplasmic reticulum","Nucleoli","Nuclear bodies","Nuclear speckles","Nuclear membrane",
    "Peroxisomes","Microtubules","Centrosome","Cytokinetic bridge","Mitotic chromosome",
    "Centriolar satellite","Focal adhesion sites","Cell Junctions","Lipid droplets",
    "Nucleoli fibrillar center","Actin filaments","Mitotic spindle","Midbody ring",
    "Cytoplasmic bodies","Nucleoli rim","Midbody","Intermediate filaments","Aggresome",
]  # list for indexing the 29 compartments

# ESM2 encoder (lazy-loaded singleton)

# change aa seq into token; 
# if too long, window around mutation site; 
# then feed into ESM2-650M to get hidden states, return the tensor (1, 1, L+2, 1280)
class _ESM2Encoder:
    """Lazy-loaded singleton that caches the ESM2-650M tokenizer and model."""
    def __init__(self):
        self._tok = None
        self._esm = None

    @property
    def tok(self):
        if self._tok is None:
            self._load()
        return self._tok

    @property
    def esm(self):
        if self._esm is None:
            self._load()
        return self._esm

    def _load(self):
        self._tok = EsmTokenizer.from_pretrained(ESM_LOCAL)
        self._esm = EsmModel.from_pretrained(ESM_LOCAL).to(DEVICE).eval()

_encoder = _ESM2Encoder()


@torch.no_grad()
def esm2_encode(seq, pos=None):
    """Encode a protein sequence into per-residue ESM2-650M hidden states.

    The output format matches what PUPS was trained on: layer-33 hidden states
    with BOS/EOS tokens preserved, shape (1, 1, L'+2, 1280).

    Parameters
    ----------
    seq : str
        Amino acid sequence (uppercase single-letter codes).
    pos : int or None
        1-indexed mutation position.  When the sequence exceeds the token budget
        (MAX_SEQ_LEN), a window is centred on this position so the mutation site
        is never clipped.  Pass ``None`` to fall back to naive head truncation.

    Returns
    -------
    X : torch.Tensor  shape (1, 1, L'+2, 1280)
    x_len : torch.Tensor  shape (1,) — token count = L'+2
    """
    tok, esm = _encoder.tok, _encoder.esm
    cap = MAX_SEQ_LEN - 2          # max amino-acid residues (room for BOS/EOS)
    L = len(seq)

    if L > cap:
        # around the mutation site, wimdow the sequence to fit within the token budget
        if pos is None:
            start = 0
        else:
            start = min(max(pos - 1 - cap // 2, 0), L - cap)
        seq = seq[start:start + cap]

    enc = tok(seq, return_tensors="pt", add_special_tokens=True)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    out = esm(**enc, output_hidden_states=True)
    rep = out.hidden_states[ESM_LAYER][0]             # (L'+2, 1280)
    x_len = int(enc["attention_mask"].sum().item())   # = L'+2
    X = rep.unsqueeze(0).unsqueeze(0).float()          # (1, 1, L'+2, 1280)
    return X, torch.tensor([x_len], dtype=torch.float32, device=DEVICE)


# model
def load_model():
    """Load the PUPS SubCellProtModel from the paper's checkpoint."""
    model = SubCellProtModel.load_from_checkpoint(
        CKPT, map_location=DEVICE,
        intermediate_layer_size=300, multilabel_weight=1, embeddings_dim=EMB_DIM,
    )
    return model.to(DEVICE).eval()


@torch.no_grad()
def predict(model, seq, landmark, pos=None):
    """Run PUPS inference on a single protein sequence.

    Parameters
    ----------
    model : SubCellProtModel
    seq : str
        Amino acid sequence.
    landmark : np.ndarray  [3, 128, 128]  in [0, 1]
        Landmark stain image (nucleus / microtubules / third channel).
    pos : int or None
        1-indexed mutation position for long-sequence windowing.

    Returns
    -------
    image : np.ndarray  (128, 128)
        Predicted antibody stain.
    multilabel : np.ndarray  (29,)
        Sigmoid probabilities for each subcellular compartment.
    """
    X, x_len = esm2_encode(seq, pos=pos)  # seq into ESM2 embedding
    lm = torch.from_numpy(np.asarray(landmark, np.float32)).unsqueeze(0).to(DEVICE)  # landmark into tensor
    img, multi = model.call_model(X, x_len, lm)  # PUPS
    img   = img.squeeze().cpu().numpy()
    multi = torch.sigmoid(multi).squeeze(0).cpu().numpy()
    return img, multi  # return the predicted image and multilabel probabilities


def delta_report(model, wt_seq, mt_seq, landmark, pos=None, topk=5):
    """Print the top-k compartments whose probability shifts most between WT and MT."""
    _, p_wt = predict(model, wt_seq, landmark, pos=pos)
    _, p_mt = predict(model, mt_seq, landmark, pos=pos)
    d = p_mt - p_wt  # L1 norm
    order = np.argsort(-np.abs(d))[:topk]
    print("|Δ| L1 =", np.abs(d).sum())
    for i in order:
        print(f"  {CLASSES[i]:<24} WT={p_wt[i]:.3f} MT={p_mt[i]:.3f} Δ={d[i]:+.3f}")
    return d


if __name__ == "__main__":
    lm = np.load("/mnt/volume6/czj/labLGN/LabLZ/pups_trial/real_landmark.npy")
    m = load_model()
    print("device:", DEVICE)
    seq = "MATLKDQLIYNLLKEEQTPQNKITVVGVGAVGMACAISILMKDLADELALVDVIEDKLKGEMMDLQHGSLFLRTPKIVSGKDYNVTANSKLVIITAGARQQEGESRLNLVQRNVNIFKFIIPNVVKYSPNCKLLIVSNPVDILTYVAWKISGFPKNRVIGSGCNLDSARFRYLMGERLGVHPLSCHGWVLGEHGDSSVPVWSGMNVAGVSLKTLHPDLGTDKDKEQWKEVHKQVVESAYEVIKLKGYTSWAIGLSVADLAESIMKNLRRVHPVSTMIKGLYGIKDDVFLSVPCILGQNGISDLVKVTLTSEEEARLKKSADTLWGIQKELQF"
    img, multi = predict(m, seq, lm)
    print("image:", img.shape, "multilabel:", multi.shape,
          "top:", CLASSES[int(multi.argmax())], float(multi.max()))
