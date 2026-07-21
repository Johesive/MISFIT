import re, requests, numpy as np
from io import BytesIO
from PIL import Image

WWW = "www.proteinatlas.org"
IMG = "images.proteinatlas.org"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}

# download one image
def _get(url):
    return requests.get(url, headers=HEADERS, timeout=120)

def load_channel(url, size=128):
    r = _get(url)
    if r.status_code != 200 or r.content[:2] != b"\xff\xd8":   # check if it is JPEG
        raise RuntimeError(f"No image {url}\n status={r.status_code} "
                           f"ctype={r.headers.get('content-type')} head={r.content[:80]!r}")
    im = Image.open(BytesIO(r.content)).convert("L")           # gray scale
    return np.asarray(im.resize((size, size)), np.float32) / 255.0   # -> [0,1]

def _stack(urls, size=128):
    lm = np.stack([load_channel(urls["blue"],   size),   # nucleus (DAPI)
                   load_channel(urls["red"],    size),   # microtubules (anti-tubulin)
                   load_channel(urls["yellow"], size)],  # third channel (inside PUPS is mitochondria_channel,
                  axis=0).astype(np.float32)             # actual download addresses match training) [3,128,128]
    return lm

# method A: 
def build_from_prefix(prefix, size=128):
    urls = {c: prefix + "_" + c + ".jpg" for c in ["blue", "red", "green", "yellow"]}
    return _stack(urls, size)

# method B:
def get_channel_urls(ensembl_id):
    r = _get("https://" + WWW + "/" + ensembl_id + ".xml")
    print("XML status:", r.status_code, "len:", len(r.text))
    pat = r"https://" + IMG.replace(".", r"\.") + r"/\S+?_(?:blue|red|green|yellow)"
    hits = re.findall(pat, r.text)
    print("hit number:", len(hits))
    if not hits:
        raise RuntimeError("XML finds no IF image, change into build_from_prefix()")
    base = re.match(r"^(.*?)_(?:blue|red|green|yellow)", hits[0] + "_blue").group(1)
    print("base prefix:", base)
    return {c: base + "_" + c + ".jpg" for c in ["blue", "red", "green", "yellow"]}

def build_landmark(ensembl_id, size=128):
    return _stack(get_channel_urls(ensembl_id), size)

if __name__ == "__main__":
    # method A: use confirmed working prefix (4 channels all return 200)
    PREFIX = "https://" + IMG + "/43542/1486_A1_1"
    lm = build_from_prefix(PREFIX)

    # backup：lm = build_landmark("ENSG00000141076")

    np.save("/mnt/volume6/czj/labLGN/LabLZ/pups_trial/real_landmark.npy", lm)
    print("landmark:", lm.shape, "scale:", round(float(lm.min()), 4), "~", round(float(lm.max()), 4))
    print("Mean for [nucleus/microtubules/ER]:", [round(float(lm[i].mean()), 4) for i in range(3)])