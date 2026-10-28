r"""
README 6.2 method 3, step 3: check the synthetic Chinese clips (02_dataset_zh/<id>.ogg from zh_vc.py::corpus_main)
with Whisper large-v3 and keep only the ones that say what the text says.

  03_asr\venv\Scripts\python tools\filter_zh.py [--max-cer 0.1] [--max-sec 25]

Output: 02_dataset_zh/_filter.csv (every clip: duration, transcript, CER, kept) and 02_dataset_zh/esd.list
(kept clips, `<id>.wav|nonoka|ZH|<text>`, the format train_gptsovits.py reads). Run with the 03_asr venv.
"""
import argparse
import csv
import glob
import os
import re
import site
import sys
import unicodedata
from math import gcd
from pathlib import Path

for sp in set(site.getsitepackages() + [site.getusersitepackages()]):  # CUDA DLLs from the pip nvidia packages
    for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
        os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
        try: os.add_dll_directory(d)
        except Exception: pass

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from scipy.signal import resample_poly

D = Path(__file__).resolve().parent.parent / "02_dataset_zh"
DIG = "零一二三四五六七八九"


def load_audio(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != 16000:
        g = gcd(sr, 16000); x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    return x, len(x) / 16000


def num_zh(n):  # Whisper sometimes writes Arabic numerals; the texts never do
    if n >= 10000: return "".join(DIG[int(c)] for c in str(n))
    out = ""
    for unit, k in ((1000, "千"), (100, "百"), (10, "十")):
        q, n = divmod(n, unit)
        if q: out += ("" if q == 1 and k == "十" and not out else DIG[q]) + k
    return out + (DIG[n] if n else "") or "零"


def norm(text):
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"\d+", lambda m: num_zh(int(m.group())), t)
    return "".join(c for c in t if unicodedata.category(c)[0] in "LN")


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cer", type=float, default=0.1)
    ap.add_argument("--max-sec", type=float, default=25)
    args = ap.parse_args()
    texts = {r["id"]: r["zh"] for r in csv.DictReader(open(D / "_translate.csv", encoding="utf-8-sig"))}
    clips = sorted(p for p in D.glob("*.ogg") if p.stem in texts)
    print(f"{len(clips)} clips", flush=True)
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    rows = []
    for k, p in enumerate(clips, 1):
        audio, sec = load_audio(p)
        segs, _ = model.transcribe(audio, language="zh", beam_size=5, condition_on_previous_text=False,
                                   temperature=0.0, initial_prompt="以下是普通话的句子。")
        hyp = "".join(s.text.strip() for s in segs)
        ref = texts[p.stem]
        rn, hn = norm(ref), norm(hyp)
        cer = edit_distance(rn, hn) / max(len(rn), 1)
        keep = cer <= args.max_cer and sec <= args.max_sec
        rows.append(dict(id=p.stem, sec=round(sec, 2), cer=round(cer, 4), keep=int(keep), text=ref, hyp=hyp))
        print(f"{k}/{len(clips)} {p.stem} {sec:5.1f}s cer={cer:.3f} {'' if keep else 'DROP '}{hyp}", flush=True)
    with open(D / "_filter.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    kept = [r for r in rows if r["keep"]]
    with open(D / "esd.list", "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(f"{r['id']}.wav|nonoka|ZH|{r['text']}\n" for r in kept)
    print(f"kept {len(kept)}/{len(rows)} ({sum(r['sec'] for r in kept) / 60:.1f} min); "
          f"mean CER {sum(r['cer'] for r in rows) / len(rows):.2%}\nALL DONE")


if __name__ == "__main__":
    sys.exit(main())
