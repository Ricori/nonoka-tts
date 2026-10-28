r"""
Transcribe the TTS test samples back with Whisper large-v3 and compute CER per checkpoint.

  venv\Scripts\python cer_samples.py [samples_dir]      (default: ..\04_train\out\samples)
  venv\Scripts\python cer_samples.py ..\04_train\out\zh\samples --zh   (Chinese: plain character CER, no kana)

Layout expected: <samples_dir>/<checkpoint>/NN.wav + sentences.txt (written by train_gptsovits.py).
Both sides are normalized (NFKC, punctuation removed, Arabic numerals -> kanji numerals) and
converted to hiragana with pykakasi before comparison, so kanji/kana spelling choices don't count
as errors. Output: <samples_dir>/../cer.csv (one row per sample).
"""
import os, sys, glob, csv, re, site, unicodedata

for sp in set(site.getsitepackages() + [site.getusersitepackages()]):
    for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
        os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
        try: os.add_dll_directory(d)
        except Exception: pass

import pykakasi
import numpy as np, soundfile as sf
from math import gcd
from scipy.signal import resample_poly
from faster_whisper import WhisperModel

KKS = pykakasi.kakasi()


def load_audio(path):
    # same decoder as transcribe.py (avoids PyAV/faster-whisper version mismatch)
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != 16000:
        g = gcd(sr, 16000); x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    return x
DIG = "〇一二三四五六七八九"


def kanji_num(n):
    if n == 0: return "〇"
    if n >= 10000: return "".join(DIG[int(c)] for c in str(n))
    out = ""
    for unit, k in ((1000, "千"), (100, "百"), (10, "十")):
        q, n = divmod(n, unit)
        if q: out += ("" if q == 1 else DIG[q]) + k
    return out + (DIG[n] if n else "")


ZH = "--zh" in sys.argv
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]


def to_kana(text):
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"\d+", lambda m: kanji_num(int(m.group())), t)
    t = "".join(c for c in t if unicodedata.category(c)[0] in "LN")  # drop punctuation, spaces, symbols
    return t if ZH else "".join(x["hira"] for x in KKS.convert(t))


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def step_of(name):  # SBV2 checkpoints end in _s<step>; other systems (e.g. sarashina_*) get 0
    m = re.search(r"_s(\d+)$", name)
    return int(m.group(1)) if m else 0


here = os.path.dirname(os.path.abspath(__file__))
samples = os.path.abspath(ARGS[0] if ARGS else os.path.join(here, "..", "04_train", "out", "samples"))
ckpts = sorted((d for d in os.listdir(samples) if os.path.isfile(os.path.join(samples, d, "sentences.txt"))),
               key=lambda n: (step_of(n) == 0, step_of(n), n))
print(f"{len(ckpts)} checkpoints in {samples}")

try:
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
except Exception as e:
    print("!! GPU load failed, falling back to CPU (slow):", e)
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")

rows = []
for ck in ckpts:
    lines = open(os.path.join(samples, ck, "sentences.txt"), encoding="utf-8").read().splitlines()
    for i, ref in enumerate(lines, 1):
        wav = os.path.join(samples, ck, f"{i:02d}.wav")
        segs, _ = model.transcribe(load_audio(wav), language="zh" if ZH else "ja", beam_size=5,
                                   condition_on_previous_text=False, temperature=0.0,
                                   initial_prompt="以下是普通话的句子。" if ZH else None)  # nudges toward simplified characters
        hyp = "".join(s.text.strip() for s in segs)
        rk, hk = to_kana(ref), to_kana(hyp)
        ed = edit_distance(rk, hk)
        rows.append(dict(checkpoint=ck, step=step_of(ck), idx=i, ref=ref, hyp=hyp,
                         ref_kana=rk, hyp_kana=hk, edits=ed, ref_len=len(rk), cer=round(ed / max(len(rk), 1), 4)))
        print(f"{ck} {i:02d} cer={rows[-1]['cer']:.3f}  {hyp}")

out = os.path.join(os.path.dirname(samples), "cer.csv")
with open(out, "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

print("\n== CER per checkpoint (total edits / total kana)")
for ck in ckpts:
    r = [x for x in rows if x["checkpoint"] == ck]
    bad = [f"{x['idx']:02d}" for x in r if x["edits"]]
    print(f"{ck:22s} {sum(x['edits'] for x in r) / sum(x['ref_len'] for x in r):.2%}   sentences with errors: {' '.join(bad) or '-'}")
print(f"-> {out}\nALL DONE")
