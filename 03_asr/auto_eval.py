r"""
Objective evaluation of the TTS test samples, all local (CPU is fine).

  venv\Scripts\python auto_eval.py [samples_dir] [dataset_dir]
      defaults: ..\04_train\out\samples   ..\02_dataset_v1

Per sample:
  sim      speaker similarity: cosine between the sample's ECAPA embedding (speechbrain/spkrec-ecapa-voxceleb)
           and the centroid of all dataset clips
  mos      SQUIM subjective MOS (torchaudio), averaged over 3 dataset clips as non-matching references
  dnsmos   DNSMOS P.835 overall quality (Microsoft, ONNX) - drops with noise, buzz and artifacts
  f0_med   median pitch (Hz) and
  f0_st    pitch variation (std of voiced frames, in semitones) - low means flat intonation
  mora_s   speaking rate: morae per second of speech (leading/trailing silence trimmed)
The dataset clips are scored the same way as the baseline (sim is leave-one-out for them).
Output: <samples_dir>/../auto_eval.csv
"""
import csv, os, re, sys, unicodedata
from pathlib import Path

import librosa
import numpy as np
import pykakasi
import soundfile as sf
import torch
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
from speechmos import dnsmos
from torchaudio.pipelines import SQUIM_SUBJECTIVE

here = Path(__file__).resolve().parent
samples = Path(sys.argv[1] if len(sys.argv) > 1 else here.parent / "04_train" / "out" / "samples").resolve()
dataset = Path(sys.argv[2] if len(sys.argv) > 2 else here.parent / "02_dataset_v1").resolve()
NMR = ["03002860", "03003430", "03004600"]  # same reference clips as the blind-test page
SR = 16000
torch.set_num_threads(max(1, os.cpu_count() - 2))

KKS = pykakasi.kakasi()
SMALL = set("ゃゅょぁぃぅぇぉゎ")


def morae(text):
    t = unicodedata.normalize("NFKC", text)
    t = "".join(c for c in t if unicodedata.category(c)[0] in "LN")
    kana = "".join(x["hira"] for x in KKS.convert(t))
    return sum(1 for c in kana if c not in SMALL)


def load(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return librosa.resample(x.mean(1), orig_sr=sr, target_sr=SR) if sr != SR else x.mean(1)


print("loading models (first run downloads ~100 MB)")
ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                       savedir=str(here / "models" / "ecapa"), local_strategy=LocalStrategy.COPY)
squim = SQUIM_SUBJECTIVE.get_model().eval()
nmr = [torch.from_numpy(load(dataset / f"{n}.ogg"))[None] for n in NMR]


@torch.inference_mode()
def measure(wav, text, own_ref=None):
    t = torch.from_numpy(wav)[None]
    emb = ecapa.encode_batch(t).squeeze()
    emb = emb / emb.norm()
    refs = [r for n, r in zip(NMR, nmr) if n != own_ref]
    mos = float(np.mean([squim(t, r).item() for r in refs]))
    dns = dnsmos.run(np.clip(wav, -1, 1), SR)["ovrl_mos"]  # resampling can overshoot slightly
    f0, voiced, _ = librosa.pyin(wav, fmin=120, fmax=900, sr=SR, frame_length=1024)
    f0 = f0[voiced & ~np.isnan(f0)]
    st = 12 * np.log2(f0 / np.median(f0)) if len(f0) else np.array([0.0])
    trimmed, _ = librosa.effects.trim(wav, top_db=40)
    return emb, dict(mos=round(mos, 3), dnsmos=round(float(dns), 3),
                     f0_med=round(float(np.median(f0)) if len(f0) else 0.0, 1), f0_st=round(float(st.std()), 2),
                     mora_s=round(morae(text) / (len(trimmed) / SR), 2))


# dataset baseline
esd = [l.split("|") for l in (dataset / "esd.list").read_text(encoding="utf-8").splitlines() if l.strip()]
base, base_emb = [], []
for k, (fn, _, _, text) in enumerate(esd, 1):
    stem = fn.rsplit(".", 1)[0]
    emb, m = measure(load(dataset / f"{stem}.ogg"), text, own_ref=stem)
    base_emb.append(emb)
    base.append(dict(checkpoint="原声", step=0, idx=stem, text=text, **m))
    print(f"\r dataset {k}/{len(esd)}", end="", flush=True)
print()
E = torch.stack(base_emb)
total = E.sum(0)
for row, e in zip(base, E):  # leave-one-out centroid
    c = total - e
    row["sim"] = round(float(e @ (c / c.norm())), 4)
centroid = total / total.norm()

rows = []
def step_of(name):  # SBV2 checkpoints end in _s<step>; other systems (e.g. sarashina_*) get 0
    m = re.search(r"_s(\d+)$", name)
    return int(m.group(1)) if m else 0


ckpts = sorted((d for d in samples.iterdir() if (d / "sentences.txt").exists()),
               key=lambda d: (step_of(d.name) == 0, step_of(d.name), d.name))
for d in ckpts:
    lines = (d / "sentences.txt").read_text(encoding="utf-8").splitlines()
    for i, text in enumerate(lines, 1):
        emb, m = measure(load(d / f"{i:02d}.wav"), text)
        rows.append(dict(checkpoint=d.name, step=step_of(d.name), idx=i, text=text,
                         sim=round(float(emb @ centroid), 4), **m))
    print(" scored", d.name)

out = samples.parent / "auto_eval.csv"
fields = ["checkpoint", "step", "idx", "text", "sim", "mos", "dnsmos", "f0_med", "f0_st", "mora_s"]
with open(out, "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(base + rows)

print(f"\n{'':22s} {'sim':>6s} {'MOS':>5s} {'DNSMOS':>6s} {'F0Hz':>6s} {'F0st':>5s} {'mora/s':>6s}   worst sim (idx)")
for name, r in [("原声 (baseline)", base)] + [(d.name, [x for x in rows if x["checkpoint"] == d.name]) for d in ckpts]:
    avg = lambda k: np.mean([x[k] for x in r])
    worst = min(r, key=lambda x: x["sim"])
    print(f"{name:22s} {avg('sim'):6.3f} {avg('mos'):5.2f} {avg('dnsmos'):6.2f} {avg('f0_med'):6.0f} {avg('f0_st'):5.2f} {avg('mora_s'):6.2f}   {worst['sim']:.3f} ({worst['idx']})")
print(f"-> {out}\nALL DONE")
