import os, sys, glob, csv, site, time

# Make pip-installed NVIDIA CUDA DLLs (cuBLAS / cuDNN) visible on Windows
for sp in set(site.getsitepackages() + [site.getusersitepackages()]):
    for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
        os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
        try: os.add_dll_directory(d)
        except Exception: pass

from faster_whisper import WhisperModel
import numpy as np, soundfile as sf
from math import gcd
from scipy.signal import resample_poly

def load_audio(path):
    # decode ourselves (avoids PyAV/faster-whisper version mismatch)
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != 16000:
        g = gcd(sr, 16000); x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    return x

audio_dir = os.path.abspath(sys.argv[1])
files = sorted(glob.glob(os.path.join(audio_dir, "*.ogg")) + glob.glob(os.path.join(audio_dir, "*.wav")))
print(f"{len(files)} files in {audio_dir}")

MODELS = [("large-v3", "large-v3"),
          ("kotoba", "kotoba-tech/kotoba-whisper-v2.0-faster")]
PROMPT = "はい、よろしくお願いします。今日は、いい天気ですね。"

def load(name):
    try:
        return WhisperModel(name, device="cuda", compute_type="float16")
    except Exception as e:
        print("!! GPU load failed, falling back to CPU (slow):", e)
        return WhisperModel(name, device="cpu", compute_type="int8")

for tag, name in MODELS:
    out = os.path.join(audio_dir, f"_asr_{tag}.csv")
    print(f"\n=== {tag}: loading {name} (first run downloads the model) ===")
    try:
        model = load(name)
    except Exception as e:
        print(f"!! skip {tag}: {e}"); continue
    t0 = time.time(); rows = []
    for i, f in enumerate(files, 1):
        segs, info = model.transcribe(load_audio(f), language="ja", beam_size=5,
                                      condition_on_previous_text=False,
                                      initial_prompt=PROMPT if tag == "large-v3" else None)
        segs = list(segs)
        text = "".join(s.text.strip() for s in segs)
        rows.append(dict(file=os.path.basename(f), text=text,
                         dur=round(info.duration, 2),
                         min_logprob=round(min((s.avg_logprob for s in segs), default=-9), 3),
                         max_nospeech=round(max((s.no_speech_prob for s in segs), default=1), 3),
                         max_compress=round(max((s.compression_ratio for s in segs), default=0), 2),
                         n_seg=len(segs)))
        print(f"[{i}/{len(files)}] {os.path.basename(f)}  {text}")
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"=== {tag} done in {time.time()-t0:.0f}s -> {out}")
    del model
print("\nALL DONE")
