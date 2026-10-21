"""
Build out/blind.html, the results page for the downloaded test samples: every version side by side (not blind;
no scoring), with CER, auto_eval metrics and Gemini scores/comments.

  python make_blind.py        (run again after re-rendering samples or after cer.csv changes)
  python make_blind.py out/zh (same page for another sample set, e.g. the Chinese Qwen3-TTS samples)

Reads out/samples/<checkpoint>/sentences.txt and, if present, out/cer.csv (from 03_asr/run_cer.bat).
Scores are kept in the browser's localStorage; a new checkpoint set or new sentences start a fresh sheet.
"""
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = (HERE / sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "out"
# a few clean original lines from the dataset, played as the "what she really sounds like" reference
REFS = ["03002860", "03003430", "03004600"]


def step_of(name):  # SBV2 checkpoints end in _s<step>; other systems (e.g. sarashina_*) get 0
    m = re.search(r"_s(\d+)$", name)
    return int(m.group(1)) if m else 0


ckpts = sorted((d.name for d in (OUT / "samples").iterdir() if (d / "sentences.txt").exists()),
               key=lambda n: (step_of(n) == 0, step_of(n), [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", n)]))
sentences = (OUT / "samples" / ckpts[0] / "sentences.txt").read_text(encoding="utf-8").splitlines()
for ck in ckpts:
    assert (OUT / "samples" / ck / "sentences.txt").read_text(encoding="utf-8").splitlines() == sentences, \
        f"{ck} was rendered with different sentences - run 2_render.bat again"

dataset = HERE.parent / "02_dataset_v1"
esd = {l.split("|")[0].rsplit(".", 1)[0]: l.split("|")[3] for l in (dataset / "esd.list").read_text(encoding="utf-8").splitlines() if l.strip()}


def volume(path, target_dbfs=-24.0):
    """Playback volume (<= 1) that brings a file's speech RMS down to a common level: SBV2 peak-normalizes to
    0 dBFS, ~8 dB louder than GPT-SoVITS and the originals, and louder always sounds 'better' in a blind test."""
    import numpy as np
    import soundfile as sf

    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    voiced = x[np.abs(x) > 1e-4]
    rms_db = 20 * np.log10(np.sqrt(np.mean((voiced if voiced.size > sr // 10 else x) ** 2)) + 1e-9)
    return round(float(min(1.0, 10 ** ((target_dbfs - rms_db) / 20))), 3)


refs = [{"src": Path(os.path.relpath(dataset / f"{r}.ogg", OUT)).as_posix(), "text": esd[r],
         "vol": volume(dataset / f"{r}.ogg")} for r in REFS]
vols = {ck: {i: volume(OUT / "samples" / ck / f"{i:02d}.wav") for i in range(1, len(sentences) + 1)} for ck in ckpts}

cer = None
if (OUT / "cer.csv").exists():
    cer = {}
    with open(OUT / "cer.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            c = cer.setdefault(row["checkpoint"], {"items": {}, "edits": 0, "len": 0})
            c["items"][int(row["idx"])] = {"hyp": row["hyp"], "cer": float(row["cer"])}
            c["edits"] += int(row["edits"])
            c["len"] += int(row["ref_len"])
    for c in cer.values():
        c["total"] = c.pop("edits") / max(c.pop("len"), 1)

auto = None
if (OUT / "auto_eval.csv").exists():
    groups = {}
    with open(OUT / "auto_eval.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            groups.setdefault(row["checkpoint"], []).append(row)
    metrics = ["sim", "mos", "dnsmos", "f0_st", "mora_s"]
    auto = [{"name": name, **{m: sum(float(r[m]) for r in rs) / len(rs) for m in metrics}}
            for name, rs in groups.items()]

gemini = None
if (OUT / "gemini_judge.csv").exists():
    dims = ["similarity", "pronunciation", "naturalness", "stability"]
    with open(OUT / "gemini_judge.csv", encoding="utf-8-sig") as f:
        grows = list(csv.DictReader(f))
    gemini = {"summary": [], "items": {}}
    for ck in ckpts:
        rs = [r for r in grows if r["checkpoint"] == ck]
        if rs:
            gemini["summary"].append({"name": ck, "n": len(rs), "best": sum(int(r["best"]) for r in rs),
                                      **{d: sum(int(r[d]) for r in rs) / len(rs) for d in dims}})
        for r in rs:
            gemini["items"].setdefault(ck, {}).setdefault(r["idx"], []).append(
                f"{'/'.join(r[d] for d in dims)} {r['comment']}")

key = hashlib.sha1(json.dumps([ckpts, sentences], ensure_ascii=False).encode()).hexdigest()[:10]
data = {"key": key, "sentences": sentences, "ckpts": [{"name": c, "step": step_of(c)} for c in ckpts], "refs": refs, "vols": vols, "cer": cer, "auto": auto, "gemini": gemini}
html = (HERE / "blind_tpl.html").read_text(encoding="utf-8").replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
(OUT / "blind.html").write_text(html, encoding="utf-8")
print(f"{len(ckpts)} checkpoints x {len(sentences)} sentences, CER {'included' if cer else 'not available yet'}, "
      f"auto metrics {'included' if auto else 'not available yet'}, Gemini {'included' if gemini else 'not available yet'} -> {OUT / 'blind.html'}")
