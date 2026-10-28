"""
Let Gemini (through the Antigravity CLI `agy`) blind-judge the test samples, one call per sentence.

  python gemini_judge.py [--rounds 2] [--model gemini-3.8-flash-low] [--only 1,5] [--jobs 10] [--fresh]

Each call gets 3 original clips as the voice reference and the 5 checkpoint versions of one sentence,
shuffled and renamed A, B, C... (a different order every round, to average out position bias).
NOTE: the audio is uploaded to Google with each call. Tools are auto-approved (--dangerously-skip-permissions)
with agy's terminal sandbox on; pass --confirm to turn that off.
Output: out/gemini_judge.csv (one row per round x sentence x version); run make_blind.py afterwards.
Re-running resumes: (round, sentence) pairs already in the csv are kept and skipped, unless --fresh.
The low thinking level is the default (fast, light on the agy quota); when the quota runs out the run stops early.
"""
import argparse
import csv
import json
import random
import re
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
DATASET = HERE.parent / "02_dataset_v1"
REFS = ["03002860", "03003430", "03004600"]  # same reference clips as the blind-test page
DIMS = ["similarity", "pronunciation", "naturalness", "stability"]
LANGS = {
    "ja": {"language": "Japanese", "pron": "kanji readings and Japanese pitch accent are correct for the target text (names included)"},
    "zh": {"language": "Mandarin Chinese", "pron": "every character is pronounced correctly with correct Mandarin tones, no foreign accent"},
}
LETTERS = "ABCDEFGHIJKLMNOP"


TARGET_DBFS = -23.0  # systems differ in output level (SBV2 peak-normalized ~8 dB hotter); louder wins listening tests


def copy_leveled(src, dst):
    """Copy audio as WAV at a common loudness (speech RMS), so no version is judged better just for being louder."""
    import numpy as np
    import soundfile as sf

    x, sr = sf.read(src, dtype="float32", always_2d=True)
    x = x.mean(1)
    voiced = x[np.abs(x) > 1e-4]
    rms = np.sqrt(np.mean((voiced if voiced.size > sr // 10 else x) ** 2)) + 1e-9
    y = x * (10 ** (TARGET_DBFS / 20) / rms)
    y /= max(1.0, np.abs(y).max() / 0.99)  # never clip
    sf.write(dst, y, sr)


def step_of(name):  # SBV2 checkpoints end in _s<step>; other systems (e.g. sarashina_*) get 0
    m = re.search(r"_s(\d+)$", name)
    return int(m.group(1)) if m else 0

SCHEMA = {
    "type": "object",
    "properties": {
        "versions": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                **{d: {"type": "integer", "minimum": 1, "maximum": 5} for d in DIMS},
                "issues": {"type": "array", "items": {"type": "string", "enum": ["misread", "accent", "skip_or_repeat", "noise", "unnatural_pause", "other"]}},
                "comment": {"type": "string"},
            },
            "required": ["label", *DIMS, "issues", "comment"],
        }},
        "best": {"type": "string"},
    },
    "required": ["versions", "best"],
}

PROMPT = """You are an expert evaluator of {language} text-to-speech.
Files in the current directory:
- ref_1.wav, ref_2.wav, ref_3.wav: real recordings of the target speaker (an anime-style young female character).
  They are in Japanese and may differ from the target language; use them only as the reference for voice and style.
- {files}: {n} TTS outputs from different models/checkpoints, all reading the same target text.

Target text: {text}

These {total} files are all there is; there is no need to list the directory or run any shell command.
Open and listen to EVERY file with read_file (do not rely on metadata or file sizes). Listen to the references first.
Then score each of {labels} from 1 (bad) to 5 (excellent):
- similarity: how close the timbre and speaking style are to the reference speaker
- pronunciation: {pron}
- naturalness: intonation, pauses, rhythm and breath sound like a real person
- stability: free of noise, buzzing, glitches, skipped or repeated syllables
Compare the {n} versions against each other and use the full scale; do not give everyone the same score.
List concrete issues and write a short comment in Chinese (max 40 characters) for each version, e.g. which word was misread.
Also name the single best version. Answer only with JSON matching the schema."""


def run_agy(workdir, text, n, model, timeout, skip_permissions, lang):
    labels = LETTERS[:n]
    cmd = ["agy", "-p", PROMPT.format(text=text, n=n, **LANGS[lang], total=n + len(REFS), labels=", ".join(labels),
                                       files=" ".join(f"{l}.wav" for l in labels)), "--model", model, "--output-format", "json",
           "--json-schema", json.dumps(SCHEMA), "--print-timeout", f"{timeout}s"]
    if skip_permissions:  # headless mode can't prompt; auto-approve tools, terminal stays sandboxed
        cmd += ["--dangerously-skip-permissions", "--sandbox"]
    p = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, encoding="utf-8", timeout=timeout + 60)
    raw = p.stdout.strip()
    # the CLI may wrap the answer in an envelope; find the object that holds "versions"
    for cand in [raw] + re.findall(r"\{.*\}", raw, re.S):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        for v in (obj, obj.get("result"), obj.get("structured_output"), obj.get("response")):
            if isinstance(v, str):
                try:
                    v = json.loads(v)
                except json.JSONDecodeError:
                    continue
            if isinstance(v, dict) and "versions" in v:
                return v
    raise RuntimeError(f"no JSON answer (exit {p.returncode}):\n{raw[-2000:]}\n{p.stderr[-2000:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--root", default="out", help="folder holding samples/, e.g. out/zh for the Chinese set")
    ap.add_argument("--lang", choices=sorted(LANGS), default="ja")
    ap.add_argument("--model", default="gemini-3.8-flash-low")
    ap.add_argument("--only", default="", help="comma-separated sentence numbers, e.g. 1,5")
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--jobs", type=int, default=10, help="parallel agy calls")
    ap.add_argument("--fresh", action="store_true", help="start over instead of resuming from gemini_judge.csv")
    ap.add_argument("--confirm", action="store_true",
                    help="don't pass --dangerously-skip-permissions (tools needing approval are then denied)")
    args = ap.parse_args()
    root = (HERE / args.root).resolve()

    ckpts = sorted((d for d in (root / "samples").iterdir() if (d / "sentences.txt").exists()),
                   key=lambda d: (step_of(d.name) == 0, step_of(d.name), d.name))
    sentences = (ckpts[0] / "sentences.txt").read_text(encoding="utf-8").splitlines()
    only = {int(x) for x in args.only.split(",") if x}
    out_csv = root / "gemini_judge.csv"
    rows, lock, quota = [], threading.Lock(), threading.Event()
    if out_csv.exists() and not args.fresh:  # resume: keep finished (round, sentence) calls
        with open(out_csv, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                rows.append({**r, **{k: int(r[k]) for k in ("round", "idx", "best", *DIMS)}})
    done = {(r["round"], r["idx"]) for r in rows}

    def judge(rnd, i, text):
        if quota.is_set():
            return
        order = [c.name for c in ckpts]
        random.Random(rnd * 1000 + i).shuffle(order)
        work = root / "gemini_work" / f"r{rnd}_{i:02d}"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        for k, r in enumerate(REFS, 1):
            copy_leveled(DATASET / f"{r}.ogg", work / f"ref_{k}.wav")
        for label, ck in zip(LETTERS, order):
            copy_leveled(root / "samples" / ck / f"{i:02d}.wav", work / f"{label}.wav")
        ans = None
        for attempt in (1, 2):
            try:
                ans = run_agy(work, text, len(order), args.model, args.timeout, not args.confirm, args.lang)
                break
            except Exception as e:  # noqa: BLE001 - one retry, then give up on this sentence
                if "quota reached" in str(e).lower():
                    if not quota.is_set():
                        print(f"!! agy quota used up, stopping: {str(e).split('Resets in')[-1][:20].strip()} until reset", flush=True)
                    quota.set()
                    break
                print(f"!! round {rnd} sentence {i} attempt {attempt}: {e}", flush=True)
        shutil.rmtree(work, ignore_errors=True)
        if not ans:
            return
        by_label = {v["label"].strip().upper()[:1]: v for v in ans["versions"]}
        new = [dict(round=rnd, idx=i, label=label, checkpoint=ck, **{d: by_label[label][d] for d in DIMS},
                    issues=" ".join(by_label[label].get("issues", [])), comment=by_label[label].get("comment", ""),
                    best=int(ans.get("best", "").strip().upper()[:1] == label))
               for label, ck in zip(LETTERS, order) if label in by_label]
        with lock:
            rows.extend(new)
            print(f"round {rnd} sentence {i:2d}: best={ans.get('best')} -> "
                  + ", ".join(f"{l}={by_label[l]['similarity']}/{by_label[l]['pronunciation']}/{by_label[l]['naturalness']}/{by_label[l]['stability']}"
                              for l in LETTERS if l in by_label), flush=True)
            rows.sort(key=lambda r: (r["round"], r["idx"], r["label"]))
            with open(out_csv, "w", newline="", encoding="utf-8-sig") as fh:  # rewrite as we go
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    jobs = [(rnd, i, text) for rnd in range(1, args.rounds + 1)
            for i, text in enumerate(sentences, 1) if (not only or i in only) and (rnd, i) not in done]
    if done:
        print(f"resuming: {len(done)} calls already in {out_csv.name}", flush=True)
    print(f"{len(ckpts)} versions x {len(jobs)} calls, {args.jobs} at a time", flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        list(ex.map(lambda j: judge(*j), jobs))
    shutil.rmtree(root / "gemini_work", ignore_errors=True)

    print(f"\n{'':22s} {'sim':>5s} {'pron':>5s} {'nat':>5s} {'stab':>5s} {'avg':>5s} {'best':>5s}  n")
    for c in ckpts:
        r = [x for x in rows if x["checkpoint"] == c.name]
        if not r:
            continue
        m = {d: sum(x[d] for x in r) / len(r) for d in DIMS}
        print(f"{c.name:22s} " + " ".join(f"{m[d]:5.2f}" for d in DIMS)
              + f" {sum(m.values()) / 4:5.2f} {sum(x['best'] for x in r):5d}  {len(r)}")
    print(f"-> {out_csv}\nALL DONE")


if __name__ == "__main__":
    main()
