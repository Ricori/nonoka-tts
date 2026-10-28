"""
Let Gemini (Antigravity CLI `agy`) listen to the clips compare_full.py flagged and settle their transcripts.

  python tools\\gemini_proofread.py [--priority 高,中] [--batch 5] [--jobs 4]

Each call gets a few clips (as WAV) with the large-v3 and kotoba transcripts and returns the text that matches
what is actually spoken. The answer goes to the `gemini` / `gemini_note` columns of 02_dataset_full/_proofread.csv;
`text` (and esd.list) take Gemini's text. Re-running skips clips that already have an answer.
NOTE: the audio is uploaded to Google; tools are auto-approved with agy's terminal sandbox on.
"""
import argparse
import csv
import json
import re
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "02_dataset_full"
CSV = D / "_proofread.csv"

SCHEMA = {
    "type": "object",
    "properties": {"clips": {"type": "array", "items": {
        "type": "object",
        "properties": {"file": {"type": "string"}, "text": {"type": "string"}, "note": {"type": "string"}},
        "required": ["file", "text", "note"]}}},
    "required": ["clips"],
}

PROMPT = """You are proofreading Japanese transcripts for a TTS training dataset.
The current directory holds {n} WAV files, all lines spoken by the same anime-style young female character:
{items}

For each file, two speech recognizers produced candidate transcripts (A = whisper large-v3, B = kotoba-whisper).
They can both be wrong. Open and listen to EVERY file with read_file, then write the transcript that matches
exactly what she says:
- Keep stutters, repeated words and fillers as actually spoken (e.g. 「い、いやいや」).
- Use normal Japanese orthography (kanji + kana) as in candidate A. Keep A's punctuation (、。?！…) unless it is
  clearly wrong: a rising question intonation keeps its question mark. Only change words you actually hear differently,
  and make sure the corrected words make sense in context.
- Character names: keep the spelling of candidate A (e.g. 奏先輩, 夜月先輩, とわ先輩) unless the sound clearly differs.
- If A is already correct, return it unchanged.
These files are all there is; there is no need to list the directory or run any shell command.
For each file give the final text and a short note in Chinese (max 30 characters) on what you changed, or "无改动".
Answer only with JSON matching the schema."""


def run_agy(work, items, model, timeout):
    lines = "\n".join(f"- {f}\n  A: {a}\n  B: {b}" for f, a, b in items)
    cmd = ["agy", "-p", PROMPT.format(n=len(items), items=lines), "--model", model, "--output-format", "json",
           "--json-schema", json.dumps(SCHEMA), "--print-timeout", f"{timeout}s",
           "--dangerously-skip-permissions", "--sandbox"]
    p = subprocess.run(cmd, cwd=work, capture_output=True, text=True, encoding="utf-8", timeout=timeout + 60)
    raw = p.stdout.strip()
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
            if isinstance(v, dict) and "clips" in v:
                return v["clips"]
    raise RuntimeError(f"no JSON answer (exit {p.returncode}):\n{raw[-1500:]}\n{p.stderr[-1500:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--priority", default="高,中")
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--model", default="gemini-3.8-flash-low")
    ap.add_argument("--timeout", type=int, default=480)
    args = ap.parse_args()
    import soundfile as sf

    rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
    for r in rows:
        r.setdefault("gemini", ""); r.setdefault("gemini_note", "")
    todo = [r for r in rows if r["priority"] in args.priority.split(",") and not r["gemini"]]
    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    print(f"{len(todo)} clips in {len(batches)} calls, {args.jobs} at a time", flush=True)
    lock = threading.Lock()

    def save():
        with open(CSV, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        with open(D / "esd.list", "w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(f"{r['file'][:-4]}.wav|nonoka|JP|{r['text']}\n" for r in rows)

    def work_on(k, batch):
        work = D / "_gemini_work" / f"b{k:03d}"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        items = []
        for r in batch:
            x, sr = sf.read(D / r["file"], dtype="float32")
            name = r["file"][:-4] + ".wav"
            sf.write(work / name, x, sr)
            items.append((name, r["large_v3"], r["kotoba"]))
        ans = None
        for attempt in (1, 2):
            try:
                ans = run_agy(work, items, args.model, args.timeout)
                break
            except Exception as e:  # noqa: BLE001 - one retry, then leave these clips for the next run
                print(f"!! batch {k} attempt {attempt}: {e}", flush=True)
        shutil.rmtree(work, ignore_errors=True)
        if not ans:
            return
        by = {a["file"].strip()[:8]: a for a in ans}
        with lock:
            for r in batch:
                a = by.get(r["file"][:8])
                if a and a["text"].strip():
                    r["gemini"], r["gemini_note"] = a["text"].strip(), a["note"].strip()
                    r["text"] = re.sub(r"\s+", "", r["gemini"].replace("?", "？").replace("!", "！"))
                    print(f"{r['file']} [{r['gemini_note']}] {r['text']}", flush=True)
            save()

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        list(ex.map(lambda kb: work_on(*kb), enumerate(batches)))
    shutil.rmtree(D / "_gemini_work", ignore_errors=True)
    left = sum(r["priority"] in args.priority.split(",") and not r["gemini"] for r in rows)
    print(f"ALL DONE; {left} flagged clips still without an answer")


if __name__ == "__main__":
    main()
