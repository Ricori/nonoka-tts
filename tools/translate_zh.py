"""
README 6.2 method 3, step 1: translate her 498 Japanese lines (02_dataset_full/esd.list) into spoken Chinese with
Gemini (Antigravity CLI `agy`, text only), so the synthetic Chinese training data keeps her tone and topics.

  python tools\\translate_zh.py [--batch 25] [--jobs 10]

Output: 02_dataset_zh/_translate.csv (id, ja, zh). Re-running only translates the ids that are still missing.
"""
import argparse
import csv
import json
import re
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "02_dataset_full" / "esd.list"
OUT = ROOT / "02_dataset_zh" / "_translate.csv"

SCHEMA = {
    "type": "object",
    "properties": {"lines": {"type": "array", "items": {
        "type": "object", "properties": {"id": {"type": "string"}, "zh": {"type": "string"}}, "required": ["id", "zh"]}}},
    "required": ["lines"],
}

PROMPT = """Translate these Japanese lines into natural spoken Mandarin Chinese (simplified characters).
They are all said by the same character: a polite, gentle, slightly shy high-school girl talking to her senpai.
The Chinese will be read aloud by a TTS system, so:
- Keep her tone: polite but warm, small hesitations and stutters kept (e.g. 「い、いやいや」 -> 「不、不不」), sighs and
  laughs written as Chinese interjections (嗯、啊、呜、嘿嘿、哈哈). No Japanese-only words.
- 先輩 -> 学长. Drop personal names (奏、夜月、十和、カナデ etc.): 「奏先輩」 -> 「学长」.
- Write every number in Chinese characters, no Arabic numerals, no Latin letters.
- Punctuation only ，。！？…、 (no quotes, brackets or ~).
- Translate the meaning naturally, not word for word; keep about the same length.
Lines (id: text):
{items}
Answer only with JSON matching the schema, one entry per id."""


def run_agy(items, model, timeout):
    lines = "\n".join(f"{i}: {t}" for i, t in items)
    cmd = ["agy", "-p", PROMPT.format(items=lines), "--model", model, "--output-format", "json",
           "--json-schema", json.dumps(SCHEMA), "--print-timeout", f"{timeout}s",
           "--dangerously-skip-permissions", "--sandbox"]
    with tempfile.TemporaryDirectory() as work:
        p = subprocess.run(cmd, cwd=work, capture_output=True, text=True, encoding="utf-8", timeout=timeout + 60)
    raw = p.stdout.strip()
    if "quota" in raw.lower() or "quota" in p.stderr.lower():
        raise SystemExit(f"agy quota reached:\n{(raw or p.stderr)[-500:]}")
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
            if isinstance(v, dict) and "lines" in v:
                return v["lines"]
    raise RuntimeError(f"no JSON answer (exit {p.returncode}):\n{raw[-1500:]}\n{p.stderr[-1500:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--model", default="gemini-3.8-flash-low")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    src = [l.split("|") for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = {fn.rsplit(".", 1)[0]: {"id": fn.rsplit(".", 1)[0], "ja": text, "zh": ""} for fn, _, _, text in src}
    if OUT.exists():
        for r in csv.DictReader(open(OUT, encoding="utf-8-sig")):
            if r["id"] in rows:
                rows[r["id"]]["zh"] = r["zh"]
    todo = [r for r in rows.values() if not r["zh"]]
    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    print(f"{len(todo)} lines in {len(batches)} calls, {args.jobs} at a time", flush=True)
    OUT.parent.mkdir(exist_ok=True)
    lock = threading.Lock()

    def save():
        with open(OUT, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["id", "ja", "zh"]); w.writeheader(); w.writerows(rows.values())

    def work_on(k, batch):
        for attempt in (1, 2):
            try:
                ans = run_agy([(r["id"], r["ja"]) for r in batch], args.model, args.timeout)
                break
            except RuntimeError as e:  # one retry, then leave these lines for the next run
                print(f"!! batch {k} attempt {attempt}: {e}", flush=True)
        else:
            return
        by = {a["id"].strip(): a["zh"].strip() for a in ans}
        with lock:
            got = 0
            for r in batch:
                zh = re.sub(r"\s+", "", by.get(r["id"], ""))
                if zh:
                    r["zh"] = zh; got += 1
            save()
            print(f"batch {k}: {got}/{len(batch)}", flush=True)

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        list(ex.map(lambda kb: work_on(*kb), enumerate(batches)))
    left = sum(not r["zh"] for r in rows.values())
    print(f"ALL DONE; {left} lines still untranslated")


if __name__ == "__main__":
    main()
