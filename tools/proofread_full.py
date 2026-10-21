"""
Proofreading page for 02_dataset_full, and merging its result back into esd.list.

  03_asr\\venv\\Scripts\\python tools\\proofread_full.py          -> 02_dataset_full/_proofread.html
  03_asr\\venv\\Scripts\\python tools\\proofread_full.py apply    <- 02_dataset_full/proofread_result.csv

Text per clip: round-1 clips keep their hand-checked text; clips where large-v3 and kotoba agree keep large-v3;
clips where they disagree take Gemini's text (tools/gemini_proofread.py) as the draft. The page lists the
disagreeing clips plus agreeing ones whose wording Gemini changed; `apply` writes the checked text (and drops)
into esd.list. Run after compare_full.py and gemini_proofread.py.
"""
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "02_dataset_full"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_full import V1, clean, kana  # noqa: E402  (compare_full rewrites its outputs on import)

AGREE = 0.97
rows = list(csv.DictReader(open(D / "_proofread.csv", encoding="utf-8-sig")))


def draft(r):
    """(text, shown on the page?)"""
    if r["file"] in V1:
        return V1[r["file"]], False
    g = r.get("gemini", "")
    agree = float(r["kana_sim"]) >= AGREE
    g_changed = bool(g) and kana(g) != kana(r["large_v3"])
    if agree and not g_changed:
        return clean(r["large_v3"]), False
    return (clean(g) if g else clean(r["large_v3"])), True


def write_esd(texts, drop=()):
    with open(D / "esd.list", "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(f"{f[:-4]}.wav|nonoka|JP|{t}\n" for f, t in texts.items() if f not in drop)


texts, page = {}, []
for r in rows:
    t, show = draft(r)
    texts[r["file"]] = t
    if show:
        why = [] if float(r["kana_sim"]) >= AGREE else [f"两模型不一致({float(r['kana_sim']):.0%})"]
        page.append(dict(file=r["file"], priority="高" if why else "中", reason="；".join(why), text=t,
                         large_v3=r["large_v3"], kotoba=r["kotoba"], dur=r["dur"], note=r.get("gemini_note", ""),
                         gemini_changed=bool(r.get("gemini")) and kana(r["gemini"]) != kana(r["large_v3"])))

if sys.argv[1:] == ["apply"]:
    res = list(csv.DictReader(open(D / "proofread_result.csv", encoding="utf-8-sig")))
    drop = {x["file"] for x in res if x["status"] == "drop"}
    for x in res:
        texts[x["file"]] = re.sub(r"[\r\n|]+", "", x["text"]).strip()
    todo = sum(x["status"] == "todo" for x in res)
    write_esd(texts, drop)
    print(f"esd.list: {len(texts) - len(drop)} clips ({len(res)} from the page, {len(drop)} dropped, {todo} left unconfirmed)")
else:
    write_esd(texts)
    tpl = (Path(__file__).resolve().parent / "tpl_full.html").read_text(encoding="utf-8")
    (D / "_proofread.html").write_text(tpl.replace("__DATA__", json.dumps(page, ensure_ascii=False)), encoding="utf-8")
    print(f"{len(page)} clips to proofread -> {D / '_proofread.html'}; draft esd.list: {len(texts)} clips")
