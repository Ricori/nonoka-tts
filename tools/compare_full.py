"""
Cross-check the two Whisper transcripts of 02_dataset_full and draft its esd.list.

  03_asr\\venv\\Scripts\\python tools\\compare_full.py

Clips already proofread in round 1 (02_dataset_v1/esd.list) keep their checked text. For the rest, the large-v3
text is the draft; both transcripts are turned into hiragana and compared, and low agreement, low confidence,
repetition or an odd speaking rate marks a clip for review (priority 高 / 中).
Writes 02_dataset_full/_proofread.csv (all clips) and a draft 02_dataset_full/esd.list.
"""
import csv
import difflib
import re
from pathlib import Path

import pykakasi

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "02_dataset_full"
V1 = {l.split("|")[0][:-4] + ".ogg": l.split("|")[3]
      for l in (ROOT / "02_dataset_v1" / "esd.list").read_text(encoding="utf-8").splitlines() if l.strip()}

kks = pykakasi.kakasi()
PUN = re.compile(r"[、。，,．.!！?？…・「」『』（）()\s〜~ー-]")


def kana(t):
    return PUN.sub("", "".join(x["hira"] for x in kks.convert(t)))


def clean(t):
    return re.sub(r"\s+", "", t.replace("?", "？").replace("!", "！").replace("...", "…"))


if __name__ == "__main__":
    read = lambda p: {r["file"]: r for r in csv.DictReader(open(p, encoding="utf-8-sig"))}
    L, K = read(D / "_asr_large-v3.csv"), read(D / "_asr_kotoba.csv")
    rows = []
    for f in sorted(L):
        l, k = L[f], K.get(f, {"text": ""})
        a, b = kana(l["text"]), kana(k["text"])
        sim = difflib.SequenceMatcher(None, a, b).ratio()
        dur = float(l["dur"])
        cps = len(a) / dur if dur else 0
        why = []
        if f in V1:
            prio, text, why = "", V1[f], ["第一轮已人工校对"]
        else:
            text = clean(l["text"])
            if sim < 0.9: why.append(f"两模型读音不一致({sim:.0%})")
            elif sim < 0.97: why.append(f"读音小差异({sim:.0%})")
            if float(l["min_logprob"]) < -0.35: why.append("置信度偏低")
            if float(l["max_compress"]) > 2.0: why.append("疑似重复/幻觉")
            if cps < 2.0 or cps > 11: why.append(f"语速异常({cps:.1f}字/秒)")
            high = sim < 0.9 or float(l["min_logprob"]) < -0.35 or float(l["max_compress"]) > 2 or cps < 2.0 or cps > 11
            prio = "高" if high else ("中" if why else "")
        rows.append(dict(file=f, priority=prio, reason="；".join(why), text=text, large_v3=l["text"], kotoba=k["text"],
                         dur=dur, kana_sim=round(sim, 3), cps=round(cps, 1)))

    with open(D / "_proofread.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(D / "esd.list", "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(f"{r['file'][:-4]}.wav|nonoka|JP|{r['text']}\n" for r in rows)
    from collections import Counter
    print(len(rows), "clips;", Counter(r["priority"] or "-" for r in rows), "; round-1 checked:", sum(r["file"] in V1 for r in rows))
