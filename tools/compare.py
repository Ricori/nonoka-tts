import csv, os, re, difflib, json
import pykakasi
D=os.path.expanduser("~/mnt/v/_select_100")
L={r["file"]:r for r in csv.DictReader(open(f"{D}/_asr_large-v3.csv",encoding="utf-8-sig"))}
K={r["file"]:r for r in csv.DictReader(open(f"{D}/_asr_kotoba.csv",encoding="utf-8-sig"))}
kks=pykakasi.kakasi()
PUN=re.compile(r"[、。，,．.!！?？…・「」『』（）()\s〜~ー-]")
def kana(t): return PUN.sub("","".join(x["hira"] for x in kks.convert(t)))
def clean(t):
    t=t.replace("?","？").replace("!","！").replace("...","…")
    t=re.sub(r"\s+","",t)
    return t
rows=[]
for f in sorted(L):
    l,k=L[f],K.get(f,{"text":""})
    a,b=kana(l["text"]),kana(k["text"])
    sim=difflib.SequenceMatcher(None,a,b).ratio()
    dur=float(l["dur"]); cps=len(a)/dur if dur else 0
    why=[]
    if sim<0.9: why.append(f"两模型读音不一致({sim:.0%})")
    elif sim<0.97: why.append(f"读音小差异({sim:.0%})")
    if float(l["min_logprob"])<-0.35: why.append("置信度偏低")
    if float(l["max_compress"])>2.0: why.append("疑似重复/幻觉")
    if cps<3.5 or cps>11: why.append(f"语速异常({cps:.1f}字/秒)")
    lvl="高" if (sim<0.9 or float(l["min_logprob"])<-0.35 or float(l["max_compress"])>2) else ("中" if why else "")
    rows.append(dict(file=f,priority=lvl,reason="；".join(why),text=clean(l["text"]),large_v3=l["text"],kotoba=k["text"],
                     dur=dur,kana_sim=round(sim,3),cps=round(cps,1)))
from collections import Counter
print(Counter(r["priority"] for r in rows))
for r in rows:
    if r["priority"]: print(r["priority"],r["file"],r["reason"],"\n   L:",r["large_v3"],"\n   K:",r["kotoba"])
json.dump(rows,open(os.path.expanduser("~/work/rows.json"),"w"),ensure_ascii=False)
