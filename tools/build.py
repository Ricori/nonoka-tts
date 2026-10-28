import json, os, csv
D=os.path.expanduser("~/mnt/v/_select_100")
rows=json.load(open(os.path.expanduser("~/work/rows.json")))
excl=[r for r in rows if r["file"].startswith("031")]
rows=[r for r in rows if not r["file"].startswith("031")]
for r in rows:  # recompute: normal slow speech is fine, only flag extreme
    why=[w for w in r["reason"].split("；") if w and not w.startswith("语速异常")]
    if r["cps"]<2.0: why.append(f"语速异常({r['cps']}字/秒)")
    r["reason"]="；".join(why)
    r["priority"]="高" if any(("不一致" in w) or ("置信度" in w) or ("重复" in w) for w in why) else ("中" if why else "")
from collections import Counter; print(Counter(r["priority"] for r in rows), len(rows))
# draft esd.list
with open(f"{D}/esd.list","w",encoding="utf-8",newline="\n") as f:
    for r in rows: f.write(f"{r['file'][:-4]}.wav|nonoka|JP|{r['text']}\n")
with open(f"{D}/_proofread.csv","w",encoding="utf-8-sig",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["file","priority","reason","text","large_v3","kotoba","dur"],extrasaction="ignore"); w.writeheader(); w.writerows(rows)
with open(f"{D}/_excluded.txt","w",encoding="utf-8") as f:
    f.write("以下10条为031系列（亲密场景台词：喘息、拖长音、含糊发音多），不适合朗读型训练，已从本轮验证中排除：\n"+"\n".join(r["file"] for r in excl)+"\n")
data=[{k:r[k] for k in ["file","priority","reason","text","kotoba","dur"]} for r in rows]
html=open(os.path.expanduser("~/work/tpl.html"),encoding="utf-8").read().replace("__DATA__",json.dumps(data,ensure_ascii=False))
open(f"{D}/_proofread.html","w",encoding="utf-8").write(html)
print("ok")
