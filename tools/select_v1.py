import csv, os, shutil, numpy as np
H=os.path.expanduser("~/work"); SRC=os.path.expanduser("~/mnt/v")
a={r["file"]:r for r in csv.DictReader(open(f"{H}/scores.csv",encoding="utf-8"))}
for r in csv.DictReader(open(f"{H}/scores2.csv",encoding="utf-8")): a[r["file"]].update(r)
R=list(a.values())
num=["dur","speech_db","long_pause","level_std","resid_noise_db","centroid","hf_db","f0_med","f0_sd_semi"]
for r in R:
    for k in num: r[k]=float(r[k])
def pct(k,q): return np.percentile([r[k] for r in R],q)
lim=dict(sp=(pct("speech_db",10),pct("speech_db",90)), f0=(pct("f0_med",10),pct("f0_med",85)),
         cen=(pct("centroid",5),pct("centroid",95)), hf=(pct("hf_db",5),pct("hf_db",95)),
         lv=pct("level_std",90), rn=pct("resid_noise_db",90))
med={k:np.median([r[k] for r in R]) for k in num}
iqr={k:np.subtract(*np.percentile([r[k] for r in R],[75,25])) or 1 for k in num}
for r in R:
    why=[]
    if not 7<=r["dur"]<=15: why.append("时长不在7-15秒")
    if not lim["sp"][0]<=r["speech_db"]<=lim["sp"][1]: why.append("音量偏离")
    if r["long_pause"]>1.5: why.append("长停顿>1.5秒")
    if r["resid_noise_db"]>lim["rn"]: why.append("残余噪音偏高")
    if not lim["f0"][0]<=r["f0_med"]<=lim["f0"][1]: why.append("音高偏离(喊叫/过低)")
    if not (lim["cen"][0]<=r["centroid"]<=lim["cen"][1] and lim["hf"][0]<=r["hf_db"]<=lim["hf"][1]): why.append("音色偏闷或偏尖")
    if r["level_std"]>lim["lv"]: why.append("音量起伏大")
    r["excluded_reason"]="；".join(why)
    # typicality: distance from median style + noise penalty
    r["score"]=round(sum(abs(r[k]-med[k])/iqr[k] for k in ["f0_med","centroid","speech_db","f0_sd_semi","level_std"])
                     +0.5*max(0,(r["resid_noise_db"]-med["resid_noise_db"])/iqr["resid_noise_db"]),3)
ok=[r for r in R if not r["excluded_reason"]]
bins=[(7,9),(9,11),(11,13),(13,15.01)]
cnt=[sum(lo<=r["dur"]<hi for r in ok) for lo,hi in bins]
quota=[25]*4
sel=[]
for (lo,hi),q in zip(bins,quota):
    b=sorted([r for r in ok if lo<=r["dur"]<hi],key=lambda r:r["score"]); sel+=b[:q]
if len(sel)<100:
    rest=sorted([r for r in ok if r not in sel],key=lambda r:r["score"]); sel+=rest[:100-len(sel)]
S={r["file"] for r in sel}
for r in R: r["selected"]="Y" if r["file"] in S else ""
print("通过筛选:",len(ok),"各时长段:",cnt,"选中:",len(sel))
from collections import Counter
c=Counter(); [c.update(r["excluded_reason"].split("；")) for r in R if r["excluded_reason"]]; print(c.most_common())
print("选中时长分布:",[sum(lo<=r["dur"]<hi for r in sel) for lo,hi in bins], "总时长(分):",round(sum(r["dur"] for r in sel)/60,1))
out=os.path.join(SRC,"_select_100"); os.makedirs(out,exist_ok=True)
for r in sel: shutil.copy2(os.path.join(SRC,r["file"]),out)
keys=["file","selected","score","excluded_reason"]+num+["peak_db","clip_ratio","lead_sil","trail_sil","inner_sil","zero_frac"]
R.sort(key=lambda r:(r["selected"]!="Y",r["excluded_reason"]!="",r["score"]))
with open(os.path.join(SRC,"_scores.csv"),"w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=keys,extrasaction="ignore"); w.writeheader(); w.writerows(R)
with open(os.path.join(out,"_list.txt"),"w",encoding="utf-8") as f: f.write("\n".join(sorted(S)))
print("中位音高Hz:",round(med["f0_med"]),"选中音高范围:",round(min(r["f0_med"] for r in sel)),"-",round(max(r["f0_med"] for r in sel)))
