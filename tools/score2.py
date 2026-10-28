import sys, os, glob, csv
import numpy as np, soundfile as sf
src, out = sys.argv[1], sys.argv[2]
FR=0.02; rows=[]
for p in sorted(glob.glob(os.path.join(src,"*.ogg"))):
    x, sr = sf.read(p, dtype="float32", always_2d=True); x = x.mean(1)
    n=int(sr*FR); m=len(x)//n; fr=x[:m*n].reshape(m,n)
    db=20*np.log10(np.sqrt((fr**2).mean(1))+1e-9)
    hi=np.percentile(db,95); act=db>hi-30
    zero=float((db<-150).mean())
    nz=db[(db>-150)&(~act)]
    resid=float(np.percentile(nz,50)) if len(nz)>5 else -150.0   # residual noise in non-silent pauses
    # spectral features over active frames
    win=np.hanning(n); S=np.abs(np.fft.rfft(fr[act]*win,axis=1))**2
    f=np.fft.rfftfreq(n,1/sr)
    cen=float(((S*f).sum(1)/(S.sum(1)+1e-12)).mean())
    hf=float(10*np.log10(S[:,f>8000].sum()/(S.sum()+1e-12)+1e-12))
    # crude f0 via autocorrelation on voiced active frames (40ms)
    n2=int(sr*0.04); m2=len(x)//n2; fr2=x[:m2*n2].reshape(m2,n2)
    d2=20*np.log10(np.sqrt((fr2**2).mean(1))+1e-9); f0s=[]
    lo,hi_=int(sr/600),int(sr/70)
    for seg in fr2[d2>hi-20]:
        seg=seg-seg.mean(); ac=np.correlate(seg,seg,'full')[n2-1:]
        if ac[0]<=0: continue
        k=lo+np.argmax(ac[lo:hi_]); 
        if ac[k]/ac[0]>0.5: f0s.append(sr/k)
    f0=float(np.median(f0s)) if f0s else 0; f0sd=float(np.std(12*np.log2(np.array(f0s)/f0))) if len(f0s)>3 else 0
    rows.append(dict(file=os.path.basename(p),zero_frac=round(zero,3),resid_noise_db=round(resid,1),
        centroid=round(cen),hf_db=round(hf,1),f0_med=round(f0,1),f0_sd_semi=round(f0sd,2),voiced_frames=len(f0s)))
with open(out,"w",newline="",encoding="utf-8") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("done",len(rows))
