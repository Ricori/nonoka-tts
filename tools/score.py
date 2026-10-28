import sys, os, glob, csv
import numpy as np, soundfile as sf

src = sys.argv[1]; out = sys.argv[2]
FR = 0.02  # 20ms frames
rows = []
for p in sorted(glob.glob(os.path.join(src, "*.ogg"))):
    try:
        x, sr = sf.read(p, dtype="float32", always_2d=True)
    except Exception as e:
        rows.append(dict(file=os.path.basename(p), error=str(e))); continue
    x = x.mean(1)
    dur = len(x) / sr
    peak = float(np.abs(x).max()) + 1e-12
    clip = float((np.abs(x) >= 0.99).mean())
    n = int(sr * FR); m = len(x) // n
    fr = x[: m * n].reshape(m, n)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9)
    hi = np.percentile(db, 95)
    floor = np.percentile(db, 10)
    snr = hi - floor
    act = db > hi - 30          # speech-active frames
    idx = np.where(act)[0]
    lead = idx[0] * FR if len(idx) else dur
    trail = (m - 1 - idx[-1]) * FR if len(idx) else dur
    inner = act[idx[0]: idx[-1] + 1] if len(idx) else act
    inner_sil = float(1 - inner.mean()) if len(inner) else 1.0
    # longest internal pause
    run = best = 0
    for a in inner:
        run = 0 if a else run + 1; best = max(best, run)
    long_pause = best * FR
    speech_db = float(np.mean(db[act])) if act.any() else -99
    # level fluctuation across speech frames (std of 0.5s-window loudness)
    w = 25; seg = db[act]
    var = float(np.std([seg[i:i+w].mean() for i in range(0, max(len(seg)-w,1), w)])) if len(seg) > w else 0.0
    rows.append(dict(file=os.path.basename(p), dur=round(dur,2), peak_db=round(20*np.log10(peak),1),
        clip_ratio=round(clip,5), snr_db=round(snr,1), noise_floor_db=round(floor,1),
        speech_db=round(speech_db,1), lead_sil=round(lead,2), trail_sil=round(trail,2),
        inner_sil=round(inner_sil,3), long_pause=round(long_pause,2), level_std=round(var,2), error=""))
keys = list(rows[0].keys()) if "dur" in rows[0] else ["file","error"]
keys = ["file","dur","peak_db","clip_ratio","snr_db","noise_floor_db","speech_db","lead_sil","trail_sil","inner_sil","long_pause","level_std","error"]
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys, restval=""); w.writeheader(); w.writerows(rows)
print("done", len(rows))
