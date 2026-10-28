r"""
Singing test scores (README 5.9): ECAPA similarity to the centroid of her dataset clips + DNSMOS, for the original
lead and each converted lead of one sing_vc.py run, over the whole file and in 6 s windows (quiet windows skipped).

  venv\Scripts\python sing_eval.py ..\04_train\out\sing\<song>\<run> [dataset_dir]   (default ..\02_dataset_v1)
"""
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
from speechmos import dnsmos

here = Path(__file__).resolve().parent
seg = Path(sys.argv[1]).resolve()
dataset = Path(sys.argv[2] if len(sys.argv) > 2 else here.parent / "02_dataset_v1").resolve()
SR = 16000


def load(p):
    x, sr = sf.read(p, dtype="float32", always_2d=True)
    return librosa.resample(x.mean(1), orig_sr=sr, target_sr=SR)


ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(here / "models" / "ecapa"),
                                       local_strategy=LocalStrategy.COPY)
emb = lambda x: torch.nn.functional.normalize(ecapa.encode_batch(torch.from_numpy(x)[None]).squeeze(), dim=0)
c = torch.nn.functional.normalize(torch.stack([emb(load(p)) for p in sorted(dataset.glob("*.ogg"))]).mean(0), dim=0)

for p in [seg / "_lead_dry.wav"] + sorted(q for q in seg.glob("*.wav") if not q.name.startswith("_") and "_mix" not in q.name):
    x = load(p)
    wins = [x[i:i + 6 * SR] for i in range(0, len(x) - 6 * SR + 1, 6 * SR)]
    wins = [w for w in wins if np.sqrt(np.mean(w ** 2)) > 0.01]
    sims = [float(emb(w) @ c) for w in wins]
    dm = np.mean([dnsmos.run(np.clip(w, -1, 1), SR)["ovrl_mos"] for w in wins])
    print(f"{p.stem:14s} sim {float(emb(x) @ c):.3f} (6s windows mean {np.mean(sims):.3f}, min {min(sims):.3f})  dnsmos {dm:.2f}")
