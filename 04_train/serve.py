"""
Cloud inference preview on Modal (README 6.4): one GET endpoint that returns her voice as WAV, with the current
default settings. Opening the URL in a browser plays it.

  python -m modal deploy serve.py      -> prints the endpoint URL
  <url>?text=学长，早上好！&lang=zh[&speed=1.0][&seed=-1]
  python -m modal app stop nonoka-tts-serve   (take it down)
  python serve_bench.py <url> --tag ...       (cold / warm latency, README 6.4)
  set "SERVE_GPU=A10G" && python -m modal deploy serve.py   -> separate test app nonoka-tts-serve-a10g on other GPU types
  API for other projects: API.md

Defaults (README 5.4 / 5.8): SoVITS nonoka_full e10 for both languages; GPT nonoka_full e6 for Japanese and
nonoka_mix e2 for Chinese; Japanese reference 03004690 + aux 03005670, 03002830; Chinese reference zr04 (a converted
Chinese line) + aux 03004690, 03005670, 03002830; temperature 0.7, top_k 3, cut5.

Cost layout (README 6.4):
- `tts` is the web endpoint on a small CPU container: it checks the parameters and calls the GPU worker, so a bad
  request never starts a GPU.
- `TTSWorker` is the GPU (T4, or L4 when T4 capacity runs out; one container at most). Each GPU type keeps its
  own snapshots, so the list is kept short. It shuts down SCALEDOWN
  seconds after the last request so the GPU never idles. Its cold start is cut by a memory snapshot taken after the
  models are loaded on the GPU and warmed up (GPU snapshots are alpha on Modal): restores skip the imports, the
  weight loading and the first-inference warmup. The two languages share one TTS (BERT, HuBERT, SoVITS, SV) and only
  swap the GPT and its prompt cache.
NOTE: the URL is public; anyone who has it can generate audio while the app is deployed.
"""
import os
import time
from pathlib import Path

import modal

from train_gptsovits import GPT_DIR, PRE, REPO, SOVITS_DIR, VERSION, VOL, _names, image, vol

HERE = Path(__file__).resolve().parent
DATASET = HERE.parent / "02_dataset_full"
REFS = "/refs"
ZR04 = HERE / "out" / "zh_refs" / "samples" / "vc_ft1000_ref4690" / "04.wav"
MAX_CHARS = 300  # longer input is refused, so one request can't hold the GPU for minutes
SCALEDOWN = 12
DEFAULT_GPU = "T4,L4"  # README 6.4: T4 is ~22% cheaper and as good; L4 as the fallback when T4 runs out
GPU = os.environ.get("SERVE_GPU", DEFAULT_GPU)  # anything else deploys a separate test app

LANGS = {
    "ja": {"gpt": "nonoka_full-e6.ckpt", "ref": "03004690.ogg", "ref_lang": "ja",
           "ref_text": "はい、それにですね、記念品の折り紙、いつもとても綺麗な柄なんです。",
           "aux": ["03005670.ogg", "03002830.ogg"], "warmup": "先輩、おはようございます。"},
    "zh": {"gpt": "nonoka_mix-e2.ckpt", "ref": "zr04.wav", "ref_lang": "zh",
           "ref_text": "放学以后，我想去图书馆借几本书。",
           "aux": ["03004690.ogg", "03005670.ogg", "03002830.ogg"], "warmup": "学长，早上好！"},
}
SOVITS = "nonoka_full_e10"  # SOVITS_DIR/nonoka_full_e10_s<step>.pth

gpu_image = image.env({"SERVE_GPU": GPU}).add_local_python_source("train_gptsovits")  # the container imports the constants above
for clip in ("03004690", "03005670", "03002830"):
    gpu_image = gpu_image.add_local_file(DATASET / f"{clip}.ogg", f"{REFS}/{clip}.ogg")
gpu_image = gpu_image.add_local_file(ZR04, f"{REFS}/zr04.wav")
web_image = (modal.Image.debian_slim(python_version="3.11").pip_install("fastapi[standard]")
             .env({"SERVE_GPU": GPU}).add_local_python_source("train_gptsovits"))

app = modal.App("nonoka-tts-serve" + ("" if GPU == DEFAULT_GPU else "-" + GPU.lower().replace(",", "-")))


@app.cls(image=gpu_image, gpu=GPU.split(","), cpu=2, memory=8192, volumes={VOL: vol}, timeout=600,
         scaledown_window=SCALEDOWN, max_containers=1, enable_memory_snapshot=True, experimental_options={"enable_gpu_snapshot": True})
class TTSWorker:
    @modal.enter(snap=True)
    def load(self):
        import copy
        import glob
        import os
        import sys

        t0 = time.perf_counter()
        os.chdir(REPO)
        sys.path[:0] = [REPO, f"{REPO}/GPT_SoVITS"]
        from TTS_infer_pack.TTS import TTS

        s2 = sorted(glob.glob(f"{SOVITS_DIR}/{SOVITS}_s*.pth"))
        assert s2, f"no {SOVITS} weights in {SOVITS_DIR}"
        t1 = time.perf_counter()
        self.tts, self.gpt, self.lang = None, {}, None
        for lang, cfg in LANGS.items():
            s1 = f"{GPT_DIR}/{cfg['gpt']}"
            assert os.path.exists(s1), s1  # TTS silently falls back to the base models otherwise
            if self.tts is None:
                self.tts = TTS({"custom": {"device": "cuda", "is_half": True, "version": VERSION,
                                           "t2s_weights_path": s1, "vits_weights_path": s2[-1],
                                           "bert_base_path": f"{PRE}/chinese-roberta-wwm-ext-large",
                                           "cnhuhbert_base_path": f"{PRE}/chinese-hubert-base"}})
                blank = copy.deepcopy(self.tts.prompt_cache)
            else:
                self.tts.init_t2s_weights(s1)
            # everything init_t2s_weights sets, plus the per-reference prompt cache
            c = self.tts.configs
            self.gpt[lang] = {"t2s_model": self.tts.t2s_model, "max_sec": c.max_sec,
                              "mute_emb_sim_matrix": c.mute_emb_sim_matrix, "prompt_cache": copy.deepcopy(blank)}
            print(f"loaded {lang}: {cfg['gpt']} + {os.path.basename(s2[-1])}", flush=True)
        t2 = time.perf_counter()
        for lang, cfg in LANGS.items():  # first inference (CUDA kernels, reference features) goes into the snapshot
            self.synth_raw(cfg["warmup"], lang, 1.0, 0)
        t3 = time.perf_counter()
        print(f"snapshot phase: import {t1 - t0:.1f} s, load {t2 - t1:.1f} s, warmup {t3 - t2:.1f} s", flush=True)
        self.fresh, self.snap_time = True, time.time()

    @modal.enter(snap=False)
    def restore(self):
        # right after the snapshot phase when this container took the snapshot, minutes later when restored from one
        self.restored = time.time() - self.snap_time > 10

    def use(self, lang: str):
        if lang != self.lang:
            g = self.gpt[lang]
            self.tts.t2s_model, self.tts.prompt_cache = g["t2s_model"], g["prompt_cache"]
            self.tts.configs.max_sec, self.tts.configs.mute_emb_sim_matrix = g["max_sec"], g["mute_emb_sim_matrix"]
            self.lang = lang

    def synth_raw(self, text: str, lang: str, speed: float, seed: int):
        cfg = LANGS[lang]
        self.use(lang)
        return next(self.tts.run({
            "text": _names(text) if lang == "ja" else text, "text_lang": "all_" + lang,
            "ref_audio_path": f"{REFS}/{cfg['ref']}", "prompt_text": cfg["ref_text"], "prompt_lang": cfg["ref_lang"],
            "aux_ref_audio_paths": [f"{REFS}/{a}" for a in cfg["aux"]],
            "temperature": 0.7, "top_k": 3, "speed_factor": speed, "seed": seed,
            "text_split_method": "cut5", "batch_size": 1, "parallel_infer": False,
            "use_cuda_graph": False, "use_flash_attention": False}))

    @modal.method()
    def synth(self, text: str, lang: str, speed: float = 1.0, seed: int = -1) -> tuple:
        """-> (wav bytes, seed used, timing note for the x-timing header)"""
        import io
        import random

        import soundfile as sf
        import torch

        t = time.perf_counter()
        seed = seed if seed >= 0 else random.randrange(2**31)
        sr, audio = self.synth_raw(text, lang, speed, seed)
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        note = f"synth={time.perf_counter() - t:.2f};gpu={torch.cuda.get_device_name(0)};fresh={int(self.fresh)};restored={int(self.restored)}"
        self.fresh = False
        return buf.getvalue(), seed, note


@app.function(image=web_image, cpu=0.25, memory=256, scaledown_window=60, max_containers=1)
@modal.concurrent(max_inputs=20)
@modal.fastapi_endpoint(method="GET")
def tts(text: str, lang: str = "ja", speed: float = 1.0, seed: int = -1):
    """GET ?text=...&lang=ja|zh[&speed=1.0][&seed=-1] -> audio/wav (seed used in x-seed, timings in x-timing)."""
    from fastapi import HTTPException
    from fastapi.responses import Response

    text = text.strip()
    if lang not in LANGS or not text or len(text) > MAX_CHARS or not 0.6 <= speed <= 1.5:
        raise HTTPException(400, f"lang ja/zh, 1-{MAX_CHARS} characters, speed 0.6-1.5")
    t = time.perf_counter()
    wav, seed, note = TTSWorker().synth.remote(text, lang, speed, seed)
    return Response(wav, media_type="audio/wav",
                    headers={"x-seed": str(seed), "x-timing": f"total={time.perf_counter() - t:.2f};{note}"})
