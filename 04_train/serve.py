"""
Cloud inference preview on Modal (README 6.4): one GET endpoint that returns her voice as WAV, with the current
default settings. Opening the URL in a browser plays it.

  python -m modal deploy serve.py      -> prints the endpoint URL
  <url>?text=学长，早上好！&lang=zh[&speed=1.0][&seed=-1]
  python -m modal app stop nonoka-tts-serve   (take it down)

Defaults (README 5.4 / 5.8): SoVITS nonoka_full e10 for both languages; GPT nonoka_full e6 for Japanese and
nonoka_mix e2 for Chinese; Japanese reference 03004690 + aux 03005670, 03002830; Chinese reference zr04 (a converted
Chinese line) + aux 03004690, 03005670, 03002830; temperature 0.7, top_k 3, cut5.
One GPU container at most (L4, or A10G when L4 capacity runs out); it shuts down SCALEDOWN seconds after the last request so the GPU never idles, which means
nearly every request is a cold start (model load ~10-20 s on top of the synthesis).
NOTE: the URL is public; anyone who has it can generate audio while the app is deployed.
"""
from pathlib import Path

import modal

from train_gptsovits import GPT_DIR, PRE, REPO, SOVITS_DIR, VERSION, VOL, _names, image, vol

HERE = Path(__file__).resolve().parent
DATASET = HERE.parent / "02_dataset_full"
REFS = "/refs"
ZR04 = HERE / "out" / "zh_refs" / "samples" / "vc_ft1000_ref4690" / "04.wav"
MAX_CHARS = 300  # longer input is refused, so one request can't hold the GPU for minutes
SCALEDOWN = 5

LANGS = {
    "ja": {"gpt": "nonoka_full-e6.ckpt", "ref": "03004690.ogg", "ref_lang": "ja",
           "ref_text": "はい、それにですね、記念品の折り紙、いつもとても綺麗な柄なんです。",
           "aux": ["03005670.ogg", "03002830.ogg"]},
    "zh": {"gpt": "nonoka_mix-e2.ckpt", "ref": "zr04.wav", "ref_lang": "zh",
           "ref_text": "放学以后，我想去图书馆借几本书。",
           "aux": ["03004690.ogg", "03005670.ogg", "03002830.ogg"]},
}
SOVITS = "nonoka_full_e10"  # SOVITS_DIR/nonoka_full_e10_s<step>.pth

serve_image = image.add_local_python_source("train_gptsovits")  # the container imports the constants above
for clip in ("03004690", "03005670", "03002830"):
    serve_image = serve_image.add_local_file(DATASET / f"{clip}.ogg", f"{REFS}/{clip}.ogg")
serve_image = serve_image.add_local_file(ZR04, f"{REFS}/zr04.wav")

app = modal.App("nonoka-tts-serve")


@app.cls(image=serve_image, gpu=["L4", "A10G"], volumes={VOL: vol}, timeout=600, scaledown_window=SCALEDOWN, max_containers=1)
class TTSService:
    @modal.enter()
    def load(self):
        import glob
        import os
        import sys

        os.chdir(REPO)
        sys.path[:0] = [REPO, f"{REPO}/GPT_SoVITS"]
        from TTS_infer_pack.TTS import TTS

        s2 = sorted(glob.glob(f"{SOVITS_DIR}/{SOVITS}_s*.pth"))
        assert s2, f"no {SOVITS} weights in {SOVITS_DIR}"
        self.models = {}
        for lang, cfg in LANGS.items():
            s1 = f"{GPT_DIR}/{cfg['gpt']}"
            assert os.path.exists(s1), s1  # TTS silently falls back to the base models otherwise
            self.models[lang] = TTS({"custom": {"device": "cuda", "is_half": True, "version": VERSION,
                                             "t2s_weights_path": s1, "vits_weights_path": s2[-1],
                                             "bert_base_path": f"{PRE}/chinese-roberta-wwm-ext-large",
                                             "cnhuhbert_base_path": f"{PRE}/chinese-hubert-base"}})
            print(f"loaded {lang}: {cfg['gpt']} + {os.path.basename(s2[-1])}", flush=True)

    def synth(self, text: str, lang: str, speed: float = 1.0, seed: int = -1) -> tuple:
        """-> (wav bytes, seed used)"""
        import io
        import random

        import soundfile as sf

        cfg = LANGS[lang]
        seed = seed if seed >= 0 else random.randrange(2**31)
        sr, audio = next(self.models[lang].run({
            "text": _names(text) if lang == "ja" else text, "text_lang": "all_" + lang,
            "ref_audio_path": f"{REFS}/{cfg['ref']}", "prompt_text": cfg["ref_text"], "prompt_lang": cfg["ref_lang"],
            "aux_ref_audio_paths": [f"{REFS}/{a}" for a in cfg["aux"]],
            "temperature": 0.7, "top_k": 3, "speed_factor": speed, "seed": seed,
            "text_split_method": "cut5", "batch_size": 1, "parallel_infer": False,
            "use_cuda_graph": False, "use_flash_attention": False}))
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        return buf.getvalue(), seed

    @modal.fastapi_endpoint(method="GET")
    def tts(self, text: str, lang: str = "ja", speed: float = 1.0, seed: int = -1):
        """GET ?text=...&lang=ja|zh[&speed=1.0][&seed=-1] -> audio/wav (seed used in the x-seed header)."""
        from fastapi import HTTPException
        from fastapi.responses import Response

        text = text.strip()
        if lang not in LANGS or not text or len(text) > MAX_CHARS or not 0.6 <= speed <= 1.5:
            raise HTTPException(400, f"lang ja/zh, 1-{MAX_CHARS} characters, speed 0.6-1.5")
        wav, seed = self.synth(text, lang, speed, seed)
        return Response(wav, media_type="audio/wav", headers={"x-seed": str(seed)})
