"""
Sarashina2.2-TTS zero-shot baseline on Modal: same test sentences, voice cloned from a few dataset clips.

  python -m modal run sarashina_zero_shot.py            (or double-click 6_sarashina.bat)

Writes out/samples/sarashina_<ref>/NN.wav + sentences.txt, one folder per reference clip, so the blind-test
page, CER, auto_eval and gemini_judge treat each reference as one more "checkpoint".
Sarashina embeds an inaudible SilentCipher watermark; it is left on as the model's README asks.
Model license: Sarashina Model NonCommercial License Agreement.
"""
from pathlib import Path

import modal

SARASHINA_COMMIT = "e0ac9c99160ea4bf8dde46892892c945e66fcc13"
MODEL_ID = "sbintuitions/sarashina2.2-tts"
REPO = "/sarashina"
MODELS = "/models/sarashina2.2-tts"
REFS = ["03002860", "03003430", "03004600"]  # same reference clips as the blind-test page

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "gcc", "libsndfile1", "ffmpeg")
    .pip_install("torch==2.9.0", "torchaudio==2.9.0")
    .run_commands(
        f"git clone https://github.com/sbintuitions/sarashina2.2-tts.git {REPO}",
        f"cd {REPO} && git checkout {SARASHINA_COMMIT} && pip install -e .",
        f"python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{MODEL_ID}', local_dir='{MODELS}', ignore_patterns=['samples/*'])\"",
    )
    .env({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
)
app = modal.App("nonoka-sarashina", image=image)


@app.function(gpu="L4", timeout=3600)
def synth(refs: list, sentences: list) -> dict:
    """refs: [(name, ogg_bytes, transcript)] -> {name: [wav_bytes per sentence]}"""
    import io
    import os

    import soundfile as sf
    from sarashina_tts.generate.generate import SarashinaTTSGenerator

    os.chdir(REPO)
    gen = SarashinaTTSGenerator(model_dir=MODELS, model_id=MODEL_ID)
    out = {}
    for name, ogg, transcript in refs:
        audio, sr = sf.read(io.BytesIO(ogg), dtype="float32")
        path = f"/tmp/{name}.wav"
        sf.write(path, audio, sr)
        kw = dict(
            flow_embedding=gen._extract_zero_shot_embedding(audio_prompt_path=path),
            audio_prompt_text=transcript,
            audio_prompt_tokens=gen._extract_audio_prompt_tokens(audio_prompt_path=path),
            audio_prompt_feat=gen._extract_audio_prompt_feat(audio_prompt_path=path),
            audio_prompt_path=path,
            flow_embedding_only=False,
        )
        wavs = []
        for text in sentences:  # one call per sentence so a failure is easy to spot
            wav = gen.generate([text], **kw)[0]
            buf = io.BytesIO()
            sf.write(buf, wav.detach().cpu().numpy().T, 24000, format="WAV")
            wavs.append(buf.getvalue())
        out[name] = wavs
        print("rendered", name)
    return out


@app.local_entrypoint()
def main():
    here = Path(__file__).resolve().parent
    dataset = here.parent / "02_dataset_v1"
    esd = {l.split("|")[0].rsplit(".", 1)[0]: l.split("|")[3]
           for l in (dataset / "esd.list").read_text(encoding="utf-8").splitlines() if l.strip()}
    lines = [l.strip() for l in (here / "test_sentences.txt").read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#")]
    refs = [(r, (dataset / f"{r}.ogg").read_bytes(), esd[r]) for r in REFS]
    result = synth.remote(refs, lines)
    for name, wavs in result.items():
        d = here / "out" / "samples" / f"sarashina_{name}"
        d.mkdir(parents=True, exist_ok=True)
        for i, w in enumerate(wavs, 1):
            (d / f"{i:02d}.wav").write_bytes(w)
        (d / "sentences.txt").write_text("\n".join(lines), encoding="utf-8")
        print(f"{len(wavs)} files -> {d}")
