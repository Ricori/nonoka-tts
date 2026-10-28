"""
Qwen3-TTS zero-shot, cross-lingual: her Japanese clips as the voice reference, Chinese text as the target.

  python -m modal run qwen3_zh.py            (or double-click 7_qwen3_zh.bat)

Two cloning modes per reference clip:
  icl     reference audio + its Japanese transcript (the model continues from the prompt; carries timbre and style)
  xvec    speaker embedding only (x_vector_only_mode; no transcript, usually weaker timbre)
Writes out/zh/samples/qwen3_<mode>_<ref>/NN.wav + sentences.txt (Chinese lines from zh_sentences.txt).
Model: Qwen/Qwen3-TTS-12Hz-1.7B-Base (Apache-2.0).
"""
from pathlib import Path

import modal

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODELS = "/models/qwen3-tts"
REFS = ["03002860", "03003430", "03004600"]  # same reference clips as the blind-test page

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1", "sox", "ffmpeg")
    .pip_install("torch==2.8.0", "torchaudio==2.8.0")
    .pip_install("qwen-tts==0.1.1")
    .run_commands(f"python -c \"from huggingface_hub import snapshot_download; snapshot_download('{MODEL_ID}', local_dir='{MODELS}')\"")
    .env({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
)
app = modal.App("nonoka-qwen3-zh", image=image)


# The default cap is 2048 steps (~170 s at 12 Hz, each step also decodes 15 sub-codes), so one utterance that
# never emits EOS keeps an L4 busy for half an hour. Test lines are < 10 s: cap at ~25 s and retry runaways.
MAX_NEW_TOKENS = 300
RUNAWAY_SECONDS = 20


@app.function(gpu="L4", timeout=1800)
def synth(job: tuple) -> tuple:
    """job: (folder, ogg_bytes, transcript, xvec_only, sentences) -> (folder, [wav_bytes per sentence], notes)"""
    import io

    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    folder, ogg, transcript, xvec, sentences = job
    model = Qwen3TTSModel.from_pretrained(MODELS, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    audio, sr = sf.read(io.BytesIO(ogg), dtype="float32")
    prompt = model.create_voice_clone_prompt(ref_audio=(audio, sr), ref_text=None if xvec else transcript,
                                             x_vector_only_mode=xvec)
    wavs, notes = [], []
    for i, text in enumerate(sentences, 1):
        for attempt in range(1, 4):
            torch.manual_seed(1000 * attempt + i)
            w, osr = model.generate_voice_clone(text=text, language="Chinese", voice_clone_prompt=prompt,
                                                max_new_tokens=MAX_NEW_TOKENS)
            dur = len(w[0]) / osr
            print(f"{folder} {i:02d} try {attempt}: {dur:.1f}s", flush=True)
            if dur <= RUNAWAY_SECONDS:
                break
        if dur > RUNAWAY_SECONDS:
            notes.append(f"{i:02d}: still {dur:.0f}s after 3 tries (no clean stop)")
        buf = io.BytesIO()
        sf.write(buf, w[0], osr, format="WAV")
        wavs.append(buf.getvalue())
    return folder, wavs, notes


@app.local_entrypoint()
def main():
    here = Path(__file__).resolve().parent
    dataset = here.parent / "02_dataset_v1"
    esd = {l.split("|")[0].rsplit(".", 1)[0]: l.split("|")[3]
           for l in (dataset / "esd.list").read_text(encoding="utf-8").splitlines() if l.strip()}
    lines = [l.strip() for l in (here / "zh_sentences.txt").read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#")]
    jobs = [(f"qwen3_{mode}_{r}", (dataset / f"{r}.ogg").read_bytes(), esd[r], mode == "xvec", lines)
            for r in REFS for mode in ("icl", "xvec")]
    for name, wavs, notes in synth.map(jobs, order_outputs=False):  # one container per job, saved as each finishes
        d = here / "out" / "zh" / "samples" / name
        d.mkdir(parents=True, exist_ok=True)
        for i, w in enumerate(wavs, 1):
            (d / f"{i:02d}.wav").write_bytes(w)
        (d / "sentences.txt").write_text("\n".join(lines), encoding="utf-8")
        print(f"{len(wavs)} files -> {d}" + (f"  !! {'; '.join(notes)}" if notes else ""), flush=True)
