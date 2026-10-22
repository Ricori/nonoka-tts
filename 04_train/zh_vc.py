"""
Chinese feasibility test (README 6.2 method 2): native-Chinese TTS first, then convert the timbre to hers.

  python -m modal run zh_vc.py                    (source TTS + Seed-VC fine-tune in parallel, then convert)
  python -m modal run zh_vc.py --steps 2000       (longer fine-tune; a finished run with the same steps is reused)

1. Source: Qwen3-TTS-12Hz-1.7B-Base in x-vector mode with her clip REF as the speaker prompt. Round 1 showed this mode
   reads Chinese correctly but only sounds "somewhat like" her (ECAPA 0.71-0.73), so it supplies pronunciation and tone.
2. Timbre: Seed-VC v1 (whisper-small content encoder, 22.05 kHz, timbre-only; v2's convert_style would pull the
   Japanese accent back in) fine-tuned on the 498 clips of 02_dataset_full. Checkpoints live on the Modal volume.
3. Convert every source line with the zero-shot model and the fine-tuned checkpoints, REF as the target voice.
Writes out/zh/samples/<version>/NN.wav + sentences.txt; then run CER / gemini_judge / make_blind on out/zh as usual.
"""
from pathlib import Path

import modal

REF = "03004690"  # bright declarative clip, best GPT-SoVITS reference so far (README 5.2/5.3)
SEEDVC_COMMIT = "51383efd921027683c89e5348211d93ff12ac2a8"  # last commit (2025-04-20)
SEEDVC = "/seedvc"
VOL = "/vol"
QWEN_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
QWEN_DIR = "/models/qwen3-tts"
CFG = "configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
DIFFUSION_STEPS = 30  # README: 30-50 for best quality

# Qwen3-TTS: same caps as round 1 (an utterance that never emits EOS otherwise runs ~170 s and blocks the GPU)
MAX_NEW_TOKENS = 400  # longest new test line is ~25 characters, < 15 s
RUNAWAY_SECONDS = 25

qwen_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1", "sox", "ffmpeg")
    .pip_install("torch==2.8.0", "torchaudio==2.8.0")
    .pip_install("qwen-tts==0.1.1")
    .run_commands(f"python -c \"from huggingface_hub import snapshot_download; snapshot_download('{QWEN_ID}', local_dir='{QWEN_DIR}')\"")
    .env({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
)

# requirements.txt lists torch twice (nightly + 2.4.0) and GUI packages; install the pinned torch and the rest by hand.
# The last step pre-downloads every inference checkpoint into the image and doubles as an import smoke test.
PREFETCH = f"""
import os; os.chdir('{SEEDVC}'); os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
from huggingface_hub import snapshot_download
from hf_utils import load_custom_model_from_hf
load_custom_model_from_hf('Plachta/Seed-VC', 'DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth', 'config_dit_mel_seed_uvit_whisper_small_wavenet.yml')
load_custom_model_from_hf('funasr/campplus', 'campplus_cn_common.bin', config_filename=None)
for repo in ('openai/whisper-small', 'nvidia/bigvgan_v2_22khz_80band_256x'):
    snapshot_download(repo, cache_dir='./checkpoints/hf_cache')
import modules.commons, data.ft_dataset, optimizers  # noqa: F401  import smoke test
print('seed-vc prefetch ok')
"""
seedvc_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libsndfile1", "ffmpeg")
    .pip_install("torch==2.4.0", "torchaudio==2.4.0", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("numpy==1.26.4", "transformers==4.46.3", "librosa", "soundfile", "pydub", "scipy", "munch",
                 "hydra-core", "pyyaml", "python-dotenv", "einops", "accelerate", "huggingface-hub<0.27",
                 "descript-audio-codec", "funasr", "modelscope")
    .run_commands(f"git clone https://github.com/Plachtaa/seed-vc {SEEDVC} && cd {SEEDVC} && git checkout {SEEDVC_COMMIT}")
    .run_commands(f"python -c \"import base64; exec(base64.b64decode('{__import__('base64').b64encode(PREFETCH.encode()).decode()}'))\"")
    .env({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "HF_HUB_CACHE": f"{SEEDVC}/checkpoints/hf_cache"})
)

app = modal.App("nonoka-zh-vc")
vol = modal.Volume.from_name("nonoka-zhvc", create_if_missing=True)


@app.function(image=qwen_image, gpu="L4", timeout=1800)
def qwen_source(ref_ogg: bytes, sentences: list) -> tuple:
    """-> ([wav bytes per sentence], notes)"""
    import io

    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(QWEN_DIR, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    audio, sr = sf.read(io.BytesIO(ref_ogg), dtype="float32")
    prompt = model.create_voice_clone_prompt(ref_audio=(audio, sr), ref_text=None, x_vector_only_mode=True)
    wavs, notes = [], []
    for i, text in enumerate(sentences, 1):
        for attempt in range(1, 4):
            torch.manual_seed(1000 * attempt + i)
            w, osr = model.generate_voice_clone(text=text, language="Chinese", voice_clone_prompt=prompt,
                                                max_new_tokens=MAX_NEW_TOKENS)
            dur = len(w[0]) / osr
            print(f"qwen {i:02d} try {attempt}: {dur:.1f}s", flush=True)
            if dur <= RUNAWAY_SECONDS:
                break
        if dur > RUNAWAY_SECONDS:
            notes.append(f"{i:02d}: still {dur:.0f}s after 3 tries")
        buf = io.BytesIO()
        sf.write(buf, w[0], osr, format="WAV")
        wavs.append(buf.getvalue())
    return wavs, notes


def _run(cmd):
    import subprocess
    import sys

    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=SEEDVC, check=True, stdout=sys.stdout, stderr=sys.stderr)


@app.function(image=seedvc_image, gpu="L40S", cpu=8, memory=32768, volumes={VOL: vol}, timeout=2 * 3600)
def finetune(clips: list, steps: int, batch: int = 8) -> dict:
    """clips: [(name, ogg bytes)]. Returns {label: checkpoint path on the volume} for the kept checkpoints."""
    import glob

    import yaml

    run = f"{VOL}/runs/nonoka_{steps}"
    if not glob.glob(f"{run}/ft_model.pth"):
        data = Path(f"{VOL}/data/train")
        data.mkdir(parents=True, exist_ok=True)
        for name, ogg in clips:
            (data / f"{name}.ogg").write_bytes(ogg)
        cfg = yaml.safe_load(open(f"{SEEDVC}/{CFG}"))
        cfg["log_dir"] = f"{VOL}/runs"
        Path(f"{VOL}/cfg.yml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        vol.commit()
        _run(["python", "train.py", "--config", f"{VOL}/cfg.yml", "--dataset-dir", str(data), "--run-name", f"nonoka_{steps}",
              "--batch-size", str(batch), "--max-steps", str(steps), "--max-epochs", "1000",
              "--save-every", str(steps // 2), "--num-workers", "4"])
        vol.commit()
    ckpts = {"final": f"{run}/ft_model.pth"}
    for p in sorted(glob.glob(f"{run}/DiT_epoch_*_step_*.pth")):
        ckpts[f"step{int(p.rsplit('_step_', 1)[1].split('.')[0])}"] = p
    print("checkpoints:", ckpts, flush=True)
    return ckpts


@app.function(image=seedvc_image, gpu="L4", volumes={VOL: vol}, timeout=3600)
def convert(job: tuple) -> tuple:
    """job: (version, checkpoint path or None for zero-shot, ref ogg bytes, [source wav bytes]) -> (version, [wav bytes])"""
    import glob
    import shutil

    version, ckpt, ref_ogg, sources = job
    vol.reload()
    work = Path(f"/tmp/{version}")
    shutil.rmtree(work, ignore_errors=True)
    (work / "src").mkdir(parents=True)
    Path(f"/tmp/{REF}.ogg").write_bytes(ref_ogg)
    for i, w in enumerate(sources, 1):
        (work / "src" / f"{i:02d}.wav").write_bytes(w)
    extra = ["--checkpoint", ckpt, "--config", glob.glob(str(Path(ckpt).parent / "*.yml"))[0]] if ckpt else []
    outs = []
    for i in range(1, len(sources) + 1):
        out = work / f"out{i:02d}"
        _run(["python", "inference.py", "--source", str(work / "src" / f"{i:02d}.wav"), "--target", f"/tmp/{REF}.ogg",
              "--output", str(out), "--diffusion-steps", str(DIFFUSION_STEPS), "--length-adjust", "1.0",
              "--inference-cfg-rate", "0.7", "--f0-condition", "False", "--fp16", "True", *extra])
        outs.append(next(out.glob("vc_*.wav")).read_bytes())
    return version, outs


def _save(here, name, wavs, lines):
    d = here / "out" / "zh" / "samples" / name
    d.mkdir(parents=True, exist_ok=True)
    for i, w in enumerate(wavs, 1):
        (d / f"{i:02d}.wav").write_bytes(w)
    (d / "sentences.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(wavs)} files -> {d}", flush=True)


@app.local_entrypoint()
def main(steps: int = 1000):
    here = Path(__file__).resolve().parent
    dataset = here.parent / "02_dataset_full"
    names = [l.split("|")[0].rsplit(".", 1)[0] for l in (dataset / "esd.list").read_text(encoding="utf-8").splitlines() if l.strip()]
    clips = [(n, (dataset / f"{n}.ogg").read_bytes()) for n in names]
    lines = [l.strip() for l in (here / "zh_sentences.txt").read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#")]
    ref = (dataset / f"{REF}.ogg").read_bytes()
    print(f"fine-tuning Seed-VC on {len(clips)} clips ({steps} steps) while Qwen3-TTS reads {len(lines)} lines", flush=True)
    ft = finetune.spawn(clips, steps)
    src, notes = qwen_source.remote(ref, lines)
    _save(here, f"qwen3_xvec_ref{REF[-4:]}", src, lines)
    if notes:
        print("!! qwen runaways:", "; ".join(notes), flush=True)
    ckpts = ft.get()
    jobs = [(f"vc_zs_ref{REF[-4:]}", None, ref, src)]
    jobs += [(f"vc_ft{steps if k == 'final' else k[4:]}_ref{REF[-4:]}", p, ref, src)
             for k, p in ckpts.items() if k != f"step{steps}"]  # step<steps> is the same weights as final
    for version, wavs in convert.map(jobs, order_outputs=False):
        _save(here, version, wavs, lines)
