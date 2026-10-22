"""
Singing test: convert a song's lead vocal to her timbre with Seed-VC's singing model (zero-shot).

  python -m modal run sing_vc.py --song <mp3>                         (separate stems only -> out/sing/<song>/stems/)
  python -m modal run sing_vc.py --song <mp3> --start 60 --length 30  (also convert that slice)
  python -m modal run sing_vc.py --song <mp3> --length 0              (convert the whole lead vocal)
  python -m modal run sing_vc.py::remix --folder out/sing/<song>/<slice> --boost 4   (redo the mixes only, local)

1. Stems (audio-separator): BS-Roformer splits vocals / instrumental, the Mel-Roformer karaoke model splits the
   vocals into lead / backing, the Mel-Roformer dereverb model dries the lead. Cached under out/sing/<song>/stems/.
2. Timbre: Seed-VC v1 singing model (whisper-small content, RMVPE F0, 44.1 kHz BigVGAN), zero-shot. The source F0 is
   kept and shifted by --shift semitones (no auto-adjust: that would pull the melody down to her speaking pitch).
   Each reference in REFS gives one version: a single clip, or several clips joined (Seed-VC reads up to ~25 s).
3. Mix: converted lead + backing + instrumental with a light reverb -> out/sing/<song>/<version>_mix.wav.
Song audio is not committed (out/sing/ is in .gitignore).
"""
from pathlib import Path

import modal

SEEDVC_COMMIT = "51383efd921027683c89e5348211d93ff12ac2a8"  # same as zh_vc.py
SEEDVC = "/seedvc"
SEP_DIR = "/models/sep"
SEP_MODELS = {
    "vocals": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "karaoke": "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    "dereverb": "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
}
DIFFUSION_STEPS = 50  # README: 30-50; singing gets the top of the range
GPUS = ["L4", "A10G", "L40S"]
VOCAL_BOOST_DB = 3.0  # lead over the original's level in the mix: the instrumental stem keeps some vocal bleed
# references: the GPT-SoVITS reference clips (README 5.2/5.3)
REFS = {
    "ref4690": ["03004690"],
    "refcat": ["03004690", "03010230", "03002860"],  # ~25 s, three styles
}

sep_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("audio-separator[gpu]==0.44.5", "onnxruntime-gpu")
    .run_commands(*[f"audio-separator --download_model_only --model_file_dir {SEP_DIR} -m {m}" for m in SEP_MODELS.values()])
    .env({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
)

# Pre-download every singing checkpoint by running one CPU conversion of two short tones at build time.
PREFETCH = f"""
import os, subprocess, numpy as np, soundfile as sf
os.chdir('{SEEDVC}')
t = np.arange(44100 * 2) / 44100
sf.write('/tmp/a.wav', 0.3 * np.sin(2 * np.pi * 220 * t), 44100)
sf.write('/tmp/b.wav', 0.3 * np.sin(2 * np.pi * 330 * t), 44100)
subprocess.run(['python', 'inference.py', '--source', '/tmp/a.wav', '--target', '/tmp/b.wav', '--output', '/tmp/o',
                '--diffusion-steps', '2', '--f0-condition', 'True', '--fp16', 'False'], check=True)
print('seed-vc singing prefetch ok')
"""
seedvc_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libsndfile1", "ffmpeg")
    .pip_install("torch==2.4.0", "torchaudio==2.4.0", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("numpy==1.26.4", "transformers==4.46.3", "librosa", "soundfile", "pydub", "scipy", "munch",
                 "hydra-core", "pyyaml", "python-dotenv", "einops", "accelerate", "huggingface-hub<0.27",
                 "descript-audio-codec", "funasr", "modelscope")
    .run_commands(f"git clone https://github.com/Plachtaa/seed-vc {SEEDVC} && cd {SEEDVC} && git checkout {SEEDVC_COMMIT}")
    .env({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "HF_HUB_CACHE": f"{SEEDVC}/checkpoints/hf_cache"})
    .run_commands(f"python -c \"import base64; exec(base64.b64decode('{__import__('base64').b64encode(PREFETCH.encode()).decode()}'))\"")
)

app = modal.App("nonoka-sing-vc")


def _run(cmd, cwd=None):
    import subprocess
    import sys

    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True, stdout=sys.stdout, stderr=sys.stderr)


@app.function(image=sep_image, gpu=GPUS, timeout=1800)
def separate(song: bytes, name: str) -> dict:
    """-> {stem: flac bytes} for instrumental, backing, lead, lead_dry (stereo 44.1 kHz)."""
    from audio_separator.separator import Separator

    work = Path("/tmp/sep")
    work.mkdir(exist_ok=True)
    src = work / f"{name}{Path(name).suffix or '.mp3'}"
    src.write_bytes(song)
    sep = Separator(model_file_dir=SEP_DIR, output_dir=str(work), output_format="FLAC", sample_rate=44100)

    def split(path, model, want):
        """Run one model; return {lowercased stem label: path}, checking the labels we rely on exist."""
        sep.load_model(model_filename=model)
        outs = {}
        for f in sep.separate(str(path)):
            label = f.rsplit("_(", 1)[1].split(")", 1)[0].lower()
            outs[label] = work / Path(f).name
        print(model, "->", sorted(outs), flush=True)
        missing = [w for w in want if w not in outs]
        assert not missing, f"{model}: no stem {missing} in {sorted(outs)}"
        return outs

    v = split(src, SEP_MODELS["vocals"], ["vocals", "instrumental"])
    k = split(v["vocals"], SEP_MODELS["karaoke"], ["vocals", "instrumental"])  # karaoke: vocals = lead, rest = backing
    d = split(k["vocals"], SEP_MODELS["dereverb"], ["noreverb"])
    stems = {"instrumental": v["instrumental"], "backing": k["instrumental"], "lead": k["vocals"], "lead_dry": d["noreverb"]}
    return {s: p.read_bytes() for s, p in stems.items()}


@app.function(image=seedvc_image, gpu=GPUS, timeout=3600, max_containers=5)
def convert(job: tuple) -> tuple:
    """job: (version, source wav bytes, ref wav bytes, semitone shift) -> (version, wav bytes at 44.1 kHz)"""
    version, source, ref, shift = job
    work = Path(f"/tmp/{version}")
    work.mkdir(parents=True, exist_ok=True)
    (work / "src.wav").write_bytes(source)
    (work / "ref.wav").write_bytes(ref)
    _run(["python", "inference.py", "--source", str(work / "src.wav"), "--target", str(work / "ref.wav"),
          "--output", str(work / "out"), "--diffusion-steps", str(DIFFUSION_STEPS), "--length-adjust", "1.0",
          "--inference-cfg-rate", "0.7", "--f0-condition", "True", "--auto-f0-adjust", "False",
          "--semi-tone-shift", str(shift), "--fp16", "True"], cwd=SEEDVC)
    return version, next((work / "out").glob("vc_*.wav")).read_bytes()


def _ref_wav(clips):
    """Join her clips (0.3 s gaps) into one 44.1 kHz mono wav."""
    import subprocess

    dataset = Path(__file__).resolve().parent.parent / "02_dataset_full"
    inputs = sum([["-i", str(dataset / f"{c}.ogg")] for c in clips], [])
    pads = "".join(f"[{i}:a]aresample=44100,apad=pad_dur=0.3[a{i}];" for i in range(len(clips)))
    joined = "".join(f"[a{i}]" for i in range(len(clips))) + f"concat=n={len(clips)}:v=0:a=1[o]"
    return subprocess.run(["ffmpeg", "-loglevel", "error", *inputs, "-filter_complex", pads + joined, "-map", "[o]",
                           "-ac", "1", "-f", "wav", "-"], check=True, capture_output=True).stdout


def _lufs(path):
    """Integrated loudness (EBU R128) as it plays in the stereo mix: a mono file counts on both channels."""
    import re
    import subprocess

    import soundfile as sf

    c1 = "c1" if sf.info(str(path)).channels > 1 else "c0"
    log = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", f"pan=stereo|c0=c0|c1={c1},ebur128=framelog=quiet",
                          "-f", "null", "-"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace").stderr
    return float(re.findall(r"I:\s+(-?[\d.]+) LUFS", log)[-1])


def _mix(lead, original, backing, instrumental, out, boost=VOCAL_BOOST_DB):
    """Converted lead (mono) centred over backing + instrumental, peak-limited. The lead is first matched to the
    original lead's loudness (Seed-VC comes out ~5 dB quieter), then raised by `boost` dB; light reverb on top."""
    gain = _lufs(original) - _lufs(lead) + boost
    print(f"lead gain {gain:+.1f} dB", flush=True)
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(lead), "-i", str(backing), "-i", str(instrumental),
          "-filter_complex",
          f"[0:a]aresample=44100,volume={gain:.2f}dB,pan=stereo|c0=c0|c1=c0,aecho=1.0:1.0:60|120:0.2|0.1[l];"
          "[1:a]aresample=44100[b];[2:a]aresample=44100[i];"
          "[l][b][i]amix=inputs=3:duration=shortest:normalize=0,alimiter=limit=0.95[m]",
          "-map", "[m]", str(out)])


def _mix_version(out_dir, version, boost=VOCAL_BOOST_DB):
    _mix(out_dir / f"{version}.wav", out_dir / "_lead_dry.wav", out_dir / "_backing.wav", out_dir / "_instrumental.wav",
         out_dir / f"{version}_mix.wav", boost)


@app.local_entrypoint()
def main(song: str, start: float = 60.0, length: float = 30.0, shift: int = 0, refs: str = ",".join(REFS)):
    import io

    import soundfile as sf

    song_path = Path(song)
    root = Path(__file__).resolve().parent / "out" / "sing" / song_path.stem
    stems_dir = root / "stems"
    if not (stems_dir / "lead_dry.flac").exists():
        stems_dir.mkdir(parents=True, exist_ok=True)
        print(f"separating {song_path.name}", flush=True)
        for s, b in separate.remote(song_path.read_bytes(), song_path.name).items():
            (stems_dir / f"{s}.flac").write_bytes(b)
        print(f"stems -> {stems_dir}", flush=True)
    if length < 0:
        return

    # slice every stem the same way so the mix lines up; length 0 = whole song
    tag = "full" if length == 0 else f"{int(start)}s_{int(length)}s"
    seg = {}
    for s in ("instrumental", "backing", "lead_dry"):
        audio, sr = sf.read(stems_dir / f"{s}.flac", dtype="float32")
        if length:
            audio = audio[int(start * sr):int((start + length) * sr)]
        seg[s] = (audio, sr)
    out_dir = root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for s, (audio, sr) in seg.items():
        sf.write(out_dir / f"_{s}.wav", audio, sr)
    lead, sr = seg["lead_dry"]
    buf = io.BytesIO()
    sf.write(buf, lead.mean(axis=1) if lead.ndim == 2 else lead, sr, format="WAV")

    jobs = [(f"{r}_k{shift:+d}", buf.getvalue(), _ref_wav(REFS[r]), shift) for r in refs.split(",")]
    for version, wav in convert.map(jobs, order_outputs=False):
        (out_dir / f"{version}.wav").write_bytes(wav)
        _mix_version(out_dir, version)
        print(f"{version} -> {out_dir}", flush=True)


@app.local_entrypoint()
def remix(folder: str, boost: float = VOCAL_BOOST_DB):
    """python -m modal run sing_vc.py::remix --folder out/sing/<song>/<slice> [--boost 4]  (local only, no GPU)"""
    out_dir = Path(folder).resolve()
    for lead in sorted(out_dir.glob("*.wav")):
        if not lead.name.startswith("_") and not lead.stem.endswith("_mix"):
            _mix_version(out_dir, lead.stem, boost)
            print(f"{lead.stem}_mix.wav", flush=True)
