"""
Song cover in her voice (README 5.9): convert a song's lead vocal to her timbre with Seed-VC's singing model (zero-shot).
4_sing.bat wraps the first form; drop a song file onto it.

  python -m modal run sing_vc.py::main --song <audio>                  (whole song)
  python -m modal run sing_vc.py::main --song <audio> --start 60 --length 30 (preview a slice)
  python -m modal run sing_vc.py::main --song <audio> --shift 12             (override the key shift, semitones)
  python -m modal run sing_vc.py::main --song <audio> --length -1            (stems + pitch only, no conversion)
  python -m modal run sing_vc.py::remix --folder out/sing/<song>/<run> --boost 5 --wet -6   (redo the mix only, local)

1. Stems (audio-separator): BS-Roformer splits vocals / instrumental, the Mel-Roformer karaoke model splits the
   vocals into lead / backing, the Mel-Roformer dereverb model dries the lead. Cached under out/sing/<song>/stems/,
   with the lead's median pitch in stems/pitch.txt.
2. Key: --shift auto moves the melody by the multiple of 12 closest to the gap between the singer's median pitch and
   hers (HER_F0), so a male song goes up an octave and the accompaniment stays in key. Any other shift transposes the
   backing and instrumental with it (ffmpeg rubberband) - a key change. Seed-VC's own auto-adjust is off: it would
   pull the melody down to her speaking pitch.
3. Timbre: Seed-VC v1 singing model (whisper-small content, RMVPE F0, 44.1 kHz BigVGAN), zero-shot. Each reference
   in --refs (keys of REFS) gives one version; refcat (three clips joined) clearly beat a single clip.
4. Mix: converted lead + backing + instrumental, lead matched to the original lead's loudness + VOCAL_BOOST_DB,
   KTV-style echo + hall reverb on the lead (REVERB_WET_DB), peak-limited.
Output: out/sing/<song>/<full|START s_LENGTH s>_k<shift>/<ref>_mix.mp3 (+ wavs of every part).
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
SCALEDOWN = 3  # s: a container stops 3 s after its last call (default 60) - each function is called once per run
VOCAL_BOOST_DB = 3.0  # lead over the original's level in the mix: the instrumental stem keeps some vocal bleed
# KTV effect on the lead: repeating echo, then a stereo hall reverb on a send
KTV_ECHO_MS, KTV_ECHO_DECAYS = 200, (0.3, 0.15, 0.07)
REVERB_RT60 = 1.8  # s
REVERB_WET_DB = -9.0  # reverb send level against the dry lead
HER_F0 = 320  # Hz, median pitch of 51 of her 030 clips (pyin, README 5.9)
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


@app.function(image=sep_image, gpu=GPUS, timeout=1800, scaledown_window=SCALEDOWN)
def separate(song: bytes, name: str) -> dict:
    """-> {stem: flac bytes} for instrumental, backing, lead, lead_dry (stereo 44.1 kHz)."""
    from audio_separator.separator import Separator

    work = Path("/tmp/sep")
    work.mkdir(exist_ok=True)
    src = work / (name if Path(name).suffix else f"{name}.mp3")
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


@app.function(image=sep_image, cpu=4, timeout=1800, scaledown_window=SCALEDOWN)
def median_f0(audio: bytes) -> float:
    """Median pitch (Hz) of the voiced frames of a vocal stem (pyin, same settings as HER_F0)."""
    import io

    import librosa
    import numpy as np

    y, sr = librosa.load(io.BytesIO(audio), sr=16000, mono=True)
    f, voiced, _ = librosa.pyin(y, fmin=80, fmax=1200, sr=sr, frame_length=1024)
    return float(np.median(f[voiced & ~np.isnan(f)]))


@app.function(image=seedvc_image, gpu=GPUS, timeout=3600, max_containers=5, scaledown_window=SCALEDOWN)
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


def _reverb_ir(path, sr=44100, rt60=REVERB_RT60, seed=0):
    """Synthetic hall impulse response (stereo, unit energy per channel): 25 ms pre-delay, sparse early
    reflections, then decaying noise whose highs die ~2.5x faster than the lows."""
    import numpy as np
    import soundfile as sf

    rng = np.random.default_rng(seed)
    pre, n = int(0.025 * sr), int(rt60 * sr)
    t = np.arange(n) / sr
    ir = np.zeros((pre + n, 2), np.float32)
    for ch in range(2):  # independent noise per channel = width
        noise = rng.standard_normal(n)
        low = np.convolve(noise, np.ones(6) / 6, "same")
        tail = low * np.exp(-6.91 * t / rt60) + (noise - low) * np.exp(-6.91 * t / (rt60 * 0.4))
        taps = rng.integers(int(0.005 * sr), int(0.08 * sr), 8)
        tail[taps] += rng.uniform(0.3, 0.8, 8) * rng.choice([-1, 1], 8) * np.abs(tail[:sr // 100]).max()
        ir[pre:, ch] = tail / np.sqrt(np.sum(tail ** 2))
    sf.write(str(path), ir, sr)


def _mix(lead, original, backing, instrumental, out, boost=VOCAL_BOOST_DB, wet=REVERB_WET_DB):
    """Converted lead (mono) centred over backing + instrumental, peak-limited; writes out (.wav) and an .mp3 next to it.
    The lead is first matched to the original lead's loudness (Seed-VC comes out ~5 dB quieter), then raised by
    `boost` dB; KTV echo + reverb (`wet` dB send) on top."""
    gain = _lufs(original) - _lufs(lead) + boost
    print(f"lead gain {gain:+.1f} dB, reverb {wet:+.1f} dB", flush=True)
    ir = out.with_name("_reverb_ir.wav")
    _reverb_ir(ir)
    delays = "|".join(str(KTV_ECHO_MS * (i + 1)) for i in range(len(KTV_ECHO_DECAYS)))
    decays = "|".join(str(d) for d in KTV_ECHO_DECAYS)
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(lead), "-i", str(backing), "-i", str(instrumental),
          "-i", str(ir), "-filter_complex",
          f"[0:a]aresample=44100,volume={gain:.2f}dB,aecho=1.0:1.0:{delays}:{decays},"
          "pan=stereo|c0=c0|c1=c0,asplit[dry][send];"
          f"[send]highpass=f=250,lowpass=f=7000,apad=pad_dur={REVERB_RT60}[s];"
          f"[s][3:a]afir=gtype=none:irnorm=-1,volume={wet:.2f}dB[wet];"
          "[1:a]aresample=44100[b];[2:a]aresample=44100[i];"
          "[dry][wet][b][i]amix=inputs=4:duration=shortest:normalize=0,alimiter=limit=0.95[m]",
          "-map", "[m]", str(out)])
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out), "-b:a", "256k", str(out.with_suffix(".mp3"))])


def _mix_version(out_dir, version, boost=VOCAL_BOOST_DB, wet=REVERB_WET_DB):
    _mix(out_dir / f"{version}.wav", out_dir / "_lead_dry.wav", out_dir / "_backing.wav", out_dir / "_instrumental.wav",
         out_dir / f"{version}_mix.wav", boost, wet)


def _transpose(path, semitones):
    """Pitch-shift a wav in place by `semitones`, tempo unchanged (rubberband, formants kept)."""
    tmp = path.with_name(path.stem + ".tmp.wav")
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-af",
          f"rubberband=pitch={2 ** (semitones / 12):.6f}:formant=preserved", str(tmp)])
    tmp.replace(path)


def _stems(song_path, stems_dir):
    """Separate once and measure the lead's pitch once; both are cached in stems_dir. -> median F0 (Hz)."""
    if not (stems_dir / "lead_dry.flac").exists():
        stems_dir.mkdir(parents=True, exist_ok=True)
        print(f"separating {song_path.name}", flush=True)
        for s, b in separate.remote(song_path.read_bytes(), song_path.name).items():
            (stems_dir / f"{s}.flac").write_bytes(b)
        print(f"stems -> {stems_dir}", flush=True)
    pitch = stems_dir / "pitch.txt"
    if not pitch.exists():
        print("measuring the lead's pitch", flush=True)
        pitch.write_text(f"{median_f0.remote((stems_dir / 'lead_dry.flac').read_bytes()):.1f}", encoding="utf-8")
    return float(pitch.read_text(encoding="utf-8"))


@app.local_entrypoint()
def main(song: str, start: float = 0.0, length: float = 0.0, shift: str = "auto", refs: str = "refcat"):
    import io
    import math

    import soundfile as sf

    song_path = Path(song).resolve()
    assert song_path.exists(), f"no such file: {song_path}"
    unknown = [r for r in refs.split(",") if r not in REFS]
    assert not unknown, f"unknown refs {unknown}; choose from {list(REFS)}"
    root = Path(__file__).resolve().parent / "out" / "sing" / song_path.stem
    f0 = _stems(song_path, root / "stems")
    gap = 12 * math.log2(HER_F0 / f0)
    k = 12 * round(gap / 12) if shift == "auto" else int(shift)
    print(f"singer's median pitch {f0:.0f} Hz, hers {HER_F0} Hz: gap {gap:+.1f} semitones -> shift {k:+d}"
          + (" (auto: octaves only, keeps the accompaniment's key)" if shift == "auto" else ""), flush=True)
    if shift == "auto" and abs(gap - k) > 4:
        print(f"!! the gap is far from a whole octave; if it sounds strained try --shift {k + (12 if gap > k else -12):+d}"
              " or a key change such as --shift " + f"{round(gap):+d}", flush=True)
    if length < 0:
        return

    # slice every stem the same way so the mix lines up; length 0 = whole song
    out_dir = root / (("full" if length == 0 else f"{int(start)}s_{int(length)}s") + f"_k{k:+d}")
    out_dir.mkdir(parents=True, exist_ok=True)
    lead = None
    for s in ("instrumental", "backing", "lead_dry"):
        audio, sr = sf.read(root / "stems" / f"{s}.flac", dtype="float32")
        if length:
            audio = audio[int(start * sr):int((start + length) * sr)]
        sf.write(out_dir / f"_{s}.wav", audio, sr)
        if s == "lead_dry":
            lead = (audio.mean(axis=1) if audio.ndim == 2 else audio, sr)
        elif k % 12:
            _transpose(out_dir / f"_{s}.wav", k)  # key change: the accompaniment moves with the melody
    buf = io.BytesIO()
    sf.write(buf, *lead, format="WAV")

    jobs = [(r, buf.getvalue(), _ref_wav(REFS[r]), k) for r in refs.split(",")]
    for version, wav in convert.map(jobs, order_outputs=False):
        (out_dir / f"{version}.wav").write_bytes(wav)
        _mix_version(out_dir, version)
        print(f"DONE {out_dir / (version + '_mix.mp3')}", flush=True)


@app.local_entrypoint()
def remix(folder: str, boost: float = VOCAL_BOOST_DB, wet: float = REVERB_WET_DB):
    """python -m modal run sing_vc.py::remix --folder out/sing/<song>/<run> [--boost 5] [--wet -6]  (local only, no GPU)"""
    out_dir = Path(folder).resolve()
    for lead in sorted(out_dir.glob("*.wav")):
        if not lead.name.startswith("_") and not lead.stem.endswith("_mix"):
            _mix_version(out_dir, lead.stem, boost, wet)
            print(f"DONE {out_dir / (lead.stem + '_mix.mp3')}", flush=True)
