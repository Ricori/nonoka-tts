"""
Style-Bert-VITS2 (JP-Extra) training on Modal for the "nonoka" voice.

Usage (from this folder, see the .bat files):
  modal run --detach train_sbv2.py::train_main          # upload data, preprocess, train, render samples
  modal run --detach train_sbv2.py::train_main --resume # continue an interrupted run (no re-preprocess)
  modal run train_sbv2.py::samples_main                 # re-render samples for every checkpoint
  modal volume get nonoka-sbv2 samples out --force      # download samples
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

MODEL = "nonoka"
SBV2_COMMIT = "66de777e06392c0f313600be03c43ef96658b244"
VOL_NAME = "nonoka-sbv2"
VOL = "/vol"
REPO = "/sbv2"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "build-essential", "cmake", "libsndfile1")
    .pip_install(
        "torch==2.3.1", "torchaudio==2.3.1", "torchvision==0.18.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .run_commands(
        f"git clone https://github.com/litagin02/Style-Bert-VITS2.git {REPO}",
        f"cd {REPO} && git checkout {SBV2_COMMIT}",
        # Drop WebUI transcription/slicing deps (faster-whisper pins av==10, which needs a source build)
        # and keep libraries at the major versions this commit was written against.
        f"cd {REPO} && sed -i -E '/^(faster-whisper|stable_ts|punctuators|pyannote\\.audio)/d' requirements.txt",
        f"cd {REPO} && pip install -r requirements.txt 'transformers<4.50' 'gradio<5' 'huggingface_hub<1.0' 'pyannote.audio>=3.1,<3.4'",
        # style_gen.py's speaker-embedding model; loading it here also smoke-tests the imports used later
        f"cd {REPO} && python -c \"from pyannote.audio import Model; Model.from_pretrained('pyannote/wespeaker-voxceleb-resnet34-LM'); "
        f"from style_bert_vits2.tts_model import TTSModel; from gradio_tabs.train import preprocess_all\"",
        # BERT, WavLM (SLM) and JP-Extra pretrained weights are baked into the image
        f"cd {REPO} && python initialize.py --skip_default_models",
    )
    .env({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
)

app = modal.App("nonoka-sbv2", image=image)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

DATASET = f"{VOL}/Data/{MODEL}"
ASSETS = f"{VOL}/model_assets"
SAMPLES = f"{VOL}/samples"


def _write_configs():
    """Point SBV2 at the volume and set the model name (needed in every new container)."""
    import yaml

    with open(f"{REPO}/configs/paths.yml", "w", encoding="utf-8") as f:
        yaml.dump({"dataset_root": f"{VOL}/Data", "assets_root": ASSETS}, f)
    with open(f"{REPO}/default_config.yml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["model_name"] = MODEL
    cfg["dataset_path"] = DATASET
    with open(f"{REPO}/config.yml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)


def _run(cmd, commit_every=300):
    """Run a command in the repo, streaming output and committing the volume periodically."""
    print("$", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=REPO)
    last = time.time()
    while p.poll() is None:
        time.sleep(5)
        if time.time() - last > commit_every:
            vol.commit()
            last = time.time()
    vol.commit()
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}")


PREPROCESS = """
import sys
from gradio_tabs.train import preprocess_all
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker
from style_bert_vits2.nlp.japanese.user_dict import update_dict
pyopenjtalk_worker.initialize_worker()
update_dict()
ok, msg = preprocess_all(
    model_name="{model}", batch_size={batch}, epochs={epochs}, save_every_steps={save},
    num_processes=4, normalize=False, trim=False,
    freeze_EN_bert=False, freeze_JP_bert=False, freeze_ZH_bert=False,
    freeze_style=False, freeze_decoder=False, use_jp_extra=True,
    val_per_lang=0, log_interval=100, yomi_error="skip",
)
print(msg)
sys.exit(0 if ok else 1)
"""

RENDER = """
import sys, glob, os
from pathlib import Path
import soundfile as sf
from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker
from style_bert_vits2.nlp.japanese.user_dict import update_dict
from style_bert_vits2.tts_model import TTSModel
pyopenjtalk_worker.initialize_worker()
update_dict()
bert_models.load_model(Languages.JP, device_map="cuda")
bert_models.load_tokenizer(Languages.JP)

assets = Path("{assets}/{model}")
lines = [l.strip() for l in open("{dataset}/test_sentences.txt", encoding="utf-8") if l.strip() and not l.startswith("#")]
ckpts = sorted(assets.glob("{model}_e*_s*.safetensors"), key=lambda p: int(p.stem.split("_s")[-1]))
print(f"{{len(ckpts)}} checkpoints, {{len(lines)}} sentences")
for ck in ckpts:
    out = Path("{samples}") / ck.stem
    done = out / "sentences.txt"
    if done.exists() and done.read_text(encoding="utf-8") == "\\n".join(lines):
        continue  # already rendered with the same sentences
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.wav"):
        old.unlink()
    m = TTSModel(model_path=ck, config_path=assets / "config.json",
                 style_vec_path=assets / "style_vectors.npy", device="cuda")
    for i, t in enumerate(lines, 1):
        sr, audio = m.infer(text=t)
        sf.write(out / f"{{i:02d}}.wav", audio, sr)
    (out / "sentences.txt").write_text("\\n".join(lines), encoding="utf-8")
    print("rendered", ck.stem)
    del m
"""


@app.function(gpu="L40S", cpu=8, memory=32768, volumes={VOL: vol}, timeout=5 * 3600)
def train(batch: int = 4, epochs: int = 150, save_every: int = 500, resume: bool = False):
    vol.reload()
    _write_configs()
    if not resume:
        n_esd = sum(1 for l in open(f"{DATASET}/esd.list", encoding="utf-8") if l.strip())
        print(f"== preprocess ({n_esd} lines in esd.list)")
        _run([sys.executable, "-c", PREPROCESS.format(model=MODEL, batch=batch, epochs=epochs, save=save_every)])
        n_train = sum(1 for l in open(f"{DATASET}/train.list", encoding="utf-8") if l.strip())
        n_val = sum(1 for l in open(f"{DATASET}/val.list", encoding="utf-8") if l.strip())
        print(f"== preprocess done: train={n_train} val={n_val} (skipped {n_esd - n_train - n_val})")
    print("== training")
    _run([sys.executable, "train_ms_jp_extra.py",
          "--config", f"{DATASET}/config.json",
          "--model", DATASET,
          "--assets_root", ASSETS,
          "--no_progress_bar"])
    print("== rendering samples")
    _run([sys.executable, "-c", RENDER.format(assets=ASSETS, model=MODEL, dataset=DATASET, samples=SAMPLES)])
    print("ALL DONE")


@app.function(gpu="L40S", volumes={VOL: vol}, timeout=3600)
def render():
    vol.reload()
    _write_configs()
    _run([sys.executable, "-c", RENDER.format(assets=ASSETS, model=MODEL, dataset=DATASET, samples=SAMPLES)])


def _upload_sentences(here: Path):
    with vol.batch_upload(force=True) as b:
        b.put_file(str(here / "test_sentences.txt"), f"/Data/{MODEL}/test_sentences.txt")


@app.local_entrypoint()
def train_main(data_dir: str = "", batch: int = 4, epochs: int = 150, save_every: int = 500, resume: bool = False):
    here = Path(__file__).resolve().parent
    src = Path(data_dir) if data_dir else here.parent / "02_dataset_v1"
    if not resume:
        audios = sorted(src.glob("*.ogg")) + sorted(src.glob("*.wav"))
        esd = src / "esd.list"
        assert esd.exists(), f"esd.list not found in {src}"
        names = {l.split("|")[0].rsplit(".", 1)[0] for l in esd.read_text(encoding="utf-8").splitlines() if l.strip()}
        audios = [a for a in audios if a.stem in names]
        print(f"uploading {len(audios)} audio files + esd.list from {src}")
        with vol.batch_upload(force=True) as b:
            for a in audios:
                b.put_file(str(a), f"/Data/{MODEL}/raw/{a.name}")
            b.put_file(str(esd), f"/Data/{MODEL}/esd.list")
    _upload_sentences(here)
    train.remote(batch=batch, epochs=epochs, save_every=save_every, resume=resume)


@app.local_entrypoint()
def samples_main():
    _upload_sentences(Path(__file__).resolve().parent)
    render.remote()
