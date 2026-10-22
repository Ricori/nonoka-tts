"""
GPT-SoVITS (v2ProPlus) fine-tuning on Modal for the "nonoka" voice.

  python -m modal run --detach train_gptsovits.py::train_main      (or double-click 1_train.bat)
  python -m modal run train_gptsovits.py::render_main              (or 2_render.bat: re-render after editing the sentence files)

Both default to the full dataset (02_dataset_full, experiment "nonoka_full"); `--dataset 02_dataset_v1` runs the
round-1 90-clip set (experiment "nonoka"). Each experiment keeps its own features and weights on the volume.

Pipeline (the WebUI's one-click steps, run headless): 1-get-text -> 2-get-hubert-wav32k -> 2-get-sv
-> 3-get-semantic -> SoVITS training (s2) -> GPT training (s1) -> render.
Renders test_sentences.txt (Japanese) into out/samples/gsv_<combo>/ and zh_sentences.txt (Chinese) into
out/zh/samples/gsv_<combo>/, so the blind-test pages, CER, auto_eval and gemini_judge pick them up.
Reference clip for inference: 03002860 (GPT-SoVITS needs a 3-10 s reference).
"""
import base64
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import modal

GSV_COMMIT = "b4570b876f5c086819ba84e3215472733c6a7b70"
VERSION = "v2ProPlus"
DATASETS = {"02_dataset_full": ("nonoka_full", "full"), "02_dataset_v1": ("nonoka", "gsv")}  # dir -> (experiment, sample prefix)
REPO = "/gsv"
PRE = f"{REPO}/GPT_SoVITS/pretrained_models"
VOL = "/vol"
REF = "03002860"
# names as she says them (from the full-set transcripts), in katakana so the g2p can't misread the kanji;
# applied to the training text and the Japanese test sentences, esd.list keeps the readable spelling
NAMES = [(re.compile(r"(奏|かなで)(?=先輩)"), "カナデ"), (re.compile(r"(夜月|よるつき)(?=先輩)"), "ヨルツキ"),
         (re.compile(r"(十和|とわ)(?=先輩)"), "トワ")]


def _names(text):
    for pat, kana in NAMES:
        text = pat.sub(kana, text)
    return text


S2_EPOCHS, S2_SAVE_EVERY = 10, 2
S1_EPOCHS, S1_SAVE_EVERY = 15, 3
# versions to render: (SoVITS epoch, GPT epoch, reference clip, sampling overrides for TTS.run).
# GPT epoch 0 = the official base GPT (s1v3), untouched by Japanese. The folder name gets the GPT epoch, then
# _t<temperature*10>k<top_k> when sampling is overridden and _ref<last 4 digits> when the reference isn't REF.
# Round 2 (full set) rendered (6, 9), (10, 6), (10, 9), (10, 15), (10, 0) with REF and default sampling.
# Round 3 rendered (10, 3, REF, {}), (10, 3, REF, LOW), (10, 6, REF, LOW), (10, 6, 03010230, LOW), (10, 6, 03004690, LOW).
# Extra params: "speed_factor" (name gets _sp<x10>) and "aux": [clip ids] fused as aux_ref_audio_paths (name gets _aux<n>).
LOW = {"temperature": 0.7, "top_k": 3}
COMBOS = [(10, 3, "03004690", LOW), (10, 6, "03005670", LOW), (10, 6, "03002830", LOW), (10, 6, "03004800", LOW),
          (10, 6, "03004690", {**LOW, "speed_factor": 0.9}),
          (10, 6, "03004690", {**LOW, "aux": ["03005670", "03002830"]})]


def _combo_name(prefix, e2, e1, ref, params):
    name = f"{prefix}_s2e{e2}_s1{f'e{e1}' if e1 else 'base'}"
    if params:
        name += f"_t{round(params.get('temperature', 1) * 10)}k{params.get('top_k', 'd')}"
    if "speed_factor" in params:
        name += f"_sp{round(params['speed_factor'] * 10)}"
    if params.get("aux"):
        name += f"_aux{len(params['aux'])}"
    return name + ("" if ref == REF else f"_ref{ref[-4:]}")


def _clip_ids():
    return {c[2] for c in COMBOS} | {a for c in COMBOS for a in c[3].get("aux", [])}
XXXXRT = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main"
SMOKE = f"""
import sys; sys.path.insert(0, 'GPT_SoVITS')
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from TTS_infer_pack.TTS import resample
from TTS_infer_pack.TextPreprocessor import TextPreprocessor
bert = '{PRE}/chinese-roberta-wwm-ext-large'
tp = TextPreprocessor(AutoModelForMaskedLM.from_pretrained(bert), AutoTokenizer.from_pretrained(bert), torch.device('cpu'))
for text, lang in [('先輩、おはようございます。', 'ja'), ('2026年10月21日でした。', 'all_ja'), ('学长，早上好！', 'all_zh'), ('学长，早上好！', 'zh')]:
    print(lang, len(tp.preprocess(text, lang, 'cut5', 'v2ProPlus')))
print(resample(torch.zeros(1, 48000), 48000, 16000, 'cpu', match_librosa=True).shape)
"""

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "build-essential", "cmake", "wget", "unzip", "libsndfile1")
    .pip_install("torch==2.7.0", "torchaudio==2.7.0", index_url="https://download.pytorch.org/whl/cu128")
    .run_commands(
        f"git clone https://github.com/RVC-Boss/GPT-SoVITS.git {REPO}",
        f"cd {REPO} && git checkout {GSV_COMMIT}",
        f"cd {REPO} && pip install -r extra-req.txt --no-deps",
        f"cd {REPO} && pip install -r requirements.txt torch==2.7.0 torchaudio==2.7.0",
        # what webui.py does on start: put the repo's module folders on the path (the scripts import `text`, `tools`, ...)
        "python -c \"import site; open(site.getsitepackages()[0] + '/users.pth', 'w').write('\\n'.join("
        f"['{REPO}', '{REPO}/GPT_SoVITS/BigVGAN', '{REPO}/tools', '{REPO}/tools/asr', '{REPO}/GPT_SoVITS', '{REPO}/tools/uvr5']))\"",
        f"python -c \"from huggingface_hub import snapshot_download; snapshot_download('lj1995/GPT-SoVITS', "
        f"local_dir='{PRE}', allow_patterns=['chinese-hubert-base/*', 'chinese-roberta-wwm-ext-large/*', "
        f"'s1v3.ckpt', 'v2Pro/s2*v2ProPlus.pth', 'sv/*'])\"",
        f"cd /tmp && wget -q {XXXXRT}/G2PWModel.zip && unzip -q -o G2PWModel.zip -d {REPO}/GPT_SoVITS/text && rm G2PWModel.zip",
        f"cd /tmp && wget -q {XXXXRT}/nltk_data.zip && unzip -q -o nltk_data.zip -d $(python -c 'import sys;print(sys.prefix)') && rm nltk_data.zip",
        f"cd /tmp && wget -q {XXXXRT}/open_jtalk_dic_utf_8-1.11.tar.gz && "
        f"tar -xzf open_jtalk_dic_utf_8-1.11.tar.gz -C $(python -c 'import os,pyopenjtalk;print(os.path.dirname(pyopenjtalk.__file__))') && rm open_jtalk_dic_utf_8-1.11.tar.gz",
        # smoke-test the imports the prep scripts and the renderer need, so path problems fail the build, not the GPU run
        f"cd {REPO} && python -s -c \"from text.cleaner import clean_text; print(clean_text('先輩、おはようございます。', 'ja', 'v2')[0][:8]); "
        f"print(clean_text('学长，早上好。', 'zh', 'v2')[0][:8]); from TTS_infer_pack.TTS import TTS\"",
    )
    .pip_install("resampy")  # TTS.py's GPU resampler imports it lazily, but requirements.txt doesn't list it
    # fast_langdetect refuses to run unless its cache dir exists (the official zip ships it); the smoke test below
    # then downloads its model and walks the renderer's text + resample path on CPU
    # (base64 because run_commands become single Dockerfile RUN lines)
    .run_commands(f"mkdir -p {PRE}/fast_langdetect",
                  f"echo {base64.b64encode(SMOKE.encode()).decode()} | base64 -d > /tmp/smoke.py && cd {REPO} && python -s /tmp/smoke.py")
    .env({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "version": VERSION, "is_half": "True"})
)
app = modal.App("nonoka-gptsovits", image=image)
vol = modal.Volume.from_name("nonoka-gptsovits", create_if_missing=True)

SOVITS_DIR = f"{VOL}/SoVITS_weights_{VERSION}"
GPT_DIR = f"{VOL}/GPT_weights_{VERSION}"


def _run(script, env=None, args=()):
    print("$", script, *args, flush=True)
    r = subprocess.run([sys.executable, "-s", script, *args], cwd=REPO, env={**os.environ, **(env or {})})
    vol.commit()
    if r.returncode:
        raise RuntimeError(f"{script} failed ({r.returncode})")


def _merge(parts_glob_prefix, out_path, header=None):
    part = Path(f"{parts_glob_prefix}-0.{out_path.rsplit('.', 1)[1]}")
    lines = part.read_text(encoding="utf-8").strip("\n").split("\n")
    part.unlink()
    Path(out_path).write_text("\n".join(([header] if header else []) + lines) + "\n", encoding="utf-8")
    return len(lines)


def _prepare(clips: list, exp: str):
    """clips: [(stem, ogg_bytes, text)] -> wav files + list, then the four feature-extraction steps."""
    import io
    import shutil

    import soundfile as sf

    EXP_DIR = f"{VOL}/logs/{exp}"
    wav_dir = f"{VOL}/data/{exp}/wavs"
    shutil.rmtree(EXP_DIR, ignore_errors=True)  # the prep scripts silently skip steps whose outputs already exist
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(EXP_DIR, exist_ok=True)
    rows = []
    for stem, ogg, text in clips:
        audio, sr = sf.read(io.BytesIO(ogg), dtype="float32")
        sf.write(f"{wav_dir}/{stem}.wav", audio, sr)
        rows.append(f"{stem}.wav|{exp}|JA|{text}")
    inp_text = f"{VOL}/data/{exp}/{exp}.list"
    Path(inp_text).write_text("\n".join(rows) + "\n", encoding="utf-8")
    common = {"inp_text": inp_text, "inp_wav_dir": wav_dir, "exp_name": exp, "opt_dir": EXP_DIR,
              "i_part": "0", "all_parts": "1", "_CUDA_VISIBLE_DEVICES": "0"}
    _run("GPT_SoVITS/prepare_datasets/1-get-text.py",
         {**common, "bert_pretrained_dir": f"{PRE}/chinese-roberta-wwm-ext-large"})
    n_text = _merge(f"{EXP_DIR}/2-name2text", f"{EXP_DIR}/2-name2text.txt")
    hub = {**common, "cnhubert_base_dir": f"{PRE}/chinese-hubert-base", "sv_path": f"{PRE}/sv/pretrained_eres2netv2w24s4ep4.ckpt"}
    _run("GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py", hub)
    _run("GPT_SoVITS/prepare_datasets/2-get-sv.py", hub)
    _run("GPT_SoVITS/prepare_datasets/3-get-semantic.py",
         {**common, "pretrained_s2G": f"{PRE}/v2Pro/s2G{VERSION}.pth", "s2config_path": f"GPT_SoVITS/configs/s2{VERSION}.json"})
    n_sem = _merge(f"{EXP_DIR}/6-name2semantic", f"{EXP_DIR}/6-name2semantic.tsv", header="item_name\tsemantic_audio")
    print(f"== prepared: {len(rows)} clips, {n_text} text rows, {n_sem} semantic rows", flush=True)
    assert n_text == n_sem == len(rows), "some clips were dropped during preparation"


def _train_s2(batch, exp):
    EXP_DIR = f"{VOL}/logs/{exp}"
    data = json.loads(Path(f"{REPO}/GPT_SoVITS/configs/s2{VERSION}.json").read_text())
    data["train"].update(batch_size=batch, epochs=S2_EPOCHS, text_low_lr_rate=0.4,
                         pretrained_s2G=f"{PRE}/v2Pro/s2G{VERSION}.pth", pretrained_s2D=f"{PRE}/v2Pro/s2D{VERSION}.pth",
                         if_save_latest=True, if_save_every_weights=True, save_every_epoch=S2_SAVE_EVERY,
                         gpu_numbers="0", grad_ckpt=False, lora_rank="32")
    data["model"]["version"] = VERSION
    data["data"]["exp_dir"] = data["s2_ckpt_dir"] = EXP_DIR
    data.update(save_weight_dir=SOVITS_DIR, name=exp, version=VERSION)
    os.makedirs(f"{EXP_DIR}/logs_s2_{VERSION}", exist_ok=True)
    os.makedirs(SOVITS_DIR, exist_ok=True)
    cfg = f"{VOL}/tmp_s2.json"
    Path(cfg).write_text(json.dumps(data))
    _run("GPT_SoVITS/s2_train.py", args=("--config", cfg))


def _train_s1(batch, exp):
    import yaml

    EXP_DIR = f"{VOL}/logs/{exp}"
    data = yaml.safe_load(Path(f"{REPO}/GPT_SoVITS/configs/s1longer-v2.yaml").read_text())
    data["train"].update(batch_size=batch, epochs=S1_EPOCHS, save_every_n_epoch=S1_SAVE_EVERY,
                         if_save_every_weights=True, if_save_latest=True, if_dpo=False,
                         half_weights_save_dir=GPT_DIR, exp_name=exp)
    data.update(pretrained_s1=f"{PRE}/s1v3.ckpt", train_semantic_path=f"{EXP_DIR}/6-name2semantic.tsv",
                train_phoneme_path=f"{EXP_DIR}/2-name2text.txt", output_dir=f"{EXP_DIR}/logs_s1_{VERSION}")
    os.makedirs(GPT_DIR, exist_ok=True)
    cfg = f"{VOL}/tmp_s1.yaml"
    Path(cfg).write_text(yaml.dump(data, default_flow_style=False))
    _run("GPT_SoVITS/s1_train.py", {"hz": "25hz", "_CUDA_VISIBLE_DEVICES": "0"}, ("--config_file", cfg))


RENDER = r'''
import json, os, sys, glob
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "GPT_SoVITS"))
import inspect, re
import soundfile as sf
from TTS_infer_pack.TTS import TTS
job = json.load(open(sys.argv[1], encoding="utf-8"))
# log the sampling defaults this commit uses, so "default" in the folder names means something
print("TTS.run defaults:", re.findall(r'"(top_k|top_p|temperature|repetition_penalty)", ?([0-9.]+)', inspect.getsource(TTS.run)), flush=True)
for combo in job["combos"]:
    s2, s1 = combo["s2"], combo["s1"]
    assert os.path.exists(s2) and os.path.exists(s1), (s2, s1)  # TTS silently falls back to base models otherwise
    tts = TTS({"custom": {"device": "cuda", "is_half": True, "version": job["version"], "t2s_weights_path": s1,
                          "vits_weights_path": s2, "bert_base_path": job["bert"], "cnhuhbert_base_path": job["hubert"]}})
    for lang, lines in job["sets"].items():
        out = os.path.join(job["out"], lang, combo["name"])
        os.makedirs(out, exist_ok=True)
        for i, text in enumerate(lines, 1):
            sr, audio = next(tts.run({"text": text, "text_lang": "all_" + lang, "ref_audio_path": combo["ref"],
                                      "prompt_text": combo["ref_text"], "prompt_lang": "ja", "seed": 1234 + i,
                                      "text_split_method": "cut5", "batch_size": 1, "parallel_infer": False,
                                      "use_cuda_graph": False, "use_flash_attention": False, **combo["params"]}))
            sf.write(os.path.join(out, f"{i:02d}.wav"), audio, sr)
        open(os.path.join(out, "sentences.txt"), "w", encoding="utf-8").write("\n".join(lines))
        print("rendered", lang, combo["name"], flush=True)
'''


def _render(refs, sets, exp, prefix):
    """refs: {clip id: (ogg bytes, text)} for every reference COMBOS uses."""
    import glob
    import io
    import shutil

    import soundfile as sf

    for rid, (ogg, _) in refs.items():
        audio, sr = sf.read(io.BytesIO(ogg), dtype="float32")
        sf.write(f"{VOL}/data/ref_{rid}.wav", audio, sr)
    combos = []
    for e2, e1, rid, params in COMBOS:
        s2 = sorted(glob.glob(f"{SOVITS_DIR}/{exp}_e{e2}_s*.pth"))
        s1 = f"{GPT_DIR}/{exp}-e{e1}.ckpt" if e1 else f"{PRE}/s1v3.ckpt"
        assert s2, f"no SoVITS weights for epoch {e2} in {os.listdir(SOVITS_DIR)}"
        run_params = {k: v for k, v in params.items() if k != "aux"}
        if params.get("aux"):
            run_params["aux_ref_audio_paths"] = [f"{VOL}/data/ref_{a}.wav" for a in params["aux"]]
        combos.append({"name": _combo_name(prefix, e2, e1, rid, params), "s2": s2[-1], "s1": s1,
                       "ref": f"{VOL}/data/ref_{rid}.wav", "ref_text": refs[rid][1], "params": run_params})
    out = f"{VOL}/render"
    shutil.rmtree(out, ignore_errors=True)
    job = {"version": VERSION, "combos": combos, "sets": sets, "out": out,
           "bert": f"{PRE}/chinese-roberta-wwm-ext-large", "hubert": f"{PRE}/chinese-hubert-base"}
    Path(f"{VOL}/render_job.json").write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    Path(f"{REPO}/render_job.py").write_text(RENDER, encoding="utf-8")
    _run("render_job.py", args=(f"{VOL}/render_job.json",))
    result = {}
    for lang in sets:
        for c in combos:
            d = Path(out) / lang / c["name"]
            result[f"{lang}/{c['name']}"] = {p.name: p.read_bytes() for p in d.iterdir()}
    return result


@app.function(gpu="L40S", cpu=8, memory=32768, volumes={VOL: vol}, timeout=3 * 3600)
def train(clips: list, refs: dict, sets: dict, exp: str, prefix: str, batch: int = 8) -> dict:
    vol.reload()
    _prepare(clips, exp)
    _train_s2(batch, exp)
    print("== SoVITS weights:", sorted(os.listdir(SOVITS_DIR)), flush=True)
    _train_s1(batch, exp)
    print("== GPT weights:", sorted(os.listdir(GPT_DIR)), flush=True)
    return _render(refs, sets, exp, prefix)


@app.function(gpu="L40S", volumes={VOL: vol}, timeout=3600)
def render(refs: dict, sets: dict, exp: str, prefix: str) -> dict:
    vol.reload()
    return _render(refs, sets, exp, prefix)


def _inputs(here, dataset_name):
    dataset = here.parent / dataset_name
    esd = [l.split("|") for l in (dataset / "esd.list").read_text(encoding="utf-8").splitlines() if l.strip()]
    clips = [(fn.rsplit(".", 1)[0], (dataset / f"{fn.rsplit('.', 1)[0]}.ogg").read_bytes(), _names(text))
             for fn, _, _, text in esd]
    read = lambda f: [l.strip() for l in (here / f).read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    sets = {"ja": [_names(t) for t in read("test_sentences.txt")], "zh": read("zh_sentences.txt")}
    refs = {s: (b, t) for s, b, t in clips if s in _clip_ids()}  # reference clips must be in the dataset
    return clips, sets, refs


def _save(here, result):
    for key, files in result.items():
        lang, name = key.split("/")
        d = here / "out" / ("samples" if lang == "ja" else "zh/samples") / name
        d.mkdir(parents=True, exist_ok=True)
        for fn, data in files.items():
            (d / fn).write_bytes(data)
        print(f"{len(files) - 1} files -> {d}")


@app.local_entrypoint()
def train_main(dataset: str = "02_dataset_full", batch: int = 8):
    here = Path(__file__).resolve().parent
    clips, sets, refs = _inputs(here, dataset)
    print(f"[{dataset}] uploading {len(clips)} clips; rendering {len(sets['ja'])} ja + {len(sets['zh'])} zh sentences x {len(COMBOS)} combos")
    _save(here, train.remote(clips, refs, sets, *DATASETS[dataset], batch))


@app.local_entrypoint()
def render_main(dataset: str = "02_dataset_full", langs: str = "ja,zh"):
    here = Path(__file__).resolve().parent
    _, sets, refs = _inputs(here, dataset)
    sets = {k: v for k, v in sets.items() if k in langs.split(",")}
    print(f"[{dataset}] rendering {', '.join(sets)} x {len(COMBOS)} combos with references {', '.join(refs)}")
    _save(here, render.remote(refs, sets, *DATASETS[dataset]))
