"""
Cloud inference preview on Modal (README 6.4): a small web page to type a line and hear her say it, with the
current default settings.

  python -m modal deploy serve.py      -> prints the page URL (https://<workspace>--nonoka-tts-serve-tts-web.modal.run)
  python -m modal app stop nonoka-tts-serve   (take it down)

Defaults (README 5.4 / 5.8): SoVITS nonoka_full e10 for both languages; GPT nonoka_full e6 for Japanese and
nonoka_mix e2 for Chinese; Japanese reference 03004690 + aux 03005670, 03002830; Chinese reference zr04 (a converted
Chinese line) + aux 03004690, 03005670, 03002830; temperature 0.7, top_k 3, cut5.
One L4 container at most; it shuts down after SCALEDOWN seconds without requests (a cold start takes ~1 minute).
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
SCALEDOWN = 300

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

PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>nonoka TTS</title><style>
body{font-family:system-ui,sans-serif;max-width:760px;margin:24px auto;padding:0 16px;background:#fafafa;color:#222}
textarea{width:100%;box-sizing:border-box;height:110px;font-size:16px;padding:8px}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:10px 0}
button{font-size:16px;padding:6px 18px;cursor:pointer}.ex{font-size:13px;padding:2px 8px}
.item{background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px 10px;margin:8px 0}
.meta{font-size:12px;color:#777}audio{width:100%;margin-top:4px}#st{color:#a33}
</style></head><body>
<h2>nonoka TTS</h2>
<div class="row"><label><input type="radio" name="lang" value="ja" checked> 日语</label>
<label><input type="radio" name="lang" value="zh"> 中文</label>
<label>语速 <input id="speed" type="number" min="0.6" max="1.5" step="0.05" value="1.0" style="width:60px"></label>
<label>种子 <input id="seed" type="number" value="-1" style="width:90px"></label><span class="meta">（-1 为随机）</span></div>
<textarea id="text" placeholder="输入要说的话（最多 300 字）"></textarea>
<div class="row"><button id="go">生成</button><span id="st"></span></div>
<div class="row" id="exs"></div>
<div id="out"></div>
<script>
const EX={ja:["先輩、おはようございます。今日もいい天気ですね。","ふふっ、先輩ったら、また寝癖がついてますよ？","約束ですよ。明日の朝七時半に、いつもの場所で待ってますから。"],
zh:["学长，早上好！今天能一起回家吗？","嘿嘿，学长你看，头发又翘起来了。","谢谢你，学长。有你在身边，我就什么都不怕了。"]};
const lang=()=>document.querySelector('input[name=lang]:checked').value;
function exs(){const d=document.getElementById('exs');d.innerHTML='';for(const t of EX[lang()]){const b=document.createElement('button');b.className='ex';b.textContent=t;b.onclick=()=>{document.getElementById('text').value=t};d.appendChild(b)}}
document.querySelectorAll('input[name=lang]').forEach(r=>r.onchange=exs);exs();
document.getElementById('go').onclick=async()=>{
 const text=document.getElementById('text').value.trim();if(!text)return;
 const st=document.getElementById('st'),go=document.getElementById('go');go.disabled=true;
 st.style.color='#555';st.textContent='生成中…（首次启动约 1 分钟）';const t0=performance.now();
 try{const r=await fetch('tts',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({text,lang:lang(),speed:+document.getElementById('speed').value,seed:+document.getElementById('seed').value})});
  if(!r.ok)throw new Error(await r.text());
  const url=URL.createObjectURL(await r.blob()),it=document.createElement('div');it.className='item';
  it.innerHTML=`<div></div><div class="meta"></div><audio controls autoplay src="${url}"></audio>`;
  it.children[0].textContent=text;
  it.children[1].textContent=`${lang()==='ja'?'日语':'中文'} · 语速 ${document.getElementById('speed').value} · 种子 ${r.headers.get('x-seed')} · ${((performance.now()-t0)/1000).toFixed(1)} 秒`;
  document.getElementById('out').prepend(it);st.textContent='';
 }catch(e){st.style.color='#a33';st.textContent='出错：'+e.message}finally{go.disabled=false}};
</script></body></html>"""


@app.cls(image=serve_image, gpu="L4", volumes={VOL: vol}, timeout=600, scaledown_window=SCALEDOWN, max_containers=1)
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
        self.tts = {}
        for lang, cfg in LANGS.items():
            s1 = f"{GPT_DIR}/{cfg['gpt']}"
            assert os.path.exists(s1), s1  # TTS silently falls back to the base models otherwise
            self.tts[lang] = TTS({"custom": {"device": "cuda", "is_half": True, "version": VERSION,
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
        sr, audio = next(self.tts[lang].run({
            "text": _names(text) if lang == "ja" else text, "text_lang": "all_" + lang,
            "ref_audio_path": f"{REFS}/{cfg['ref']}", "prompt_text": cfg["ref_text"], "prompt_lang": cfg["ref_lang"],
            "aux_ref_audio_paths": [f"{REFS}/{a}" for a in cfg["aux"]],
            "temperature": 0.7, "top_k": 3, "speed_factor": speed, "seed": seed,
            "text_split_method": "cut5", "batch_size": 1, "parallel_infer": False,
            "use_cuda_graph": False, "use_flash_attention": False}))
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        return buf.getvalue(), seed

    @modal.method()
    def speak(self, text: str, lang: str, speed: float = 1.0, seed: int = -1) -> bytes:
        return self.synth(text, lang, speed, seed)[0]

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse, Response
        from pydantic import BaseModel

        class Req(BaseModel):
            text: str
            lang: str = "ja"
            speed: float = 1.0
            seed: int = -1

        api = FastAPI()

        @api.get("/", response_class=HTMLResponse)
        def page():
            return PAGE

        @api.post("/tts")
        def tts(req: Req):
            text = req.text.strip()
            if req.lang not in LANGS or not text or len(text) > MAX_CHARS or not 0.6 <= req.speed <= 1.5:
                raise HTTPException(400, f"lang ja/zh, 1-{MAX_CHARS} characters, speed 0.6-1.5")
            wav, seed = self.synth(text, req.lang, req.speed, req.seed)
            return Response(wav, media_type="audio/wav", headers={"x-seed": str(seed)})

        return api


@app.local_entrypoint()
def main(text: str = "先輩、おはようございます。今日もいい天気ですね。", lang: str = "ja", out: str = "out/_serve_test.wav"):
    """Smoke test without the web page: python -m modal run serve.py --text ... --lang zh"""
    Path(HERE / out).write_bytes(TTSService().speak.remote(text, lang))
    print(f"-> {HERE / out}")
