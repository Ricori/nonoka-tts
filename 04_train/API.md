# nonoka TTS 接口说明（给外部项目调用）

用 nonoka 的声音朗读日语或中文，返回 WAV 音频。服务部署在 Modal 上，代码是 `04_train/serve.py`，设计和实测数据见 README 6.4。

> 仅限个人非商用（README 第 8 节）。网址是公开的，没有鉴权，**不要把网址写进公开的前端页面或公开仓库**。

---

## 1. 接口

```
GET https://ricori--nonoka-tts-serve-tts.us-east.modal.run/?text=<文本>&lang=<ja|zh>[&speed=1.0][&seed=-1]
```

| 参数 | 必填 | 类型 | 说明 |
|---|---|---|---|
| `text` | 是 | 字符串 | 要朗读的文本，1–300 字（去掉首尾空白后计算）。必须 URL 编码（UTF-8）。 |
| `lang` | 否 | `ja` / `zh` | 文本语言，默认 `ja`。整段按这一种语言读，不支持一句话里混着日语和中文。 |
| `speed` | 否 | 数字 | 语速，0.6–1.5，默认 1.0。 |
| `seed` | 否 | 整数 | 随机种子，默认 -1（随机）。想让同一句话每次读得一样，就固定一个 ≥ 0 的值。 |

- 只有这一个接口，只支持 GET。浏览器直接打开就能播放。
- 长文本会在服务端按标点切句再合成，返回一整段音频。
- 日语里的「奏先輩」「夜月先輩」会自动换成正确读音（カナデ、ヨルツキ），调用方不用处理。

## 2. 返回

**成功：HTTP 200**
- 正文：WAV 音频，32 kHz、16 bit、单声道（`Content-Type: audio/wav`）。
- 响应头：

| 响应头 | 例子 | 说明 |
|---|---|---|
| `x-seed` | `1234567` | 实际用的种子。想复现这次的结果，下次把它作为 `seed` 传进来。 |
| `x-timing` | `total=2.10;synth=1.93;gpu=Tesla T4;fresh=0;restored=1` | 调试用。`total` 是服务端总耗时（秒），`synth` 是推理耗时，`gpu` 是卡型，`fresh=1` 表示这次请求是冷启动，`restored=1` 表示容器是从快照恢复的。 |

**失败**（正文是 JSON，`detail` 字段说明原因）

| 状态码 | 什么时候 | 正文例子 |
|---|---|---|
| 400 | `lang` 不是 ja/zh、文本为空或超过 300 字、`speed` 超出 0.6–1.5 | `{"detail":"lang ja/zh, 1-300 characters, speed 0.6-1.5"}` |
| 422 | 缺 `text`，或者参数类型不对（比如 `speed=abc`） | `{"detail":[{"type":"missing","loc":["query","text"],...}]}` |
| 5xx | 服务端出错，或者应用已经下线 | — |

400 和 422 不会启动 GPU，也不花钱。

## 3. 延迟和并发

| 情况 | 大概耗时（客户端看到的，含网络） |
|---|---|
| 热请求（上一个请求结束后 12 秒内） | 3–5 秒 |
| 冷启动（GPU 容器已停） | 10–20 秒 |
| 重新部署后头几次冷启动（要重建快照） | 30 秒–2 分钟 |
| T4 和 L4 都没货 | 排队，时间不定 |

- GPU 容器请求结束 12 秒后就停机，所以隔一段时间来一个的请求基本都是冷启动。
- **同时只有一个 GPU 容器、一次处理一个请求**，同时发来的请求会排队依次处理。批量生成时请按顺序一个一个发，不要并发。
- 客户端超时建议设成 **180 秒以上**。
- 失败重试：5xx 或超时可以重试 1–2 次；400 和 422 不用重试，改参数再发。

## 4. 调用例子

**Python（标准库）**

```python
import urllib.parse, urllib.request

URL = "https://ricori--nonoka-tts-serve-tts.us-east.modal.run/"

def tts(text: str, lang: str = "ja", speed: float = 1.0, seed: int = -1) -> tuple[bytes, int]:
    q = urllib.parse.urlencode({"text": text, "lang": lang, "speed": speed, "seed": seed})
    with urllib.request.urlopen(URL + "?" + q, timeout=180) as r:
        return r.read(), int(r.headers["x-seed"])

wav, seed = tts("学长，早上好！", lang="zh")
open("out.wav", "wb").write(wav)
```

**Python（requests）**

```python
import requests

r = requests.get("https://ricori--nonoka-tts-serve-tts.us-east.modal.run/",
                 params={"text": "先輩、おはようございます。", "lang": "ja"}, timeout=180)
r.raise_for_status()
open("out.wav", "wb").write(r.content)
```

**JavaScript（浏览器或 Node 18+）**

```js
const url = new URL("https://ricori--nonoka-tts-serve-tts.us-east.modal.run/");
url.search = new URLSearchParams({ text: "学长，早上好！", lang: "zh" });
const r = await fetch(url, { signal: AbortSignal.timeout(180_000) });
if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
const wav = await r.arrayBuffer();          // Node: fs.writeFileSync("out.wav", Buffer.from(wav))
// 浏览器里直接播放：new Audio(URL.createObjectURL(new Blob([wav], { type: "audio/wav" }))).play();
```

**curl（Linux / macOS）**

```bash
curl -G --max-time 180 -o out.wav "https://ricori--nonoka-tts-serve-tts.us-east.modal.run/" --data-urlencode "text=先輩、おはようございます。" --data-urlencode "lang=ja"
```

- Windows 的 Git Bash 里用 curl 传中日文会被转错码，读出来的内容不对。Windows 上请用 Python、浏览器或 PowerShell 的 `Invoke-WebRequest`。

## 5. 费用（参考）

按 Modal 价目（2026-10）：T4 容器（含 2 核、8 GiB）约 $0.000208/秒，L4 约 $0.000266/秒。

- 单独一次请求（冷启动 + 推理 + 停机前空闲 12 秒）约 25 GPU 秒，≈ $0.005。
- 连续请求时，每句只多花推理的 2–3 秒。
- 参数错误不花钱；接口所在的 CPU 小容器的费用可以忽略。

## 6. 运维

在 `04_train` 目录下：

```bash
python -m modal deploy serve.py
```

```bash
python -m modal app stop nonoka-tts-serve
```

- 第一条是部署或更新，网址不变；第二条是下线，下线后接口返回错误，重新部署就能恢复。
- 每次重新部署，快照都会作废，头几次冷启动会慢（见第 3 节）。
- 默认设置（模型、参考音频、采样参数）见 README 6.4；要改就改 `serve.py` 里的 `LANGS`，然后重新部署。
- 延迟测试：`python serve_bench.py <网址> --tag <标签>`，结果追加到 `out/serve_bench.csv`。
