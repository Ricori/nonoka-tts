"""
Render the test sentences through a deployed serve.py endpoint, to compare GPU types (README 6.4): every sentence of
test_sentences.txt (ja) and zh_sentences.txt (zh) once per seed, into the layout the CER and auto_eval tools read:
  out/gpu_test/<lang>/samples/<name>_s<seed>/NN.wav + sentences.txt
Per-request timings go to out/gpu_test/render_timing.csv.

  python serve_render.py <endpoint url> --name t4 [--seeds 1 2]
"""
import argparse
import csv
from pathlib import Path

from serve_bench import get
from train_gptsovits import _names

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "gpu_test"


def read(f):
    return [l.strip() for l in (HERE / f).read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    a = ap.parse_args()
    sets = {"ja": read("test_sentences.txt"), "zh": read("zh_sentences.txt")}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "render_timing.csv"
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["name", "lang", "seed", "idx", "chars", "seconds", "status", "audio_s", "x_timing"])
        for lang, lines in sets.items():
            for seed in a.seeds:
                d = OUT / lang / "samples" / f"{a.name}_s{seed}"
                d.mkdir(parents=True, exist_ok=True)
                # the endpoint applies _names itself; sentences.txt holds what is actually read, like train_gptsovits
                (d / "sentences.txt").write_text("\n".join(_names(t) if lang == "ja" else t for t in lines), encoding="utf-8")
                for i, text in enumerate(lines, 1):
                    sec, status, body, hdr = get(a.url, lang, text=text, seed=seed)
                    (d / f"{i:02d}.wav").write_bytes(body)
                    audio_s = (len(body) - 44) / 2 / 32000 if status == 200 else 0  # 32 kHz 16-bit mono WAV
                    w.writerow([a.name, lang, seed, i, len(text), f"{sec:.2f}", status, f"{audio_s:.2f}",
                                hdr.get("x-timing", "")])
                    f.flush()
                    print(f"{a.name} {lang} s{seed} {i:02d}: {sec:.2f} s HTTP {status} {hdr.get('x-timing', '')}", flush=True)


if __name__ == "__main__":
    main()
