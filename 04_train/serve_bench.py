"""
Latency benchmark for the serve.py endpoint (README 6.4): each round waits long enough for the GPU container to stop,
then sends one cold request (ja) and two warm ones (zh, ja), all with a fixed seed so outputs can be compared between
deployments. Rows are appended to out/serve_bench.csv; round 0's audio goes to out/_bench_<tag>_<lang>.wav.

  python serve_bench.py <endpoint url> --tag snap [--rounds 3] [--gap 60]
"""
import argparse
import csv
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
TEXT = {"ja": "先輩、おはようございます。今日もいい天気ですね。", "zh": "学长，早上好！今天天气真不错呢。"}
SEED = 42


def get(url, lang, timeout=600):
    q = urllib.parse.urlencode({"text": TEXT[lang], "lang": lang, "seed": SEED})
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/?{q}", timeout=timeout) as r:
            body, status, hdr = r.read(), r.status, dict(r.headers)
    except urllib.error.HTTPError as e:
        body, status, hdr = e.read(), e.code, dict(e.headers)
    return time.perf_counter() - t, status, body, hdr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--gap", type=int, default=60, help="seconds to wait so the GPU container is gone")
    a = ap.parse_args()
    path = OUT / "serve_bench.csv"
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tag", "time", "round", "kind", "lang", "seconds", "status", "bytes", "x_timing"])
        for r in range(a.rounds):
            if r:
                time.sleep(a.gap)
            for kind, lang in (("cold", "ja"), ("warm", "zh"), ("warm", "ja")):
                sec, status, body, hdr = get(a.url, lang)
                timing = hdr.get("x-timing", hdr.get("X-Timing", ""))
                w.writerow([a.tag, time.strftime("%Y-%m-%d %H:%M:%S"), r, kind, lang, f"{sec:.2f}", status, len(body), timing])
                f.flush()
                print(f"{a.tag} r{r} {kind} {lang}: {sec:.2f} s  HTTP {status}  {len(body)} B  {timing}", flush=True)
                if r == 0 and status == 200 and kind == "warm":
                    (OUT / f"_bench_{a.tag}_{lang}.wav").write_bytes(body)


if __name__ == "__main__":
    main()
