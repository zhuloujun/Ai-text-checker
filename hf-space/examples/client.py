"""API 调用示例：python examples/client.py <论文.docx|.pdf|.txt>
需要先设置环境变量：
  ATC_URL = https://tdyso-ai-text-checker.hf.space   （你的 Space 地址）
  ATC_KEY = atc-……                                   （在 /admin 签发的 Key）
只用到 Python 标准库。
"""
import json
import mimetypes
import os
import sys
import time
import urllib.request
import uuid

URL = os.environ.get("ATC_URL", "").rstrip("/")
KEY = os.environ.get("ATC_KEY", "")


def req(method, path, body=None, headers=None):
    h = {"Authorization": f"Bearer {KEY}", **(headers or {})}
    r = urllib.request.Request(URL + path, data=body, headers=h, method=method)
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')}")


def upload(path, mode="full"):
    boundary = uuid.uuid4().hex
    name = os.path.basename(path)
    ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        data = f.read()
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"mode\"\r\n\r\n{mode}\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"wait\"\r\n\r\nfalse\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\nContent-Type: {ctype}\r\n\r\n".encode(),
        data, f"\r\n--{boundary}--\r\n".encode(),
    ]
    return req("POST", "/v1/detect/file", b"".join(parts), {"Content-Type": f"multipart/form-data; boundary={boundary}"})


if __name__ == "__main__":
    if not URL or not KEY or len(sys.argv) < 2:
        raise SystemExit(__doc__)
    job = upload(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "full")
    while job["status"] in ("queued", "running"):
        print(f"\r{job['status']} {job.get('done', 0)}/{job.get('total', 0)}  剩余约 {round(job['eta_sec']) if job.get('eta_sec') else '?'} 秒   ", end="", flush=True)
        time.sleep(3)
        job = req("GET", f"/v1/jobs/{job['id']}")
    print()
    if job["status"] == "error":
        raise SystemExit(job["error"])
    s = job["result"]["summary"]
    print(f"AI 率 {s['ai_rate']:.1%} · 平均 AI 概率 {s['mean_prob']:.1%} · 阈值 {s['threshold']} · {'已校准' if s['calibrated'] else '未校准'}")
    for seg in job["result"]["segments"]:
        if seg["level"] in ("high", "mid"):
            print(f"[{seg['prob']:.0%}] {seg['text'][:80]}…")
