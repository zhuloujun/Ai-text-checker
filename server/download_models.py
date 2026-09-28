"""构建镜像时预下载模型。任一模型下载失败都不会中断构建（运行时会再尝试，并在 /health 里报告）。

参数：Hugging Face 模型名，或 "网址=本地目录"（下载 .tar.gz 并解压到该目录，用于本仓库 Release 里的诗词模型）。"""
import io
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (ai-text-checker model download)"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.read()
    except Exception as e:  # noqa: BLE001 —— 再用 curl 试一次（处理某些环境下的代理 / 重定向问题）
        print(f"[warn] urllib 下载失败（{e}），改用 curl", flush=True)
        import subprocess
        return subprocess.run(["curl", "-fsSL", "--retry", "3", url], check=True, capture_output=True).stdout


def fetch_tarball(url: str, dest: str):
    data = _download(url)
    tmp = Path(tempfile.mkdtemp())
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        tf.extractall(tmp)
    inner = [p for p in tmp.iterdir() if p.is_dir()]
    src = inner[0] if len(inner) == 1 else tmp
    dest_p = Path(dest)
    if dest_p.exists():
        shutil.rmtree(dest_p)
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest_p))
    return dest_p

PATTERNS = ["*.json", "*.safetensors", "pytorch_model*.bin", "*.txt", "*.model", "tokenizer*", "vocab*", "merges*", "*.tiktoken"]

def main(args):
    for repo in [a for a in args if a]:
        try:
            if repo.startswith("http") and "=" in repo:
                url, dest = repo.rsplit("=", 1)
                print(f"[ok] {url} -> {fetch_tarball(url, dest)}", flush=True)
                continue
            path = snapshot_download(repo_id=repo, allow_patterns=PATTERNS)
            print(f"[ok] {repo} -> {path}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 下载 {repo} 失败：{type(e).__name__}: {e}", flush=True)
            print(f"::warning title=模型下载失败::{repo}：{type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
