"""构建镜像时预下载模型。任一模型下载失败都不会中断构建（运行时会再尝试，并在 /health 里报告）。"""
import sys

from huggingface_hub import snapshot_download

PATTERNS = ["*.json", "*.safetensors", "pytorch_model*.bin", "*.txt", "*.model", "tokenizer*", "vocab*", "merges*", "*.tiktoken"]

for repo in [a for a in sys.argv[1:] if a]:
    try:
        path = snapshot_download(repo_id=repo, allow_patterns=PATTERNS)
        print(f"[ok] {repo} -> {path}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 下载 {repo} 失败：{e}", flush=True)
