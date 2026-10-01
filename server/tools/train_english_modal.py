"""在 Modal 的 GPU 上运行 tools/train_english.py（GitHub 免费机器只有 CPU，数据量大了跑不动）。

由 .github/workflows/train-english.yml 调用：modal run server/tools/train_english_modal.py --args "..."
训练结果打包成 english-classifier.tar.gz 写到当前目录，再由工作流发布到 Release。
费用：T4 约 0.6 美元 / 小时，一次训练约 20–40 分钟；从 Modal 每月免费额度里扣。
"""
from pathlib import Path

import modal

SERVER = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", "transformers>=4.45,<5", "tokenizers>=0.20", "huggingface_hub>=0.24",
                 "numpy>=1.26", "safetensors", "sentencepiece", "protobuf")
    .run_commands("python -c \"from huggingface_hub import snapshot_download; snapshot_download('FacebookAI/roberta-base')\"",
                  "python -c \"from huggingface_hub import hf_hub_download; "
                  "hf_hub_download('yaful/MAGE', 'valid.csv', repo_type='dataset', local_dir='/mage')\"")
    .add_local_dir(str(SERVER / "app"), "/root/server/app", ignore=["__pycache__"])
    .add_local_dir(str(SERVER / "tools"), "/root/server/tools", ignore=["__pycache__", "scores"])
)
app = modal.App("ai-text-checker-train-english")
# 训练结果先存到卷里（模型约 250 MB，不直接作为返回值传输），本地再从卷里取回
out_volume = modal.Volume.from_name("ai-text-checker-train-out", create_if_missing=True)


@app.function(image=image, gpu="T4", timeout=3 * 3600, cpu=4, memory=16384, volumes={"/out": out_volume})
def train(args: list[str]) -> str:
    import subprocess
    import sys
    import tarfile
    subprocess.run([sys.executable, "-u", "/root/server/tools/train_english.py", "--mage-dir", "/mage",
                    "--out", "/tmp/english-classifier", *args], check=True, cwd="/root/server")
    with tarfile.open("/out/english-classifier.tar.gz", "w:gz") as t:
        t.add("/tmp/english-classifier", arcname="english-classifier")
    out_volume.commit()
    return "english-classifier.tar.gz"


@app.local_entrypoint()
def main(args: str = ""):
    name = train.remote(args.split())
    with open(name, "wb") as fh:
        for chunk in out_volume.read_file(name):
            fh.write(chunk)
    print(f"已保存 {name}（{Path(name).stat().st_size / 1e6:.0f} MB）")
