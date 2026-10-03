"""在 Modal 的 GPU 上运行 tools/train_chinese.py（由 .github/workflows/train-chinese.yml 调用）。"""
from pathlib import Path

import modal

SERVER = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.7.1", "transformers>=4.45,<5", "tokenizers>=0.20", "huggingface_hub>=0.24",
                 "numpy>=1.26", "safetensors", "sentencepiece", "protobuf")
    .run_commands("python -c \"from huggingface_hub import snapshot_download; snapshot_download('hfl/chinese-roberta-wwm-ext')\"",
                  "git clone --depth 1 https://github.com/NLP2CT/NLPCC-2025-Task1 /nlpcc")
    .add_local_dir(str(SERVER / "app"), "/root/server/app", ignore=["__pycache__"])
    .add_local_dir(str(SERVER / "tools"), "/root/server/tools", ignore=["__pycache__", "scores", "data/gen_en"])
)
app = modal.App("ai-text-checker-train-chinese")
out_volume = modal.Volume.from_name("ai-text-checker-train-out", create_if_missing=True)


@app.function(image=image, gpu="T4", timeout=3 * 3600, cpu=4, memory=16384, volumes={"/out": out_volume})
def train(args: list[str]) -> str:
    import subprocess
    import sys
    import tarfile
    subprocess.run([sys.executable, "-u", "/root/server/tools/train_chinese.py", "--nlpcc", "/nlpcc/data",
                    "--out", "/tmp/chinese-classifier", *args], check=True, cwd="/root/server")
    with tarfile.open("/out/chinese-classifier.tar.gz", "w:gz") as t:
        t.add("/tmp/chinese-classifier", arcname="chinese-classifier")
    out_volume.commit()
    return "chinese-classifier.tar.gz"


@app.local_entrypoint()
def main(args: str = ""):
    name = train.remote(args.split())
    with open(name, "wb") as fh:
        for chunk in out_volume.read_file(name):
            fh.write(chunk)
    print(f"已保存 {name}（{Path(name).stat().st_size / 1e6:.0f} MB）")
