"""在 Modal（https://modal.com）上运行检测服务。

部署：modal deploy modal_app.py （GitHub Actions 会自动执行，见仓库根目录 README）
网址：https://zhuloujun--ai-text-checker-web.modal.run

计费说明：Modal 每月送 $30 免费额度（需绑定付款方式；未绑定时为 $1），只在容器运行时计费。
没人访问时容器会在 scaledown_window（10 分钟）后自动关闭，不再计费；
下次访问会自动启动，约需 1–2 分钟加载模型（加载完成前提交的检测会排队等待）。
"""
from pathlib import Path

import modal

OBSERVER_MODEL = "Qwen/Qwen2.5-0.5B"
PERFORMER_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CLASSIFIER_MODEL = "yuchuantian/AIGC_detector_zhv3"
EN_CLASSIFIER_MODEL = "desklib/ai-text-detector-v1.01"   # 英文分类器（DeBERTa-v3-large，约 1.7 GB）
# 诗词专用分类器：由本仓库 .github/workflows/train-poetry.yml 训练并发布在 Release。
# 仓库是私有的，部署工作流会先用 GitHub 令牌把它下载解压到 server/poetry-classifier，再随镜像上传。
POETRY_LOCAL = Path(__file__).resolve().parent / "poetry-classifier"
POETRY_DIR = "/models/poetry-classifier"
HAS_POETRY = (POETRY_LOCAL / "config.json").exists()
# 文言专用分类器（tools/train_classical.py 训练，Release classical-classifier-v3）：部署工作流按需下载到 server/classical-classifier
CLASSICAL_LOCAL = Path(__file__).resolve().parent / "classical-classifier"
CLASSICAL_DIR = "/models/classical-classifier"
HAS_CLASSICAL = (CLASSICAL_LOCAL / "config.json").exists()
CPU_CORES = 8

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install_from_requirements("requirements.txt")
    .env({
        "HF_HOME": "/models",
        "TOKENIZERS_PARALLELISM": "false",
        "TORCH_THREADS": str(CPU_CORES),
        "OBSERVER_MODEL": OBSERVER_MODEL,
        "PERFORMER_MODEL": PERFORMER_MODEL,
        "CLASSIFIER_MODEL": CLASSIFIER_MODEL,
        "EN_CLASSIFIER_MODEL": EN_CLASSIFIER_MODEL,
        "POETRY_CLASSIFIER_MODEL": POETRY_DIR if HAS_POETRY else "",
        "CLASSICAL_CLASSIFIER_MODEL": CLASSICAL_DIR if HAS_CLASSICAL else "",
        "USER_CALIBRATION_FILE": "/data/user_calibration.json",
        "USER_LABELS_FILE": "/data/user_labels.json",
        "CALIBRATION_VOLUME": "ai-text-checker-data",
    })
    # 构建镜像时就把模型下载进去，启动时不用再下载
    .add_local_file("download_models.py", "/root/download_models.py", copy=True)
    .run_commands(f"python /root/download_models.py {OBSERVER_MODEL} {PERFORMER_MODEL} {CLASSIFIER_MODEL} {EN_CLASSIFIER_MODEL}")
    .add_local_dir("app", "/root/app")
    .add_local_dir("static", "/root/static")
)
if HAS_POETRY:
    image = image.add_local_dir(str(POETRY_LOCAL), POETRY_DIR)
if HAS_CLASSICAL:
    image = image.add_local_dir(str(CLASSICAL_LOCAL), CLASSICAL_DIR)

app = modal.App("ai-text-checker")
# 持久卷：保存管理页“用我的标注校准”的结果，服务重启 / 重新部署后仍然有效
CALIB_VOLUME_NAME = "ai-text-checker-data"
calib_volume = modal.Volume.from_name(CALIB_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    cpu=CPU_CORES,
    memory=12288,                 # MB
    timeout=3600,
    min_containers=0,             # 没人用时不保留容器，不计费
    max_containers=1,             # 只用一个容器：任务队列、用量统计都在内存里
    scaledown_window=600,         # 最后一次访问 10 分钟后关闭
    volumes={"/data": calib_volume},
    secrets=[modal.Secret.from_name("ai-text-checker")],   # 含 ADMIN_TOKEN
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    import sys
    sys.path.insert(0, "/root")
    from app.main import app as fastapi_app
    return fastapi_app
