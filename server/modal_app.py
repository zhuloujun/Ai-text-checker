"""在 Modal（https://modal.com）上运行检测服务。

部署：modal deploy modal_app.py （GitHub Actions 会自动执行，见仓库根目录 README）
网址：https://zhuloujun--ai-text-checker-web.modal.run

计费说明：Modal 每月送 $30 免费额度（需绑定付款方式；未绑定时为 $1），只在容器运行时计费。
没人访问时容器会在 scaledown_window（10 分钟）后自动关闭，不再计费；
下次访问会自动启动，约需 1–2 分钟加载模型（加载完成前提交的检测会排队等待）。
"""
import modal

OBSERVER_MODEL = "Qwen/Qwen2.5-0.5B"
PERFORMER_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CLASSIFIER_MODEL = "yuchuantian/AIGC_detector_zhv3"
EN_CLASSIFIER_MODEL = "desklib/ai-text-detector-v1.01"   # 英文分类器（DeBERTa-v3-large，约 1.7 GB）
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
    })
    # 构建镜像时就把模型下载进去，启动时不用再下载
    .add_local_file("download_models.py", "/root/download_models.py", copy=True)
    .run_commands(f"python /root/download_models.py {OBSERVER_MODEL} {PERFORMER_MODEL} {CLASSIFIER_MODEL} {EN_CLASSIFIER_MODEL}")
    .add_local_dir("app", "/root/app")
    .add_local_dir("static", "/root/static")
)

app = modal.App("ai-text-checker")


@app.function(
    image=image,
    cpu=CPU_CORES,
    memory=12288,                 # MB
    timeout=3600,
    min_containers=0,             # 没人用时不保留容器，不计费
    max_containers=1,             # 只用一个容器：任务队列、用量统计都在内存里
    scaledown_window=600,         # 最后一次访问 10 分钟后关闭
    secrets=[modal.Secret.from_name("ai-text-checker")],   # 含 ADMIN_TOKEN
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    import sys
    sys.path.insert(0, "/root")
    from app.main import app as fastapi_app
    return fastapi_app
