FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

# Hugging Face Spaces 以 uid 1000 运行容器
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# 只装 CPU 版 PyTorch，镜像小很多
RUN pip install --upgrade pip && \
    pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

USER user
# 构建时就把模型下载进镜像，Space 启动/唤醒时不用再下载
COPY --chown=user download_models.py .
ARG OBSERVER_MODEL=Qwen/Qwen2.5-0.5B
ARG PERFORMER_MODEL=Qwen/Qwen2.5-0.5B-Instruct
ARG CLASSIFIER_MODEL=yuchuantian/AIGC_detector_zhv3
ARG EN_CLASSIFIER_MODEL=desklib/ai-text-detector-v1.01
ARG POETRY_URL=https://github.com/zhuloujun/Ai-text-checker/releases/download/poetry-classifier-v1/poetry-classifier.tar.gz
ENV POETRY_CLASSIFIER_MODEL=/home/user/app/poetry-classifier
RUN python download_models.py "$OBSERVER_MODEL" "$PERFORMER_MODEL" "$CLASSIFIER_MODEL" "$EN_CLASSIFIER_MODEL" "$POETRY_URL=$POETRY_CLASSIFIER_MODEL"

COPY --chown=user . .
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
