"""有监督分类器：默认使用 MPU 中文检测模型 yuchuantian/AIGC_detector_zhv3
（Tian et al., "Multiscale Positive-Unlabeled Detection of AI-Generated Texts", ICLR 2024）。"""
from __future__ import annotations

import logging
import re
import threading

from .. import config

log = logging.getLogger("classifier")

_AI_WORDS = re.compile(r"(^|[^a-z])(ai|machine|gpt|chatgpt|generated|fake|llm|aigc|robot)([^a-z]|$)", re.I)
_HUMAN_WORDS = re.compile(r"(human|real|人类|人工)", re.I)


class Classifier:
    def __init__(self):
        self.ready = False
        self.error: str | None = None
        self.ai_index = 1
        self.labels: dict = {}
        self._lock = threading.Lock()

    def load(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(config.CLASSIFIER_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.CLASSIFIER_MODEL).eval()
        self.labels = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        self.ai_index = self._resolve_ai_index()
        log.info("classifier labels=%s ai_index=%s", self.labels, self.ai_index)
        self.ready = True

    def _resolve_ai_index(self) -> int:
        setting = config.CLASSIFIER_AI_LABEL.strip()
        if setting and setting.lower() != "auto":
            if setting.isdigit():
                return int(setting)
            for i, name in self.labels.items():
                if name.lower() == setting.lower():
                    return i
        for i, name in self.labels.items():
            if _AI_WORDS.search(name) and not _HUMAN_WORDS.search(name):
                return i
        for i, name in self.labels.items():
            if _HUMAN_WORDS.search(name) and len(self.labels) == 2:
                return 1 - i
        return 1 if len(self.labels) > 1 else 0

    def predict(self, texts: list[str], batch_size: int = 8) -> list[float]:
        torch = self.torch
        out: list[float] = []
        with self._lock, torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                enc = self.tok(texts[i:i + batch_size], truncation=True, max_length=config.CLS_MAX_TOKENS,
                               padding=True, return_tensors="pt")
                probs = torch.softmax(self.model(**enc).logits.float(), dim=-1)
                out.extend(probs[:, self.ai_index].tolist())
        return out
