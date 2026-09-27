"""零样本检测：Binoculars（ICML 2024）与 Fast-DetectGPT（ICLR 2024）。

两种方法都只需要"观察者"(基础模型) 和 "执行者"(对话模型) 各做一次前向计算，
因此共用同一次推理结果，一起算出。

记号：对文本的第 t 个位置，
  O_t = softmax(observer 的 logits)          观察者预测的下一个字的分布
  L_t = log_softmax(performer 的 logits)     执行者的对数概率
  x_t = 实际出现的下一个 token

Binoculars（按官方实现 ahans30/Binoculars）：
  ppl   = mean_t( -L_t[x_t] )                执行者对原文的对数困惑度
  x_ppl = mean_t( -Σ_v O_t[v] · L_t[v] )      观察者分布与执行者之间的交叉困惑度
  B     = ppl / x_ppl                        越低越像 AI

Fast-DetectGPT（解析版，按官方实现 baoguangsheng/fast-detect-gpt，采样模型=观察者，打分模型=执行者）：
  μ_t   = Σ_v O_t[v] · L_t[v]
  σ²_t  = Σ_v O_t[v] · L_t[v]² − μ_t²
  D     = ( Σ_t L_t[x_t] − Σ_t μ_t ) / sqrt( Σ_t σ²_t )   越高越像 AI
"""
from __future__ import annotations

import logging
import math
import threading

from .. import config

log = logging.getLogger("lm_scorer")


class LMScorer:
    def __init__(self):
        self.ready = False
        self.error: str | None = None
        self._lock = threading.Lock()

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(config.TORCH_THREADS)
        self.torch = torch
        log.info("loading tokenizer %s", config.OBSERVER_MODEL)
        self.tok = AutoTokenizer.from_pretrained(config.OBSERVER_MODEL)
        tok2 = AutoTokenizer.from_pretrained(config.PERFORMER_MODEL)
        if self.tok.get_vocab() != tok2.get_vocab():
            raise RuntimeError("OBSERVER_MODEL 与 PERFORMER_MODEL 的分词器不一致，必须使用同一系列的基础版与对话版模型")
        kw = dict(torch_dtype=torch.float32, low_cpu_mem_usage=True)
        log.info("loading observer %s", config.OBSERVER_MODEL)
        self.observer = AutoModelForCausalLM.from_pretrained(config.OBSERVER_MODEL, **kw).eval()
        log.info("loading performer %s", config.PERFORMER_MODEL)
        self.performer = AutoModelForCausalLM.from_pretrained(config.PERFORMER_MODEL, **kw).eval()
        # 两个模型的输出维度可能因填充而不同，只比较共同的词表部分
        self.vocab = min(self.observer.config.vocab_size, self.performer.config.vocab_size, len(self.tok))
        self.ready = True

    def _chunks(self, ids: list[int]):
        n = config.LM_MAX_TOKENS
        if len(ids) <= n:
            yield ids
            return
        # 按固定长度切块；每块单独计算，最后按 token 数汇总
        for i in range(0, len(ids), n - 1):
            chunk = ids[i:i + n]
            if len(chunk) >= 8:
                yield chunk

    def score(self, text: str) -> dict | None:
        """返回 {'binoculars', 'fastdetect', 'ppl', 'x_ppl', 'tokens'}；文本太短返回 None。"""
        return self.score_ids(self.tok(text, add_special_tokens=False)["input_ids"])

    def score_ids(self, ids: list[int]) -> dict | None:
        torch = self.torch
        if len(ids) < 16:
            return None
        sum_lp = sum_mu = sum_var = sum_xent = 0.0
        n_tok = 0
        with self._lock, torch.inference_mode():
            for chunk in self._chunks(ids):
                inp = torch.tensor([chunk])
                o = self.observer(input_ids=inp).logits[0, :-1, : self.vocab].float()
                p = self.performer(input_ids=inp).logits[0, :-1, : self.vocab].float()
                labels = inp[0, 1:]
                lp_perf = torch.log_softmax(p, dim=-1)
                del p
                pr_obs = torch.softmax(o, dim=-1)
                del o
                ll = lp_perf.gather(-1, labels.unsqueeze(-1)).squeeze(-1)       # L_t[x_t]
                mu = (pr_obs * lp_perf).sum(-1)                                  # μ_t
                var = (pr_obs * lp_perf.square()).sum(-1) - mu.square()          # σ²_t
                sum_lp += ll.sum().item()
                sum_mu += mu.sum().item()
                sum_var += var.clamp_min(0).sum().item()
                sum_xent += (-mu).sum().item()
                n_tok += labels.numel()
                del lp_perf, pr_obs, ll, mu, var
        if n_tok == 0:
            return None
        ppl = -sum_lp / n_tok
        x_ppl = sum_xent / n_tok
        return {
            "binoculars": ppl / x_ppl if x_ppl > 0 else None,
            "fastdetect": (sum_lp - sum_mu) / math.sqrt(sum_var) if sum_var > 0 else None,
            "ppl": ppl,
            "x_ppl": x_ppl,
            "tokens": n_tok,
        }
