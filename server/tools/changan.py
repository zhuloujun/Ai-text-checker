"""ChangAn 诗词数据集（ACL 2026，https://github.com/VelikayaScarlet/ChangAn，MIT）的读取与划分。

训练诗词分类器（tools/train_poetry.py）、校准和评估（tools/evaluate.py）共用同一套划分，保证互不重叠：
  - 人写诗词按"作者"划分：同一作者的作品只会出现在一份里；
  - AI 诗词按"题目编号"划分：同一题目下两种生成策略的作品在同一份里；
  - Kimi-K2 生成的诗词不参与训练和校准，只用于评估（检验对"没见过的 AI 模型"的效果）。
比例：训练 70% · 校准 10% · 评估 20%。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

UNSEEN_MODEL = "kimi-k2"


def split_of(key: str) -> str:
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 10
    return "train" if h < 7 else ("cal" if h == 7 else "test")


def _read_xlsx(path):
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True)
    rows = list(wb.worksheets[0].iter_rows(values_only=True))
    head = [str(h).strip() if h is not None else "" for h in rows[0]]
    return [dict(zip(head, r)) for r in rows[1:]]


def load(changan_dir) -> list[dict]:
    """返回 [{text, y(0 人写 / 1 AI), model, split}]。"""
    d = Path(changan_dir) / "data"
    out = []
    for r in _read_xlsx(d / "Untouched_Classical_Poetry.xlsx"):
        t = str(r.get("Text") or "").strip()
        if len(t) >= 16 and t != "None":
            out.append({"text": t, "y": 0, "model": "human", "split": split_of("h:" + str(r.get("Author")))})
    for r in _read_xlsx(d / "AIGen_Cleaned.xlsx"):
        t = str(r.get("ai_content") or "").strip()
        m = str(r.get("author") or "")
        if len(t) >= 16 and t != "None":
            sp = "test" if m.lower() == UNSEEN_MODEL else split_of("a:" + str(r.get("no")))
            out.append({"text": t, "y": 1, "model": m, "split": sp})
    return out
