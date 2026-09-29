"""训练"文言专用"AI 检测分类器（在 GitHub Actions 的 CPU 上运行，见 .github/workflows/train-classical.yml）。

为什么需要：通用中文分类器（MPU）和语言模型信号是在白话文上建立的；对 DeepSeek / Kimi / 文心一言
仿写的聊斋体、笔记体文言，它们给出的分数与真人古籍大量重叠，调阈值无法分开（见 tools/EVAL_REPORT.md）。
各大检测平台的做法是用"新模型写的同类文本"持续训练专门的分类器，这里照此处理文言。

数据（三者互不重叠）：
  人写 · 训练：NiuTrans 古文语料中 TRAIN_BOOKS 列出的书（不与校准、评估用的书重叠）
  人写 · 评估：evaluate.py 的 CLASSICAL_TEST_BOOKS（含《聊斋志异》《搜神记》《唐传奇》等志怪传奇，专门检查会不会冤枉真人仿古）
  AI：tools/data/ai_classical_*.txt 与 ai_user_classical_*.txt；每个来源按顺序每 3 篇留 1 篇作评估，
      划分规则与 evaluate.py 一致，评估用的 AI 文言从不参与训练。
人写段落按 AI 样本的长度分布截取，避免"长度"成为区分线索；两类文字都去掉引号（古籍语料排印时不用引号，
AI 文言几乎都带引号，不去掉的话模型会学成"有引号就是 AI"，第一版就因此把带引号的《搜神记》段落误判为 AI）。

用法：python tools/train_classical.py --classical-dir <NiuTrans 目录> --out <输出目录>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402

TRAIN_BOOKS = ["搜神后记", "新齐谐", "博物志", "神仙传", "西京杂记", "古今谭概", "笑林广记", "郁离子", "夜航船",
               "五代新说", "高士传", "洛阳伽蓝记", "清代名人轶事", "万历野获编", "龙川别志", "江南野史", "东观奏记",
               "南唐书", "唐才子传", "浮生六记", "小窗幽记", "岭外代答", "吴船录", "庐山记", "晋书", "旧唐书",
               "北史", "三国志", "后汉书", "随园诗话", "闲情偶寄", "列女传"]


def ai_rows():
    """所有 AI 文言样本，带来源和训练 / 评估划分（与 evaluate.py 相同：每个来源内 i % 3 == 2 留作评估）。"""
    rows = []
    repo = []
    for f in sorted(ev.DATA_DIR.glob("ai_classical_*.txt")):
        repo += ev.read_blocks(f)
    rows += [{"text": t, "y": 1, "model": "repo-llm-classical", "split": "test" if i % 3 == 2 else "train"}
             for i, t in enumerate(repo)]
    for r in ev.user_ai_rows("classical"):
        rows.append({**r, "split": "test" if r["split"] == "test" else "train"})
    return rows


LONG_CHECK_BOOKS = ["聊斋志异", "唐传奇", "阅微草堂笔记", "太平广记", "搜神记", "剪灯新话", "资治通鉴", "世说新语"]


def long_passages(root, per_book=40):
    """专门检查"会不会冤枉真人志怪传奇"：从这些书（《聊斋》《唐传奇》只取评估那一半篇目）截 120–220 字的连续段落。
    这些段落都没有参与训练；比按 AI 长度截取的短段更接近用户实际粘贴的文字。"""
    import glob
    import re
    rnd = random.Random(7)
    out = []
    for b in LONG_CHECK_BOOKS:
        files = sorted(glob.glob(os.path.join(root, "古文原文", b, "**", "text.txt"), recursive=True))
        if b in ev.SPLIT_BOOKS:
            files = [f for f in files if ev.file_half(f) == "eval"]
        rnd.shuffle(files)
        got = 0
        for f in files:
            t = "".join(l.strip() for l in Path(f).read_text("utf-8", errors="ignore").splitlines())
            t = re.sub(r"（出《[^》]*》）|\(出《[^》]*》\)", "", t)
            if len(t) < 160:
                continue
            st = rnd.randint(0, max(0, len(t) - 220))
            out.append({"text": t[st:st + rnd.randint(120, 220)], "y": 0, "model": b})
            got += 1
            if got >= per_book:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classical-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="hfl/chinese-roberta-wwm-ext")
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--n-human-train", type=int, default=900)
    ap.add_argument("--n-human-test", type=int, default=400)
    ap.add_argument("--n-split-train", type=int, default=400, help="《聊斋志异》《唐传奇》训练一半篇目中取多少段")
    ap.add_argument("--time-budget-min", type=float, default=240)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.set_num_threads(int(os.getenv("TORCH_THREADS", os.cpu_count() or 2)))
    torch.manual_seed(0)
    rnd = random.Random(0)

    from app.segmenter import normalize_classical
    ai = [dict(r, text=normalize_classical(r["text"])) for r in ai_rows()]
    ai_train = [r for r in ai if r["split"] == "train"]
    ai_test = [r for r in ai if r["split"] == "test"]
    lengths = [len(r["text"]) for r in ai]
    h_train = ev.classical_passages(args.classical_dir, TRAIN_BOOKS, args.n_human_train, 31, lengths)
    # AI 最常模仿《聊斋志异》《唐传奇》：用它们一半的篇目作人写训练样本（另一半只用于评估），
    # 否则模型会把"聊斋风格"本身当成 AI 特征（第二版在较长的聊斋原文上误判约 40%）
    h_train += ev.classical_passages(args.classical_dir, sorted(ev.SPLIT_BOOKS), args.n_split_train, 33,
                                     lengths + [180] * len(lengths), half="train")
    h_test = ev.classical_passages(args.classical_dir, ev.CLASSICAL_TEST_BOOKS, args.n_human_test, 32, lengths)
    long_test = long_passages(args.classical_dir)
    h_train = [dict(r, text=normalize_classical(r["text"])) for r in h_train]
    h_test = [dict(r, text=normalize_classical(r["text"])) for r in h_test]
    long_test = [dict(r, text=normalize_classical(r["text"])) for r in long_test]
    if len(h_train) < 60 or len(h_test) < 30:
        raise SystemExit(f"人写古籍段落太少（训练 {len(h_train)} / 评估 {len(h_test)}），请检查数据下载")

    # AI 样本少：每个 epoch 里按人写数量过采样 AI，使两类权重相当
    rnd.shuffle(h_train); rnd.shuffle(ai_train)
    dev_h, h_train = h_train[: len(h_train) // 8], h_train[len(h_train) // 8:]
    dev_a, ai_train = ai_train[: max(8, len(ai_train) // 8)], ai_train[max(8, len(ai_train) // 8):]
    dev = dev_h + dev_a
    reps = max(1, round(len(h_train) / max(1, len(ai_train))))
    train = h_train + ai_train * reps
    rnd.shuffle(train)
    print(f"训练：人写 {len(h_train)} / AI {len(ai_train)}（×{reps} 过采样）；开发：{len(dev_h)} / {len(dev_a)}；"
          f"评估：人写 {len(h_test)}（{len(ev.CLASSICAL_TEST_BOOKS)} 部书）/ AI {len(ai_test)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=2, id2label={0: "human", 1: "ai"}, label2id={"human": 0, "ai": 1})

    def batches(data, bs, shuffle):
        idx = list(range(len(data)))
        if shuffle:
            rnd.shuffle(idx)
        for i in range(0, len(idx), bs):
            chunk = [data[j] for j in idx[i:i + bs]]
            enc = tok([r["text"] for r in chunk], truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
            yield enc, torch.tensor([r["y"] for r in chunk])

    def predict(data):
        model.eval()
        out = []
        with torch.inference_mode():
            for enc, _ in batches(data, 64, False):
                out.extend(torch.softmax(model(**enc).logits.float(), -1)[:, 1].tolist())
        model.train()
        return out

    auroc = lambda pos, neg: ev.auroc(pos, neg)
    steps_per_epoch = math.ceil(len(train) / args.batch)
    total = max(1, int(steps_per_epoch * args.epochs))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(total * 0.06), total)
    model.train()
    t0 = time.time()
    step, best, best_state = 0, -1.0, None
    evals = sorted({int(total * f) for f in (0.25, 0.5, 0.75, 1.0)})
    done = False
    while not done:
        for enc, y in batches(train, args.batch, True):
            loss = torch.nn.functional.cross_entropy(model(**enc).logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 50 == 0:
                el = time.time() - t0
                print(f"  step {step}/{total} loss {loss.item():.4f}  {el/60:.1f} 分钟（剩余约 {el/step*(total-step)/60:.0f} 分钟）", flush=True)
            over_budget = (time.time() - t0) / 60 > args.time_budget_min
            if step in evals or step >= total or over_budget:
                p = predict(dev)
                a = auroc([q for q, r in zip(p, dev) if r["y"] == 1], [q for q, r in zip(p, dev) if r["y"] == 0])
                print(f"  开发集 AUROC {a:.4f}（step {step}）", flush=True)
                if a > best:
                    best = a
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if step >= total or over_budget:
                done = True
                break

    model.load_state_dict(best_state)
    ph = predict(h_test)
    pa = predict(ai_test)
    by_ai, by_book = {}, {}
    for p, r in zip(pa, ai_test):
        by_ai.setdefault(r["model"], []).append(p)
    for p, r in zip(ph, h_test):
        by_book.setdefault(r["model"], []).append(p)
    # 固定一个"人写误判约 5%"的参考阈值，报告每个来源的检出率 / 每部书的误判率
    thr = sorted(ph)[int(len(ph) * 0.95)] if ph else 0.5
    pl = predict(long_test)
    by_long = {}
    for p, r in zip(pl, long_test):
        by_long.setdefault(r["model"], []).append(p)
    res = {"dev_auroc": round(best, 4), "test_auroc": round(auroc(pa, ph), 4),
           "test_auroc_by_ai_source": {m: round(auroc(v, ph), 4) for m, v in by_ai.items()},
           "reference_threshold_at_5pct_fpr": round(thr, 4),
           "ai_caught_at_ref_threshold": {m: round(sum(x >= thr for x in v) / len(v), 3) for m, v in by_ai.items()},
           "long_passages_flagged_at_ref_threshold": {b: round(sum(x >= thr for x in v) / len(v), 3) for b, v in by_long.items()},
           "long_passages_flagged_at_0.5": {b: round(sum(x >= 0.5 for x in v) / len(v), 3) for b, v in by_long.items()},
           "human_flagged_by_book_at_ref_threshold": {b: round(sum(x >= thr for x in v) / len(v), 3)
                                                     for b, v in by_book.items() if len(v) >= 5},
           "n_train_human": len(h_train), "n_train_ai": len(ai_train), "n_test_human": len(ph), "n_test_ai": len(pa),
           "base": args.base, "epochs": args.epochs, "lr": args.lr, "steps": step,
           "minutes": round((time.time() - t0) / 60, 1)}
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    print("::notice title=文言分类器评估（没参与训练的古籍与 AI 样本）::" + json.dumps(
        {k: res[k] for k in ("test_auroc", "test_auroc_by_ai_source", "ai_caught_at_ref_threshold",
                             "long_passages_flagged_at_ref_threshold")}, ensure_ascii=False), flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.half().save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    (out / "training_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    (out / "README.md").write_text(
        "# 文言 AI 检测分类器\n\n由 tools/train_classical.py 微调 "
        f"{args.base} 得到，标签 0 = 人写、1 = AI。\n\n评估结果：\n\n```json\n"
        + json.dumps(res, ensure_ascii=False, indent=1) + "\n```\n", "utf-8")


if __name__ == "__main__":
    main()
