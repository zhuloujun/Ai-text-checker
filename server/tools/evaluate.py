"""用公开数据集评估检测效果，并为三种文体分别生成默认校准参数。

文体与数据：
  现代汉语  NLPCC 2025 Shared Task 1（https://github.com/NLP2CT/NLPCC-2025-Task1）
            CSL 学术摘要 / 新闻 / 作文；AI 文本由 GPT-4o、GLM-4、Qwen 生成，测试集另含 DeepSeek-V3。
  英文      MAGE（https://huggingface.co/datasets/yaful/MAGE，Apache-2.0）
            校准：test.csv 中人写文本与较强模型（GPT-3.5 / GPT-4 / text-davinci / 大参数 LLaMA 等）的生成文本；
            独立评估：两个"更难"的测试集——GPT-4 在未见过的领域生成的文本（test_ood_set_gpt），
            以及同一批文本经过改写的版本（test_ood_set_gpt_para）；另加本仓库自带的英文 AI 样本（含古文英译）。
  现代汉语（短段）  同一批 NLPCC 样本截成 80–260 字（在句末截断），让校准覆盖网页上常见的短段落；
            评估另加本仓库自带的 AI 读后感 / 散文 / 随笔样本（tools/data/ai_zh_essays.txt），检验跨文体效果。
  诗词      ChangAn（ACL 2026，https://github.com/VelikayaScarlet/ChangAn，MIT）：当代人写的旧体诗词 7,707 首，
            DeepSeek、豆包（Seed）、GPT-4.1、Kimi-K2 生成的诗词 2 万余首。按作者划分人写的校准 / 评估集；
            Kimi-K2 整个模型只用于评估（检验对"没见过的 AI 模型"的效果）。
  文言      人写：NiuTrans Classical-Modern 古文原文（https://github.com/NiuTrans/Classical-Modern，MIT），
            笔记、志怪、传奇、史传、游记等；校准用一组书，独立评估用另一组书（训练时完全没见过）。
            AI：tools/data/ai_classical_*.txt（由大语言模型生成的 120 段文言，体裁覆盖志怪、传奇、史传、笔记、
            考证、游记、序跋、书信、奏议、史论、墓志、公文），按 2:1 分成校准和评估两份。

流程：
  1. score：各份样本分别打分（可在多台机器上并行），原始分数写到 --scores-dir/<份名>.json
  2. fit：读取打分结果，每种文体各自拟合逻辑回归并用交叉验证选阈值，在没见过的评估集上测检出率与误判率，
     写出 app/default_calibration.json、tools/EVAL_REPORT.md、tools/eval_result.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config, scoring  # noqa: E402
from app.engine import Engine  # noqa: E402
from app.segmenter import segment_text  # noqa: E402

DATA_DIR = ROOT / "tools" / "data"

# 每份属于哪种文体
PART_PROFILE = {
    "cal1": "zh", "cal2": "zh", "test": "zh", "csl": "zh",
    "cal1s": "zh", "cal2s": "zh", "tests": "zh",
    "po_cal": "zh_poetry", "po_test": "zh_poetry",
    "en_cal1": "en", "en_cal2": "en", "en_ood": "en", "en_para": "en",
    "cl_cal": "zh_classical", "cl_test": "zh_classical",
}
PROFILE_PARTS = {
    "zh": {"fit": ["cal1", "cal2"],
           "eval": [("test", "NLPCC 测试集（训练时未见，含 DeepSeek-V3）"),
                    ("csl", "CSL 学术摘要保留集")]},
    # 短段（不足 200 字）的信号分布与长段不同，单独校准、单独定阈值，否则短的人写段落误判偏多
    "zh_short": {"fit": ["cal1s", "cal2s"],
                 "eval": [("tests", "NLPCC 测试集截成 80–260 字的短段（含本仓库 AI 读后感 / 散文）")]},
    "zh_poetry": {"fit": ["po_cal"], "eval": [("po_test", "ChangAn 保留集（另一批作者 + 没见过的 Kimi-K2 与其他模型的新诗词）")]},
    "en": {"fit": ["en_cal1", "en_cal2"], "eval": [("en_ood", "MAGE：GPT-4 在未见过的领域生成的文本"),
                                                  ("en_para", "MAGE：GPT-4 文本经改写后（含本仓库英文 AI 样本）")]},
    "zh_classical": {"fit": ["cl_cal"], "eval": [("cl_test", "文言保留集（另一组古籍 + 未参与校准的 AI 文言）")]},
}
PROFILE_SOURCE = {
    "zh": "NLPCC 2025 Task 1（CSL 学术摘要 / 新闻 / 作文；GPT-4o、GLM-4、Qwen）",
    "en": "MAGE（人写文本与 GPT-3.5 / GPT-4 等生成文本）",
    "zh_classical": "NiuTrans 古文语料（人写）+ 大语言模型生成的文言样本",
    "zh_poetry": "ChangAn 当代旧体诗词（人写）+ DeepSeek / 豆包 / GPT-4.1 生成诗词",
    "zh_short": "NLPCC 2025 Task 1 样本截成 80–260 字的短段",
}

# 文言：校准用的书 / 评估用的书（互不重叠）
CLASSICAL_CAL_BOOKS = ["太平广记", "夷坚志", "阅微草堂笔记", "剪灯新话", "容斋随笔", "梦溪笔谈", "宋史", "元史", "明史",
                       "续资治通鉴", "东京梦华录", "徐霞客游记", "日知录", "世说新语", "酉阳杂俎", "明季北略"]
CLASSICAL_TEST_BOOKS = ["聊斋志异", "唐传奇", "搜神记", "新唐书", "金史", "资治通鉴", "困学纪闻", "武林旧事", "入蜀记",
                        "陶庵梦忆", "明夷待访录", "幽明录", "旧五代史", "西湖梦寻"]

# 英文：MAGE 里较强的生成模型（src 字段中的名称片段）
EN_STRONG = re.compile(r"gpt-3\.5|gpt_3\.5|gpt-4|gpt4|text-davinci|davinci-00[23]|65b|30b|_13b|70b", re.I)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def balanced_sample(rows, n, keyfn, seed):
    groups = {}
    for r in rows:
        groups.setdefault(keyfn(r), []).append(r)
    if not groups:
        return []
    rnd = random.Random(seed)
    per = max(1, n // len(groups))
    out = []
    for k in sorted(groups):
        g = groups[k][:]
        rnd.shuffle(g)
        out.extend(g[:per])
    return out


def read_blocks(path):
    """读取 tools/data 下以单独一行 === 分隔、# 开头为注释的样本文件。"""
    out = []
    for block in re.split(r"\n===\n", Path(path).read_text("utf-8")):
        lines = [l for l in block.strip().splitlines() if not l.startswith("#")]
        t = "\n".join(lines).strip()
        if t:
            out.append(t)
    return out


def to_segment(text, profile):
    """与正式检测一致：按同样规则分段，取最长的一段正文。
    现代汉语沿用原来的引文规则（与已提交的打分结果一致）；文言、诗词与英文样本本身就是正文，不做引文排除。
    诗词整首作为一段（与网页上"一首诗一段"一致）。"""
    if profile == "zh_poetry":
        return text.strip()
    segs = [s.text for s in segment_text(text, True, profile == "zh") if s.counted]
    if not segs:
        return None
    return max(segs, key=len)


# ---------------- 各文体的数据 ----------------

def zh_parts(args):
    train = load(Path(args.data) / "train.json")
    test = load(Path(args.data) / "test_with_label.json")
    human = [r for r in train if r["label"] == 0]
    ai = [r for r in train if r["label"] == 1]
    weights = {"csl": 0.5, "cnewsum": 0.25, "asap": 0.25}
    cal_h, cal_a = [], []
    for dom, w in weights.items():
        k = int(args.n_cal / 2 * w)
        cal_h += balanced_sample([r for r in human if r["source"] == dom], k, lambda r: r["source"], 1)
        cal_a += balanced_sample([r for r in ai if r["source"] == dom], k, lambda r: r["model"], 2)
    cal = cal_h + cal_a
    random.Random(7).shuffle(cal)
    test_s = balanced_sample([r for r in test if len(r["text"]) >= 150], args.n_test, lambda r: r["label"], 3)
    used = {id(r) for r in cal}
    csl_hold = balanced_sample([r for r in train if r["source"] == "csl" and id(r) not in used],
                               args.n_csl, lambda r: (r["label"], r["model"]), 4)
    half = len(cal) // 2
    conv = lambda rows: [{"text": r["text"], "y": r["label"], "model": r.get("model") or ""} for r in rows]
    return {"cal1": conv(cal[:half]), "cal2": conv(cal[half:]), "test": conv(test_s), "csl": conv(csl_hold)}


def truncate_short(text, rnd):
    """截成 80–260 字，在句末截断（模拟网页上一两段的短窗口）。"""
    target = rnd.randint(80, 260)
    if len(text) <= target:
        return text
    cut = text[:target]
    m = list(re.finditer(r"[。！？；」”]", cut))
    return cut[:m[-1].end()] if m and m[-1].end() >= 60 else cut


def zh_short_parts(args):
    base = zh_parts(args)
    rnd = random.Random(31)
    out = {}
    for name in ("cal1", "cal2", "test"):
        rows = []
        for r in base[name]:
            sg = to_segment(r["text"], "zh")
            if sg:
                rows.append(dict(r, text=truncate_short(sg, rnd)))
        out[name + "s"] = rows
    out["tests"] += [{"text": t, "y": 1, "model": "repo-ai-zh-essay"} for t in read_blocks(DATA_DIR / "ai_zh_essays.txt")]
    return out


def read_xlsx(path, sheet=0):
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True)
    rows = list(wb.worksheets[sheet].iter_rows(values_only=True))
    head = [str(h).strip() if h is not None else "" for h in rows[0]]
    return [dict(zip(head, r)) for r in rows[1:]]


def poetry_parts(args):
    d = Path(args.changan_dir) / "data"
    human = [{"text": str(r["Text"]).strip(), "y": 0, "model": "human", "author": str(r["Author"])}
             for r in read_xlsx(d / "Untouched_Classical_Poetry.xlsx") if r.get("Text")]
    ai = [{"text": str(r["ai_content"]).strip(), "y": 1, "model": str(r["author"])}
          for r in read_xlsx(d / "AIGen_Cleaned.xlsx") if r.get("ai_content")]
    human = [r for r in human if len(r["text"]) >= 16]
    ai = [r for r in ai if len(r["text"]) >= 16]
    # 人写：按作者分成校准 / 评估（同一作者不会同时出现在两边）
    import hashlib
    held = lambda a: int(hashlib.md5(a.encode("utf-8")).hexdigest(), 16) % 5 == 0
    h_cal = [r for r in human if not held(r["author"])]
    h_test = [r for r in human if held(r["author"])]
    rnd = random.Random(41)
    rnd.shuffle(h_cal); rnd.shuffle(h_test)
    unseen = "kimi-k2"
    seen = [r for r in ai if r["model"].lower() != unseen]
    rnd.shuffle(seen)
    a_cal = balanced_sample(seen[: len(seen) // 2], args.n_po_cal // 2, lambda r: r["model"], 42)
    a_test_seen = balanced_sample(seen[len(seen) // 2:], args.n_po_test // 4, lambda r: r["model"], 43)
    a_test_unseen = [r for r in ai if r["model"].lower() == unseen]
    rnd.shuffle(a_test_unseen)
    print(f"ChangAn：人写 {len(human)}（校准作者 {len(h_cal)} / 评估作者 {len(h_test)}），AI {len(ai)}", flush=True)
    return {"po_cal": h_cal[: args.n_po_cal // 2] + a_cal,
            "po_test": h_test[: args.n_po_test // 2] + a_test_seen + a_test_unseen[: args.n_po_test // 4]}


def read_mage(path):
    csv.field_size_limit(10 ** 8)
    rows, header, n_raw = [], None, 0
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]
        col = lambda *names: next((header.index(n) for n in names if n in header), None)
        ti, li, si = col("text", "texts"), col("label", "labels"), col("src", "source")
        if ti is None:
            ti = 0
        for r in reader:
            n_raw += 1
            if len(r) <= ti:
                continue
            text = (r[ti] or "").strip()
            src = r[si] if si is not None and si < len(r) else ""
            human = None
            if li is not None and li < len(r):
                try:
                    human = int(float(r[li])) == 1          # MAGE：1 = 人写，0 = 机器生成
                except ValueError:
                    pass
            if "human" in src:
                human = True
            elif "machine" in src:
                human = False
            if human is None:
                continue
            if len(text) >= 400:
                rows.append({"text": text, "y": 0 if human else 1, "model": src or "mage",
                             "domain": (src.split("_")[0] if src else "mage")})
    print(f"::notice title=读取 {Path(path).name}::表头 {header}，共 {n_raw} 行，可用 {len(rows)} 行"
          f"（人写 {sum(r['y'] == 0 for r in rows)} / AI {sum(r['y'] == 1 for r in rows)}）", flush=True)
    return rows


def en_parts(args):
    d = Path(args.mage_dir)
    rows = read_mage(d / "test.csv")
    human = [r for r in rows if r["y"] == 0]
    ai_all = [r for r in rows if r["y"] == 1]
    ai = [r for r in ai_all if EN_STRONG.search(r["model"])] or ai_all
    print(f"MAGE test.csv：人写 {len(human)}，AI {len(ai_all)}（其中较强模型 {len(ai)}）", flush=True)
    k = args.n_en_cal // 2
    cal_h = balanced_sample(human, k, lambda r: r["domain"], 11)
    cal_a = balanced_sample(ai, k, lambda r: r["domain"], 12)
    cal = cal_h + cal_a
    random.Random(13).shuffle(cal)
    half = len(cal) // 2
    ood = read_mage(d / "test_ood_set_gpt.csv")
    para = read_mage(d / "test_ood_set_gpt_para.csv")
    ood_s = balanced_sample(ood, args.n_en_test, lambda r: r["y"], 14)
    para_s = balanced_sample(para, args.n_en_test, lambda r: r["y"], 15)
    extra = [{"text": t, "y": 1, "model": "repo-ai-english"} for t in read_blocks(DATA_DIR / "ai_english_extra.txt")]
    return {"en_cal1": cal[:half], "en_cal2": cal[half:], "en_ood": ood_s, "en_para": para_s + extra}


def classical_passages(root, books, n, seed, lengths):
    """从古籍中抽取段落，长度按 AI 样本的长度分布截取（在句末截断），避免"长度"本身成为区分线索。"""
    rnd = random.Random(seed)
    per_book = max(3, n // len(books) + 2)
    pool = []
    for b in books:
        files = glob.glob(os.path.join(root, "古文原文", b, "**", "text.txt"), recursive=True)
        rnd.shuffle(files)
        got = 0
        for f in files:
            for line in Path(f).read_text("utf-8", errors="ignore").splitlines():
                line = re.sub(r"（出《[^》]*》）|\(出《[^》]*》\)", "", line).strip()
                if len(line) < 90:
                    continue
                target = rnd.choice(lengths)
                cut = line[:target]
                m = list(re.finditer(r"[。！？」”]", cut))
                if m and m[-1].end() >= 60:
                    cut = cut[:m[-1].end()]
                pool.append({"text": cut, "y": 0, "model": b})
                got += 1
                break
            if got >= per_book:
                break
    rnd.shuffle(pool)
    return balanced_sample(pool, n, lambda r: r["model"], seed)


def classical_parts(args):
    ai = []
    for f in sorted(glob.glob(str(DATA_DIR / "ai_classical_*.txt"))):
        ai += read_blocks(f)
    lengths = [len(t) for t in ai]
    ai_cal = [t for i, t in enumerate(ai) if i % 3 != 2]
    ai_test = [t for i, t in enumerate(ai) if i % 3 == 2]
    h_cal = classical_passages(args.classical_dir, CLASSICAL_CAL_BOOKS, args.n_cl_human, 21, lengths)
    h_test = classical_passages(args.classical_dir, CLASSICAL_TEST_BOOKS, args.n_cl_human_test, 22, lengths)
    conv = lambda ts: [{"text": t, "y": 1, "model": "llm-classical"} for t in ts]
    return {"cl_cal": h_cal + conv(ai_cal), "cl_test": h_test + conv(ai_test)}


def build_part(args, part):
    prof = PART_PROFILE[part]
    if part in ("cal1s", "cal2s", "tests"):
        return zh_short_parts(args)[part]
    if prof == "zh_poetry":
        return poetry_parts(args)[part]
    if prof == "zh":
        return zh_parts(args)[part]
    if prof == "en":
        return en_parts(args)[part]
    return classical_parts(args)[part]


# ---------------- 打分 ----------------

def stage_score(args):
    rows = build_part(args, args.part)
    prof = PART_PROFILE[args.part]
    engine = Engine()
    engine.load_all()
    st = engine.status()
    need = st["classifier_en"] if prof == "en" else st["classifier"]
    if not (st["lm"].get("ready") and need.get("ready")):
        raise SystemExit(f"模型没有全部加载成功，停止打分：{json.dumps(st, ensure_ascii=False)}")
    segs, labels, meta = [], [], []
    for r in rows:
        sg = to_segment(r["text"], prof)
        min_len = {"zh_classical": 60, "zh_poetry": 16}.get(prof, config.SEGMENT_MIN_CHARS)
        if r["y"] == 1 and r.get("model", "").startswith("repo-ai"):
            min_len = 20      # 本仓库自带的 AI 样本全部保留
        if sg and len(sg) >= min_len:
            segs.append(sg); labels.append(r["y"]); meta.append(r.get("model") or "")
    print(f"[{args.part}] 文体 {prof}，打分 {len(segs)} 段（人写 {labels.count(0)} / AI {labels.count(1)}）…", flush=True)
    t = time.time(); last = [0.0]

    def prog(d, n):
        if time.time() - last[0] > 60:
            last[0] = time.time()
            print(f"  {d}/{n}  {time.time()-t:.0f}s", flush=True)
    if not segs:
        raise SystemExit(f"[{args.part}] 没有可打分的样本（原始 {len(rows)} 条），请检查数据下载与读取")
    sc = engine.raw_scores(segs, prog, [prof] * len(segs))
    out = [{"y": l, "model": m, "chars": len(x_text), "s": {k: v for k, v in x.items() if isinstance(v, (int, float))}}
           for x, l, m, x_text in zip(sc, labels, meta, segs)]
    d = Path(args.scores_dir); d.mkdir(parents=True, exist_ok=True)
    (d / f"{args.part}.json").write_text(json.dumps(out, ensure_ascii=False), "utf-8")
    print(f"::notice title=打分完成 {args.part}::{len(out)} 段，用时 {time.time()-t:.0f} 秒", flush=True)


# ---------------- 拟合与评估 ----------------

def auroc(pos, neg):
    return scoring._auroc(pos, neg)


def evaluate_rows(rows, cal, name):
    probs = [scoring.combine(r["s"], cal)["prob"] for r in rows]
    thr = cal["threshold"]
    y = [r["y"] for r in rows]
    hp = [p for p, l in zip(probs, y) if l == 0 and p is not None]
    ap_ = [p for p, l in zip(probs, y) if l == 1 and p is not None]
    res = {"name": name, "n_human": len(hp), "n_ai": len(ap_),
           "auroc": round(auroc(ap_, hp), 4) if hp and ap_ else None,
           "ai_caught": round(sum(p >= thr for p in ap_) / len(ap_), 4) if ap_ else None,
           "human_flagged": round(sum(p >= thr for p in hp) / len(hp), 4) if hp else None,
           "per_signal_auroc": {}, "ai_caught_by_model": {}, "human_flagged_by_source": {}}
    for k in scoring.EXTENDED_FEATURES:
        pv = [scoring.feature_value(r["s"], k) for r in rows if r["y"] == 1]
        nv = [scoring.feature_value(r["s"], k) for r in rows if r["y"] == 0]
        pv = [v for v in pv if v is not None]; nv = [v for v in nv if v is not None]
        a = auroc(pv, nv)
        if a is not None:
            res["per_signal_auroc"][k] = round(a, 4)
    by, byh = {}, {}
    for p, r in zip(probs, rows):
        if p is None or not r.get("model"):
            continue
        (by if r["y"] == 1 else byh).setdefault(r["model"], []).append(p >= thr)
    if len(by) <= 12:
        res["ai_caught_by_model"] = {m: round(sum(v) / len(v), 4) for m, v in by.items()}
    if len(byh) <= 20:
        res["human_flagged_by_source"] = {m: round(sum(v) / len(v), 4) for m, v in byh.items() if len(v) >= 3}
    return res


# 各文体的人写误判率目标。短段文本信号弱、跨数据集漂移大（校准集上 2% 的误判率到测试集上会升到约 10%），
# 所以预先定得更严（与 Turnitin 对短文本从严的做法一致）。诗词同理：宁可少抓，也不冤枉写诗的人。
PROFILE_TARGET_FPR = {"zh_short": 0.01, "zh_poetry": 0.03}
# 自动选特征时，其他组合要比"全部特征"高出这么多才换（避免被交叉验证的随机波动带偏）
SELECTION_MARGIN = 0.005


def fit_profile(prof, parts, target_fpr):
    target_fpr = PROFILE_TARGET_FPR.get(prof, target_fpr)
    spec = PROFILE_PARTS[prof]
    cal_rows = [r for p in spec["fit"] for r in parts.get(p, [])]
    hs = [r["s"] for r in cal_rows if r["y"] == 0]
    as_ = [r["s"] for r in cal_rows if r["y"] == 1]
    if len(hs) < 10 or len(as_) < 10:
        return None
    fixed = scoring.PROFILE_FEATURES.get(prof)
    if fixed:
        res = scoring.calibrate(hs, as_, target_fpr, features=fixed)
    else:
        # 在校准集上用 5 折交叉验证比较几组特征，选区分能力最好的（不看评估集，避免"偷看答案"）
        tried = {}
        for name, feats in (("全部特征", scoring.EXTENDED_FEATURES), ("三个主信号", scoring.BASE_FEATURES),
                            ("语言模型特征", [f for f in scoring.EXTENDED_FEATURES if f != "logit_classifier"])):
            tried[name] = scoring.calibrate(hs, as_, target_fpr, features=feats)
        au = {k: (v["report"].get("combined_auroc") or 0) for k, v in tried.items()}
        best = "全部特征"
        for k in tried:
            if au[k] > au[best] + SELECTION_MARGIN:
                best = k
        res = tried[best]
        res["report"]["feature_selection"] = {k: v["report"].get("combined_auroc") for k, v in tried.items()}
        res["report"]["feature_set"] = best
    cal = res["calibration"]
    cal["models"] = {"observer": config.OBSERVER_MODEL, "performer": config.PERFORMER_MODEL,
                     "classifier": config.classifier_for(prof)}
    cal["source"] = PROFILE_SOURCE[prof]
    cal["note"] = (f"内置默认校准（{scoring.PROFILE_NAMES[prof]}）：用公开数据 {len(hs)} 段人写、{len(as_)} 段 AI 文本拟合；"
                   "建议再用你自己的文字校准。")
    ev = [evaluate_rows(parts[p], cal, name) for p, name in spec["eval"] if parts.get(p)]
    return {"calibration": cal, "report": res["report"], "evaluation": ev}


def stage_fit(args):
    d = Path(args.scores_dir)
    parts = {}
    for name in PART_PROFILE:
        f = d / f"{name}.json"
        if f.exists():
            parts[name] = json.loads(f.read_text("utf-8"))
    fitted = {}
    for prof in ("zh", "zh_short", "en", "zh_classical", "zh_poetry"):
        r = fit_profile(prof, parts, args.target_fpr)
        if r:
            fitted[prof] = r
        else:
            print(f"::warning title=跳过 {prof}::没有足够的打分结果", flush=True)
    if "zh" not in fitted:
        raise SystemExit("没有现代汉语校准集的打分结果")

    out_cal = dict(fitted["zh"]["calibration"])
    out_cal["profiles"] = {p: dict(r["calibration"], profile=p) for p, r in fitted.items() if p != "zh"}
    (Path(args.out) / "app" / "default_calibration.json").write_text(json.dumps(out_cal, ensure_ascii=False, indent=1), "utf-8")

    report = {p: {"calibration_report": r["report"], "evaluation": r["evaluation"], "models": r["calibration"]["models"]}
              for p, r in fitted.items()}
    (Path(args.out) / "tools" / "eval_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")

    lines = ["# 检测效果评估报告", "",
             "每种文体各自校准：先识别段落是现代汉语、文言还是英文，再用对应的分类器和阈值判断。",
             f"语言模型：{config.OBSERVER_MODEL} + {config.PERFORMER_MODEL}；中文分类器 {config.CLASSIFIER_MODEL}；"
             f"英文分类器 {config.EN_CLASSIFIER_MODEL}。", "",
             "“检出率”= AI 文本被判为 AI 的比例；“误判率”= 人写文本被误判为 AI 的比例；AUROC 1 为完美区分，0.5 为随机。", ""]
    for p, r in fitted.items():
        rep = r["report"]
        lines.append(f"## {scoring.PROFILE_NAMES[p]}")
        lines.append(f"- 数据：{PROFILE_SOURCE[p]}")
        lines.append(f"- 校准集 {rep['n_human']} 人写 / {rep['n_ai']} AI；阈值 {r['calibration']['threshold']}；"
                     f"交叉验证 AUROC {rep.get('combined_auroc')}；特征 {', '.join(rep.get('features_used') or [])}")
        if rep.get("feature_selection"):
            lines.append("- 特征组合比较（校准集交叉验证 AUROC）：" +
                         "，".join(f"{k} {v}" for k, v in rep["feature_selection"].items()) + f" → 选用{rep['feature_set']}")
        for e in r["evaluation"]:
            parts_txt = [f"AUROC {e['auroc']}"]
            if e["ai_caught"] is not None:
                parts_txt.append(f"检出率 {e['ai_caught']:.1%}")
            if e["human_flagged"] is not None:
                parts_txt.append(f"误判率 {e['human_flagged']:.1%}")
            lines.append(f"- **{e['name']}**（{e['n_ai']} AI / {e['n_human']} 人写）：" + "；".join(parts_txt))
            if e["ai_caught_by_model"]:
                lines.append("  - 各来源检出率：" + "，".join(f"{m} {v:.0%}" for m, v in e["ai_caught_by_model"].items()))
            if e["human_flagged_by_source"]:
                lines.append("  - 各来源误判率：" + "，".join(f"{m} {v:.0%}" for m, v in e["human_flagged_by_source"].items()))
            lines.append("  - 各特征 AUROC：" + "，".join(f"{k} {v}" for k, v in e["per_signal_auroc"].items()))
        lines.append("")
    (Path(args.out) / "tools" / "EVAL_REPORT.md").write_text("\n".join(lines), "utf-8")
    print("\n".join(lines))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    for p, r in fitted.items():
        for e in r["evaluation"]:
            print(f"::notice title={scoring.PROFILE_NAMES[p]}：{e['name']}::AUROC {e['auroc']} · "
                  f"检出率 {e['ai_caught']} · 误判率 {e['human_flagged']} · "
                  + " · ".join(f"{m} {v:.0%}" for m, v in list(e["ai_caught_by_model"].items())[:8]), flush=True)
        print(f"::notice title={scoring.PROFILE_NAMES[p]} 校准::" + json.dumps(r["report"], ensure_ascii=False)[:2500], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["score", "fit"], required=True)
    ap.add_argument("--part", choices=list(PART_PROFILE))
    ap.add_argument("--data", help="NLPCC 数据目录（现代汉语）")
    ap.add_argument("--mage-dir", help="MAGE 数据目录（英文）")
    ap.add_argument("--classical-dir", help="NiuTrans Classical-Modern 仓库目录（文言）")
    ap.add_argument("--changan-dir", help="ChangAn 仓库目录（诗词）")
    ap.add_argument("--n-po-cal", type=int, default=800)
    ap.add_argument("--n-po-test", type=int, default=600)
    ap.add_argument("--scores-dir", default="/tmp/eval_scores")
    ap.add_argument("--n-cal", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--n-csl", type=int, default=240)
    ap.add_argument("--n-en-cal", type=int, default=600)
    ap.add_argument("--n-en-test", type=int, default=300)
    ap.add_argument("--n-cl-human", type=int, default=240)
    ap.add_argument("--n-cl-human-test", type=int, default=160)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--out", default=str(ROOT))
    args = ap.parse_args()
    if args.stage == "score":
        if not args.part:
            raise SystemExit("--stage score 需要 --part")
        stage_score(args)
    else:
        stage_fit(args)


if __name__ == "__main__":
    main()
