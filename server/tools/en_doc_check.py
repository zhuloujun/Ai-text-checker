"""给"英文非论文整篇判断"定阈值：用指定的英文分类器（Release 里的 english-classifier-vN）给真人英文长文
（Reddit 写作社区真人故事、CNN 新闻、古腾堡电子书片段、IMDB 长影评）和用户确认来源的文档逐段打分，
按网站同一套分段规则、每篇取各段得分按字数加权的中位数（两段取较低的一段），报告真人文档的最高值与分位数、
AI 文档的分布，写到 tools/data/en_doc_check.json。

用法（工作流 en-doc-check.yml 调用）：python tools/en_doc_check.py --model /tmp/x/english-classifier --n 150
"""
import argparse
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
DATA = Path(__file__).resolve().parent / "data"
SOURCES = [  # (数据集, 配置, 字段, 名称)
    ("euclaise/writingprompts", None, "story", "reddit-stories"),
    ("abisee/cnn_dailymail", "3.0.0", "article", "cnn-news"),
    ("deepmind/pg19", None, "text", "gutenberg"),
    ("stanfordnlp/imdb", None, "text", "imdb"),
]


def _get(url):
    with urllib.request.urlopen(url, timeout=90) as r:
        return json.loads(r.read())


def rows(dataset, config, field, want, rnd, min_words=450):
    q = urllib.parse.quote(dataset, safe="")
    sp = _get(f"https://datasets-server.huggingface.co/splits?dataset={q}")["splits"]
    if config:
        sp = [s for s in sp if s["config"] == config] or sp
    sp = [s for s in sp if s["split"] in ("test", "validation")] or sp      # 尽量用测试集，避免与训练数据重叠
    cfg, split = sp[0]["config"], sp[0]["split"]
    out = []
    for off in rnd.sample(range(0, 2000, 20), 60):
        try:
            rs = _get(f"https://datasets-server.huggingface.co/rows?dataset={q}&config={urllib.parse.quote(cfg)}"
                      f"&split={split}&offset={off}&length=20")["rows"]
        except Exception as e:  # noqa: BLE001
            print(f"  {dataset} {off}: {e}", flush=True)
            time.sleep(2)
            continue
        for r in rs:
            t = r["row"].get(field) or ""
            if not isinstance(t, str):
                continue
            t = t.replace("<newline>", "\n").replace("<br />", "\n")
            if dataset.endswith("pg19"):          # 电子书：跳过开头的版权页，取中间约 1200 词
                w = t.split()
                if len(w) < 6000:
                    continue
                st = rnd.randint(2000, len(w) - 2500)
                t = " ".join(w[st:st + 1200])
                t = re.sub(r"(?<=[.!?]) ", "\n", t, count=6)
            if len(t.split()) >= min_words:
                out.append(t.strip())
        if len(out) >= want:
            break
    return out[:want]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    from app.segmenter import normalize_english, segment_text
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).float().eval()
    torch.set_num_threads(4)

    def score(texts):
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), 16):
                enc = tok(texts[i:i + 16], truncation=True, max_length=320, padding=True, return_tensors="pt")
                out += torch.softmax(model(**enc).logits.float(), -1)[:, 1].tolist()
        return out

    def doc_median(text):
        segs = [s for s in segment_text(text) if s.counted and s.register == "en"]
        if len(segs) < 2:
            return None, []
        ps = score([normalize_english(" ".join(s.text.split())) for s in segs])
        vals = sorted(zip(ps, [len(s.text) for s in segs]))
        if len(vals) == 2:
            return min(ps), ps
        half, acc = sum(n for _, n in vals) / 2, 0
        for p, n in vals:
            acc += n
            if acc >= half:
                return p, ps
        return vals[-1][0], ps

    rnd = random.Random(3)
    res = {"human": {}, "user": {}}
    for ds, cfg, field, name in SOURCES:
        try:
            docs = rows(ds, cfg, field, a.n, rnd)
        except Exception as e:  # noqa: BLE001
            print(f"::warning title={name} 下载失败::{e}", flush=True)
            continue
        meds = [m for m, _ in (doc_median(t) for t in docs) if m is not None]
        res["human"][name] = sorted(round(m, 4) for m in meds)
        print(f"::notice title=真人英文 {name}::{len(meds)} 篇；整篇中位数最高 {max(meds) if meds else None:.3f}，"
              f"≥0.5 的 {sum(m >= 0.5 for m in meds)} 篇，≥0.8 的 {sum(m >= 0.8 for m in meds)} 篇，≥0.9 的 {sum(m >= 0.9 for m in meds)} 篇", flush=True)
    by = {}
    for l in (DATA / "eval_en_user_docs.jsonl").read_text("utf-8").split("\n"):
        if l.strip():
            r = json.loads(l)
            by.setdefault((r["doc"], r["y"]), []).append(r["text"])
    for (doc, y), paras in by.items():
        m, ps = doc_median("\n\n".join(paras))
        res["user"][f"{doc}（{'AI' if y else '真人'}）"] = {"median": None if m is None else round(m, 3), "segs": [round(p, 2) for p in ps]}
    allh = [m for v in res["human"].values() for m in v]
    res["human_max"] = max(allh) if allh else None
    print("::notice title=真人英文整篇最高::" + json.dumps({"max": res["human_max"], "n": len(allh),
          "p99": sorted(allh)[int(len(allh) * 0.99) - 1] if allh else None}, ensure_ascii=False), flush=True)
    print("::notice title=用户文档整篇::" + json.dumps({k: v["median"] for k, v in res["user"].items()}, ensure_ascii=False)[:3800], flush=True)
    (DATA / "en_doc_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=0), "utf-8")


if __name__ == "__main__":
    main()
