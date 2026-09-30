"""批量生成"国产大模型写的英文学术文字"，并抓取同题材的真人英文摘要，作为训练英文检测模型的数据。

各大检测平台的核心做法：持续收集最新 AI 模型写的同类文字来训练分类器。本仓库现用的英文分类器没见过
DeepSeek / Kimi / 通义千问写的英文论文，所以对它们几乎失灵（见 tools/EVAL_REPORT.md 的"国产新模型"一项）。

做法（与 MAGE、M4 等学术数据集相同）：
  1. 真人：从 arXiv 官方接口按学科抓取 2021 年以前（ChatGPT 出现前）的论文标题与摘要；
  2. AI：把同一批标题交给各家模型，让它写摘要 / 引言 / 结论 / 结果分析等段落（中英文提示词、不同长度与语气轮换）；
  3. 保存到 tools/data/gen_en/，训练时按"标题"划分训练 / 评估，评估用的题目从不参与训练。

需要的密钥（在 GitHub 仓库 Settings → Secrets and variables → Actions 里添加，有哪个用哪个）：
  DEEPSEEK_API_KEY（DeepSeek）、MOONSHOT_API_KEY（Kimi）、DASHSCOPE_API_KEY（通义千问）

用法：python tools/gen_english_ai.py --out tools/data/gen_en --n-titles 200
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

PROVIDERS = {
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com/chat/completions", "deepseek-chat"),
    "kimi": ("MOONSHOT_API_KEY", "https://api.moonshot.cn/v1/chat/completions", "moonshot-v1-8k"),
    "qwen": ("DASHSCOPE_API_KEY", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", "qwen-plus"),
}
# 学科尽量宽：经济金融、农业与生物、医学、工程、计算机、社会科学、物理数学
CATEGORIES = ["q-fin.GN", "q-fin.ST", "econ.GN", "q-bio.PE", "q-bio.QM", "physics.soc-ph", "cs.CY", "cs.LG",
              "eess.SY", "stat.AP", "physics.ao-ph", "q-bio.NC", "cs.HC", "math.OC"]
KINDS = [
    ("abstract", "Write the abstract of an academic paper titled \"{t}\". About {n} words. Output only the abstract text."),
    ("introduction", "Write the introduction section of a research paper titled \"{t}\" (about {n} words, 2–3 paragraphs). Output only the text, no headings."),
    ("results", "Write the \"Results and Analysis\" section of a paper titled \"{t}\", with concrete (invented) numbers, about {n} words. Output only the text."),
    ("conclusion", "Write the conclusion section of a paper titled \"{t}\", about {n} words. Output only the text."),
    ("zh_prompt", "请用英文为题为“{t}”的学术论文写一段约 {n} 词的摘要，只输出英文正文。"),
    ("zh_author", "你是一位中国高校的研究者，正在给国际期刊投稿。请用英文写论文《{t}》的引言部分，约 {n} 词，只输出英文正文。"),
]


def fetch_arxiv(cat: str, n: int) -> list[dict]:
    q = urllib.parse.urlencode({"search_query": f"cat:{cat} AND submittedDate:[201501010000 TO 202112312359]",
                                "start": 0, "max_results": n, "sortBy": "relevance"})
    url = f"http://export.arxiv.org/api/query?{q}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                root = ET.fromstring(r.read())
            break
        except Exception as e:  # noqa: BLE001
            print(f"arXiv {cat} 第 {attempt + 1} 次失败：{e}", flush=True)
            time.sleep(5 * (attempt + 1))
    else:
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in root.findall("a:entry", ns):
        title = re.sub(r"\s+", " ", e.findtext("a:title", "", ns)).strip()
        abst = re.sub(r"\s+", " ", e.findtext("a:summary", "", ns)).strip()
        if title and len(abst) > 400 and "$" not in abst:
            out.append({"title": title, "text": abst, "category": cat,
                        "id": e.findtext("a:id", "", ns)})
    time.sleep(3.5)          # arXiv 接口要求两次请求间隔 3 秒以上
    return out


def chat(url, key, model, prompt, temperature):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": temperature, "max_tokens": 1200}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001
            print(f"  {model} 第 {attempt + 1} 次失败：{e}", flush=True)
            time.sleep(10 * (attempt + 1))
    return None


def clean(t: str) -> str:
    t = re.sub(r"^\s*(\*\*)?(abstract|introduction|conclusion|results( and analysis)?)(\*\*)?\s*[:：]?\s*\n", "", t, flags=re.I)
    t = t.replace("**", "").replace("#", "")
    return t.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data" / "gen_en"))
    ap.add_argument("--n-titles", type=int, default=200, help="每家模型生成多少篇（每篇随机一种段落类型）")
    ap.add_argument("--per-category", type=int, default=40)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(2026)

    human_f = out / "arxiv_human.jsonl"
    if human_f.exists():
        human = [json.loads(l) for l in human_f.read_text("utf-8").splitlines() if l.strip()]
    else:
        human = []
        for cat in CATEGORIES:
            got = fetch_arxiv(cat, args.per_category)
            print(f"arXiv {cat}: {len(got)} 篇", flush=True)
            human += got
        human_f.write_text("\n".join(json.dumps(h, ensure_ascii=False) for h in human) + "\n", "utf-8")
    if not human:
        raise SystemExit("没有抓到 arXiv 摘要")
    titles = [h["title"] for h in human]

    any_key = False
    for name, (env, url, model) in PROVIDERS.items():
        key = os.getenv(env, "").strip()
        if not key:
            print(f"未设置 {env}，跳过 {name}", flush=True)
            continue
        any_key = True
        f = out / f"{name}.jsonl"
        done = {json.loads(l)["title"] + "|" + json.loads(l)["kind"] for l in f.read_text("utf-8").splitlines()} if f.exists() else set()
        picks = rnd.sample(titles, min(args.n_titles, len(titles)))
        n_new = 0
        with f.open("a", encoding="utf-8") as fh:
            for i, t in enumerate(picks):
                kind, tpl = KINDS[i % len(KINDS)]
                if f"{t}|{kind}" in done:
                    continue
                n = rnd.choice([150, 200, 250, 300])
                text = chat(url, key, model, tpl.format(t=t, n=n), rnd.choice([0.7, 1.0, 1.2]))
                if text and len(text) > 300:
                    fh.write(json.dumps({"title": t, "kind": kind, "model": name, "text": clean(text)}, ensure_ascii=False) + "\n")
                    fh.flush()
                    n_new += 1
                if i % 20 == 0:
                    print(f"{name}: {i + 1}/{len(picks)}", flush=True)
        print(f"::notice title={name}::新生成 {n_new} 篇", flush=True)
    if not any_key:
        print("::warning title=没有可用的 API 密钥::请在仓库 Secrets 里添加 DEEPSEEK_API_KEY / MOONSHOT_API_KEY / DASHSCOPE_API_KEY 之一", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
