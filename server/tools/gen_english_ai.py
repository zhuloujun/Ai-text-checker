"""批量生成"国产大模型写的英文学术文字"，并抓取同题材的真人英文摘要，作为训练英文检测模型的数据。

各大检测平台的核心做法：持续收集最新 AI 模型写的同类文字来训练分类器。本仓库现用的英文分类器没见过
DeepSeek / Kimi / 通义千问写的英文论文，所以对它们几乎失灵（见 tools/EVAL_REPORT.md 的"国产新模型"一项）。

做法（与 MAGE、M4 等学术数据集相同）：
  1. 真人：从 arXiv 官方接口按学科抓取 2021 年以前（ChatGPT 出现前）的论文标题与摘要；
  2. AI：把同一批标题交给各家模型，让它写摘要 / 引言 / 结论 / 结果分析等段落（中英文提示词、不同长度与语气轮换）；
  3. 保存到 tools/data/gen_en/，训练时按"标题"划分训练 / 评估，评估用的题目从不参与训练。

需要的密钥（在 GitHub 仓库 Settings → Secrets and variables → Actions 里添加，有哪个用哪个）：
  DEEPSEEK_API_KEY（DeepSeek）、KIMI_API_KEY（Kimi）、WENXIN_API_KEY（文心一言 / 百度千帆）、DASHSCOPE_API_KEY（通义千问）

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
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

# 名称 → (环境变量, [(接口地址, 模型), ...备选])；同一家有多个地址时依次尝试（如 Kimi 国内站 / 国际站）
PROVIDERS = {
    "deepseek": ("DEEPSEEK_API_KEY", [("https://api.deepseek.com/chat/completions", "deepseek-chat")]),
    # 旧的 moonshot-v1-8k 在部分账号已下线（2026-09 实测国内站返回 404 “Not found the model”），依次尝试新模型名
    "kimi": ("KIMI_API_KEY", [(f"https://api.moonshot.{d}/v1/chat/completions", m)
                              for d in ("cn", "ai")
                              for m in ("moonshot-v1-8k", "moonshot-v1-auto", "kimi-k2-turbo-preview",
                                        "kimi-k2-0905-preview", "kimi-latest")]),
    # 千帆 v2：账号没开通的模型会返回 401 invalid_model；ernie-speed / ernie-lite 通常免费默认可用
    "wenxin": ("WENXIN_API_KEY", [("https://qianfan.baidubce.com/v2/chat/completions", m)
                                  for m in ("ernie-4.5-turbo-32k", "ernie-4.5-turbo-128k", "ernie-x1-turbo-32k",
                                            "ernie-4.0-turbo-8k", "ernie-3.5-8k", "ernie-speed-128k",
                                            "ernie-speed-8k", "ernie-lite-8k")]),
    "qwen": ("DASHSCOPE_API_KEY", [("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", "qwen-plus")]),
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


# PubMed（真人，ChatGPT 之前 2012–2021 年的论文摘要）：农业、经济金融、食品、环境、医学、工程、教育等，
# 每个主题一半取"中国作者"（China[Affiliation]）——英文检测最容易冤枉的就是中国作者写的英文论文，必须让模型见过。
PUBMED_TOPICS = [
    "vegetable cultivation yield", "crop yield fertilizer management", "rice cultivation technology", "greenhouse vegetable production",
    "farmer income agricultural economics", "plant disease control field", "soil fertility organic fertilizer",
    "irrigation water use efficiency", "fruit tree cultivation", "tea plantation", "food processing quality",
    "aquaculture fish farming", "livestock poultry production", "stock market returns volatility", "financial risk bank",
    "economic growth regional", "environmental pollution assessment", "climate change agriculture adaptation",
    "traditional Chinese medicine clinical", "nursing intervention patients", "machine learning prediction model",
    "material synthesis characterization", "teaching education students", "public health survey",
]
PM_KINDS = [
    ("full_paper", "请用英文写一篇题为《{t}》的学术论文，包括 Abstract、Keywords、1. Introduction、2. Materials and Methods、"
                   "3. Results and Analysis、4. Discussion、5. Conclusion，正文约 {n2} 词，数据可以合理虚构。只输出论文正文。"),
    ("full_paper_en", "Write a complete research paper titled \"{t}\" with Abstract, Keywords, Introduction, Materials and Methods, "
                      "Results, Discussion and Conclusion sections (about {n2} words in total). Use realistic (invented) data."),
    ("zh_abstract", "请帮我写论文《{t}》的英文摘要，约 {n} 词，符合 SCI 期刊风格，只输出英文摘要。"),
    ("full_paper", None),
    ("zh_author", "你是一位中国高校的研究者，正在给国际期刊投稿。请用英文写论文《{t}》的引言部分，约 {n} 词，只输出英文正文。"),
    ("results", "Write the \"Results and Discussion\" section of a paper titled \"{t}\", with concrete (invented) numbers, about {n} words."),
]


def _pubmed(url_tail, params):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/" + url_tail + "?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                data = r.read()
            time.sleep(0.4)          # NCBI 要求无密钥时每秒不超过 3 次
            return data
        except Exception as e:  # noqa: BLE001
            print(f"PubMed 第 {attempt + 1} 次失败：{e}", flush=True)
            time.sleep(5 * (attempt + 1))
    return None


def fetch_pubmed(topic: str, n: int, china: bool) -> list[dict]:
    term = (f"({topic}) AND 2012:2021[dp] AND hasabstract AND english[lang] AND "
            + ("China[Affiliation]" if china else "NOT China[Affiliation]"))
    raw = _pubmed("esearch.fcgi", {"db": "pubmed", "term": term, "retmax": n, "retmode": "json", "sort": "relevance"})
    if not raw:
        return []
    ids = json.loads(raw).get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    xml = _pubmed("efetch.fcgi", {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"})
    if not xml:
        return []
    out = []
    for art in ET.fromstring(xml).iter("PubmedArticle"):
        title = re.sub(r"\s+", " ", "".join(art.find(".//ArticleTitle").itertext()) if art.find(".//ArticleTitle") is not None else "").strip()
        parts = ["".join(a.itertext()).strip() for a in art.findall(".//Abstract/AbstractText")]
        abst = re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip()
        if title and len(abst) > 600 and not re.search(r"[\u4e00-\u9fff]", abst):
            out.append({"title": title.rstrip("."), "text": abst, "topic": topic, "cn": china,
                        "id": art.findtext(".//PMID", "")})
    return out


def fetch_pmc(topic: str, n: int, china: bool) -> list[dict]:
    """PMC 开放获取全文（真人论文正文：引言、材料与方法、结果、讨论）。只有摘要的话，模型可能把"方法 / 结果段落的写法"
    误当成 AI 特征（AI 样本里有整篇论文），所以真人一侧也要有正文。"""
    term = (f"({topic}) AND 2012:2021[pdat] AND open access[filter] AND "
            + ("China[Affiliation]" if china else "NOT China[Affiliation]"))
    raw = _pubmed("esearch.fcgi", {"db": "pmc", "term": term, "retmax": n, "retmode": "json", "sort": "relevance"})
    if not raw:
        return []
    ids = json.loads(raw).get("esearchresult", {}).get("idlist", [])
    out = []
    for i in range(0, len(ids), 10):
        xml = _pubmed("efetch.fcgi", {"db": "pmc", "id": ",".join(ids[i:i + 10]), "retmode": "xml"})
        if not xml:
            continue
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            continue
        for art in root.iter("article"):
            t = art.find(".//article-meta//article-title")
            title = re.sub(r"\s+", " ", "".join(t.itertext())).strip() if t is not None else ""
            body = art.find("body")
            if not title or body is None:
                continue
            paras = []
            for sec in body.iter("sec"):
                for p_ in sec.findall("p"):
                    for bad in p_.findall(".//xref") + p_.findall(".//table-wrap") + p_.findall(".//fig"):
                        bad.text = ""
                    txt = re.sub(r"\s+", " ", "".join(p_.itertext())).strip()
                    txt = re.sub(r"\[\s*[,–-]*\s*\]|\(\s*[,;–-]*\s*\)", "", txt)
                    if len(txt) > 200:
                        paras.append(txt)
            text = "\n".join(paras)
            if len(text) > 2000 and not re.search(r"[\u4e00-\u9fff]", text):
                out.append({"title": title.rstrip("."), "text": text[:9000], "topic": topic, "cn": china})
    return out


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


# 每家最少请求间隔（秒）：Kimi 未充值账号限速每分钟 3 次（2026-09 实测 HTTP 429 “max RPM: 3”）
MIN_INTERVAL = {"api.moonshot": 21.0}
_last_call = {}


def _throttle(url):
    for k, gap in MIN_INTERVAL.items():
        if k in url:
            wait = _last_call.get(k, 0) + gap - time.time()
            if wait > 0:
                time.sleep(wait)
            _last_call[k] = time.time()


def chat_once(url, key, model, prompt, temperature, max_tokens=1200):
    _throttle(url)
    new_kimi = model.startswith("kimi-k")          # Kimi 新模型（k2.6 / k3 等）只接受 temperature = 1，且会先"思考"，要留足 token
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 1.0 if new_kimi else min(temperature, 1.0),
                       "max_tokens": max_tokens + 5000 if new_kimi else max_tokens}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return (json.loads(r.read())["choices"][0]["message"].get("content") or "").strip()


def _get_json(url, key):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def discover_models(name, key, endpoints):
    """问接口"这个密钥能用哪些模型"（OpenAI 兼容的 /models），把能用的聊天模型排到最前面；顺便打印余额（不打印密钥）。"""
    extra = []
    for base in dict.fromkeys(u.rsplit("/chat/completions", 1)[0] for u, _ in endpoints):
        try:
            ids = [m.get("id") for m in _get_json(base + "/models", key).get("data", []) if m.get("id")]
            print(f"::notice title={name} 可用模型（{base}）::{', '.join(ids[:40]) or '（空）'}", flush=True)
            chat_ids = [i for i in ids if not re.search(r"embed|vision|tts|audio|image|rerank", i, re.I)]
            chat_ids.sort(key=lambda i: ("code" in i, ("8k" not in i and "turbo" not in i), i))
            extra += [(base + "/chat/completions", i) for i in chat_ids]
        except urllib.error.HTTPError as e:
            print(f"::warning title={name} 查询模型列表失败::{base} → HTTP {e.code} {e.read()[:200]!r}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"::warning title={name} 查询模型列表失败::{base} → {e}", flush=True)
        if "moonshot" in base:
            try:
                print(f"::notice title={name} 账户余额（{base}）::{json.dumps(_get_json(base + '/users/me/balance', key).get('data'), ensure_ascii=False)}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"::warning title={name} 查询余额失败::{base} → {e}", flush=True)
    seen, out = set(), []
    for ep in extra + list(endpoints):
        if ep not in seen:
            seen.add(ep); out.append(ep)
    return out


def pick_endpoint(name, key, endpoints):
    """先用一句话试一下，找到这个密钥能用的接口；都不行就返回 None（并打印原因，不打印密钥）。"""
    if name == "kimi":
        endpoints = discover_models(name, key, endpoints)
    for url, model in endpoints[:12]:
        try:
            try:
                chat_once(url, key, model, "Reply with the single word OK.", 0.1)
            except urllib.error.HTTPError as e:
                if e.code != 429:
                    raise
                time.sleep(30)          # 被限速：等一会儿再试一次
                chat_once(url, key, model, "Reply with the single word OK.", 0.1)
            print(f"{name}: 使用 {url} · {model}", flush=True)
            return url, model
        except urllib.error.HTTPError as e:
            print(f"::warning title={name} 接口不可用::{url} {model} → HTTP {e.code} {e.read()[:200]!r}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"::warning title={name} 接口不可用::{url} {model} → {e}", flush=True)
    return None


def chat(url, key, model, prompt, temperature, max_tokens=1200):
    for attempt in range(4):
        try:
            return chat_once(url, key, model, prompt, temperature, max_tokens)
        except Exception as e:  # noqa: BLE001
            print(f"  {model} 第 {attempt + 1} 次失败：{e}", flush=True)
            time.sleep(30 if "429" in str(e) else 10 * (attempt + 1))
    return None


def clean(t: str) -> str:
    t = re.sub(r"^\s*(\*\*)?(abstract|introduction|conclusion|results( and analysis)?)(\*\*)?\s*[:：]?\s*\n", "", t, flags=re.I)
    t = t.replace("**", "").replace("#", "")
    return t.strip()


def load_jsonl(f: Path) -> list[dict]:
    return [json.loads(l) for l in f.read_text("utf-8").splitlines() if l.strip()] if f.exists() else []


def human_pool(out: Path, source: str, per_category: int) -> list[dict]:
    if source == "pmc":
        f = out / "pmc_human.jsonl"
        human = load_jsonl(f)
        if not human:
            seen = set()
            for topic in PUBMED_TOPICS:
                for china, n in ((True, 25), (False, 15)):
                    got = [h for h in fetch_pmc(topic, n, china) if h["title"] not in seen]
                    seen.update(h["title"] for h in got)
                    print(f"PMC {topic}（{'中国作者' if china else '其他'}）：{len(got)} 篇", flush=True)
                    human += got
            f.write_text("\n".join(json.dumps(h, ensure_ascii=False) for h in human) + "\n", "utf-8")
            print(f"::notice title=PMC 真人论文正文::共 {len(human)} 篇（中国作者 {sum(h['cn'] for h in human)} 篇）", flush=True)
        return human
    if source == "pubmed":
        f = out / "pubmed_human.jsonl"
        human = load_jsonl(f)
        if not human:
            seen = set()
            for topic in PUBMED_TOPICS:
                for china, n in ((True, 50), (False, 40)):
                    got = [h for h in fetch_pubmed(topic, n, china) if h["title"] not in seen]
                    seen.update(h["title"] for h in got)
                    print(f"PubMed {topic}（{'中国作者' if china else '其他'}）：{len(got)} 篇", flush=True)
                    human += got
            f.write_text("\n".join(json.dumps(h, ensure_ascii=False) for h in human) + "\n", "utf-8")
            print(f"::notice title=PubMed 真人摘要::共 {len(human)} 篇（中国作者 {sum(h['cn'] for h in human)} 篇）", flush=True)
        return human
    f = out / "arxiv_human.jsonl"
    human = load_jsonl(f)
    if not human:
        for cat in CATEGORIES:
            got = fetch_arxiv(cat, per_category)
            print(f"arXiv {cat}: {len(got)} 篇", flush=True)
            human += got
        f.write_text("\n".join(json.dumps(h, ensure_ascii=False) for h in human) + "\n", "utf-8")
    return human


# "先用中文写、再让 AI 译成英文"：国内作者最常见的用法，译出来的英文带明显的中式表达（per mu、"see dry see wet"等），
# 读起来像中国作者自己写的英文，分类器最容易漏判。两步都由同一家模型完成。
TR_KINDS = [
    ("zh_then_en", "请写一篇题为《{t}》的中文学术论文，包括摘要、关键词、1 引言、2 材料与方法、3 结果与分析、4 讨论、5 结论，"
                   "约 {z} 字，数据可以合理虚构，适合中国农业 / 经济 / 医学类期刊。只输出论文正文。"),
    ("zh_then_en_short", "请写一篇题为《{t}》的中文科技论文，包括摘要、引言、关键技术分析（分 3–5 个小节）、结果与效益分析、结论，"
                         "约 {z} 字，语言朴实，适合技术推广类期刊。只输出论文正文。"),
]
TRANSLATE = ["请把下面这篇中文论文完整翻译成英文，保持原有结构和小标题编号，只输出英文译文：\n\n{zh}",
             "Translate the following Chinese paper into English for submission to a journal. Keep all headings and numbers. "
             "Output only the English translation.\n\n{zh}"]


def run_translate(name, key, endpoints, titles, out_f, n_titles, t_start, budget_min, lock):
    ep = pick_endpoint(name, key, endpoints)
    if not ep:
        return 0
    url, model = ep
    rnd = random.Random(f"{name}-{out_f.name}")
    done = {r["title"] for r in load_jsonl(out_f)}
    n_new = 0
    for i, t in enumerate(rnd.sample(titles, min(n_titles, len(titles)))):
        if (time.time() - t_start) / 60 > budget_min:
            print(f"::warning title={name}::已到时长上限，先保存已生成的 {n_new} 篇", flush=True)
            break
        if t in done:
            continue
        kind, tpl = TR_KINDS[i % len(TR_KINDS)]
        zh = chat(url, key, model, tpl.format(t=t, z=rnd.choice([1500, 2000, 2500])), rnd.choice([0.7, 0.9]), 3500)
        if not zh or len(zh) < 500:
            continue
        en = chat(url, key, model, rnd.choice(TRANSLATE).format(zh=clean(zh)), 0.3, 4000)
        if en and len(en) > 800 and not re.search(r"[\u4e00-\u9fff]{20}", en):
            with lock, out_f.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"title": t, "kind": kind, "model": name, "text": clean(en)}, ensure_ascii=False) + "\n")
            n_new += 1
        if i % 10 == 0:
            print(f"{name}: {i + 1}", flush=True)
    print(f"::notice title={name}（{out_f.name}）::新生成 {n_new} 篇", flush=True)
    return n_new


def run_provider(name, key, endpoints, titles, out_f, kinds, n_titles, t_start, budget_min, lock):
    ep = pick_endpoint(name, key, endpoints)
    if not ep:
        return 0
    url, model = ep
    rnd = random.Random(f"{name}-{out_f.name}")
    done = {r["title"] + "|" + r["kind"] for r in load_jsonl(out_f)}
    picks = rnd.sample(titles, min(n_titles, len(titles)))
    n_new = 0
    for i, t in enumerate(picks):
        if (time.time() - t_start) / 60 > budget_min:
            print(f"::warning title={name}::已到时长上限，先保存已生成的 {n_new} 篇（下次运行会接着生成）", flush=True)
            break
        kind, tpl = kinds[i % len(kinds)]
        tpl = tpl or next(x for k, x in kinds if k == kind and x)
        if f"{t}|{kind}" in done:
            continue
        n, n2 = rnd.choice([150, 200, 250, 300]), rnd.choice([900, 1200, 1500])
        long = kind.startswith("full_paper")
        text = chat(url, key, model, tpl.format(t=t, n=n, n2=n2), rnd.choice([0.6, 0.8, 1.0]), 3500 if long else 1200)
        if text and len(text) > 300:
            with lock, out_f.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"title": t, "kind": kind, "model": name, "text": clean(text)}, ensure_ascii=False) + "\n")
            n_new += 1
        if i % 20 == 0:
            print(f"{name}: {i + 1}/{len(picks)}", flush=True)
    print(f"::notice title={name}（{out_f.name}）::新生成 {n_new} 篇", flush=True)
    return n_new


def main():
    import threading
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data" / "gen_en"))
    ap.add_argument("--source", choices=["arxiv", "pubmed", "pmc", "translate"], default="arxiv",
                    help="题目来源：arxiv（摘要类）或 pubmed（农业、经济、医学等，含中国作者；生成整篇论文）")
    ap.add_argument("--n-titles", type=int, default=200, help="每家模型生成多少篇（每篇轮换一种类型）")
    ap.add_argument("--per-category", type=int, default=40)
    ap.add_argument("--time-budget-min", type=float, default=260,
                    help="总时长上限（分钟）：到点就停止生成、保留已生成的部分，避免工作流超时被强行终止而丢掉全部数据")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    human = human_pool(out, "pubmed" if args.source == "translate" else args.source, args.per_category)
    if not human:
        raise SystemExit("没有抓到真人摘要")
    if args.source == "pmc":
        return          # 只抓真人论文正文，不生成
    titles = [h["title"] for h in human]
    kinds = PM_KINDS if args.source == "pubmed" else KINDS
    t_start = time.time()
    lock = threading.Lock()
    threads = []
    for name, (env, endpoints) in PROVIDERS.items():
        key = os.getenv(env, "").strip()
        if not key:
            print(f"未设置 {env}，跳过 {name}", flush=True)
            continue
        out_f = out / {"pubmed": f"pm_{name}.jsonl", "translate": f"tr_{name}.jsonl"}.get(args.source, f"{name}.jsonl")
        # 各家并行生成（Kimi 限速且会先"思考"，很慢，不能让它拖住其他家）
        if args.source == "translate":
            th = threading.Thread(target=run_translate, args=(name, key, endpoints, titles, out_f, args.n_titles,
                                                              t_start, args.time_budget_min, lock), daemon=True)
        else:
            th = threading.Thread(target=run_provider, args=(name, key, endpoints, titles, out_f, kinds, args.n_titles,
                                                             t_start, args.time_budget_min, lock), daemon=True)
        th.start(); threads.append(th)
    for th in threads:
        th.join()
    if not threads:
        print("::warning title=没有可用的 API 密钥::请检查仓库 Secrets 里的 DEEPSEEK_API_KEY / KIMI_API_KEY / WENXIN_API_KEY 是否正确、账户是否有余额", flush=True)


if __name__ == "__main__":
    main()
