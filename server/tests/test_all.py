"""测试。运行：TEST_MODELS_DIR=<含 observer/performer/cls 三个小模型的目录> pytest -q
（小模型可用 tests/make_tiny_models.py 生成；它们是随机权重，只用于检查流程与公式，不代表检测效果。）"""
import os
import sys
import time
from pathlib import Path

import pytest

MD = os.environ.get("TEST_MODELS_DIR")
if not MD:
    pytest.skip("未设置 TEST_MODELS_DIR", allow_module_level=True)

os.environ.update({
    "OBSERVER_MODEL": f"{MD}/observer", "PERFORMER_MODEL": f"{MD}/performer", "CLASSIFIER_MODEL": f"{MD}/cls",
    "EN_CLASSIFIER_MODEL": f"{MD}/desklib_en",
    "ADMIN_TOKEN": "test-admin-pw", "LM_MAX_TOKENS": "128", "CALIBRATION_FILE": "/nonexistent/cal.json",
    "MAX_TEXT_CHARS": "300000",
})
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import keys, scoring  # noqa: E402
from app.segmenter import classical_ratio, detect_register, segment_text  # noqa: E402

MODERN = ("宋代的地方行政制度在很大程度上延续了唐末五代的格局，但又有所调整。据《宋史·职官志》记载，路一级机构的设置经历了反复变化，"
          "转运使司的职能也随之扩展。这一时期的文献对州县官员的任免多有记述，然而其中不乏相互矛盾之处。笔者以为，此类记载须与地方志、碑刻互相参证，方能厘清其真实面貌。")
CLASSICAL = ("太祖既受禅，患藩镇之强，乃用赵普之谋，稍夺其权。或曰：其所以然者，盖鉴于唐末五代之乱也。帝曰：善。于是诸镇皆罢，"
             "以文臣知州事，而兵权悉归于上。君子曰：此所谓强干弱枝者也，其虑深矣，岂不然哉。")
ENGLISH = ("The administrative system of the Song dynasty largely continued the structure inherited from the late Tang and "
           "the Five Dynasties, although it was adjusted in several important ways. According to the treatise on offices in "
           "the Song History, the organization of circuit-level agencies changed repeatedly, and the functions of the fiscal "
           "commissioners expanded accordingly. Sources from this period record many appointments of prefectural officials. ")


# ---------------- 公式：与官方实现逐字比对 ----------------

def ref_fast_detect(logits_ref, logits_score, labels):
    """baoguangsheng/fast-detect-gpt: get_sampling_discrepancy_analytic"""
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    return discrepancy.mean().item()


def ref_binoculars(obs_logits, perf_logits, labels):
    """ahans30/Binoculars: ppl = perplexity(encodings, performer_logits); x_ppl = entropy(observer, performer)"""
    ce = torch.nn.functional.cross_entropy(perf_logits, labels, reduction="none")
    ppl = ce.mean().item()
    p_proba = torch.softmax(obs_logits, dim=-1)
    q_scores = perf_logits
    ce2 = (p_proba * -torch.log_softmax(q_scores, dim=-1)).sum(-1)
    x_ppl = ce2.mean().item()
    return ppl / x_ppl


@pytest.fixture(scope="module")
def scorer():
    from app.detectors.lm_scorer import LMScorer
    s = LMScorer()
    s.load()
    return s


def test_formulas_match_official(scorer):
    ids = scorer.tok(MODERN, add_special_tokens=False)["input_ids"][:100]
    inp = torch.tensor([ids])
    with torch.inference_mode():
        o = scorer.observer(input_ids=inp).logits[0, :-1, : scorer.vocab].float()
        p = scorer.performer(input_ids=inp).logits[0, :-1, : scorer.vocab].float()
    labels = inp[0, 1:]
    r = scorer.score_ids(ids)
    assert r["tokens"] == len(ids) - 1
    assert r["fastdetect"] == pytest.approx(ref_fast_detect(o[None], p[None], labels[None]), rel=1e-4)
    assert r["binoculars"] == pytest.approx(ref_binoculars(o, p, labels), rel=1e-4)


def test_extra_features_match_reference(scorer):
    """Log-Rank / LRR（DetectLLM）/ 熵 / GLTR top-k 与独立实现比对。"""
    ids = scorer.tok(MODERN, add_special_tokens=False)["input_ids"][:100]
    inp = torch.tensor([ids])
    with torch.inference_mode():
        p = scorer.performer(input_ids=inp).logits[0, :-1, : scorer.vocab].float()
    labels = inp[0, 1:]
    lp = torch.log_softmax(p, -1)
    ll = lp[torch.arange(len(labels)), labels]
    # 名次：按概率从高到低排序后，实际 token 的位置（从 1 开始）
    order = torch.argsort(lp, dim=-1, descending=True)
    ranks = (order == labels.unsqueeze(-1)).nonzero()[:, 1] + 1
    ref_logrank = torch.log(ranks.float()).mean().item()
    ref_lrr = (-ll.sum() / torch.log(ranks.float()).sum()).item()
    ref_ent = (-(lp.exp() * lp).sum(-1)).mean().item()
    r = scorer.score_ids(ids)
    assert r["log_rank"] == pytest.approx(ref_logrank, rel=1e-4, abs=1e-6)
    assert r["lrr"] == pytest.approx(ref_lrr, rel=1e-4)
    assert r["entropy"] == pytest.approx(ref_ent, rel=1e-4)
    assert r["top1"] == pytest.approx((ranks == 1).float().mean().item())
    assert r["top10"] == pytest.approx((ranks <= 10).float().mean().item())
    assert r["fastdetect_norm"] == pytest.approx(r["fastdetect"] / (len(ids) - 1) ** 0.5)
    assert r["lp_burstiness"] is not None and r["lp_burstiness"] >= 0


def test_smoothing_and_levels():
    probs = [0.1, 0.95, 0.1, None, 0.9, 0.92]
    sm = scoring.smooth(probs, 0.3)
    assert sm[3] is None
    assert sm[1] < 0.95 and sm[0] > 0.1          # 孤立高分被拉低，邻居被略微拉高
    assert scoring.smooth(probs, 0) == probs
    assert scoring.level_of(0.85, 0.5) == ("high", "高度疑似")
    assert scoring.level_of(0.7, 0.5) == ("mid", "中度疑似")
    assert scoring.level_of(0.55, 0.5) == ("light", "轻度疑似")
    assert scoring.level_of(0.45, 0.5)[0] == "low"
    assert scoring.level_of(0.7, 0.75)[0] == "low"
    assert scoring.level_of(0.78, 0.75) == ("mid", "中度疑似")


def test_long_text_is_chunked(scorer):
    r = scorer.score(MODERN * 6)
    assert r and r["tokens"] > 128


# ---------------- 分段 ----------------

def test_classical_ratio_separates():
    assert classical_ratio(CLASSICAL) >= 0.03  # 学术白话也常用"其、而、以"，所以还要看现代汉语标志词
    assert detect_register(MODERN) == "zh"
    assert detect_register(CLASSICAL) == "zh_classical"
    assert detect_register(ENGLISH) == "en"
    assert detect_register(POEM) == "zh_poetry" and detect_register(COUPLET) == "zh_poetry"


def test_classical_document_is_counted_and_modern_quotes_excluded():
    # 通篇文言（文言小说、仿古文）：文言就是正文，对话多也不算引文
    story = "\n\n".join([CLASSICAL, "女曰：“妾本唐时花媪，以杜工部一诗，得窃灵气，岁久成精。郎宜自爱，勿以妾为念也。”生泣而别之。", CLASSICAL])
    segs = segment_text(story)
    assert all(s.kind == "body" and s.register == "zh_classical" for s in segs)
    # 现代汉语论文里夹一段文言：文言段落视为古籍引文，另起一段
    paper = "\n\n".join([MODERN, CLASSICAL, MODERN])
    kinds = [(s.register, s.kind) for s in segment_text(paper)]
    assert ("zh_classical", "quotation") in kinds and kinds.count(("zh", "body")) == 2


POEM = "诗·七律《咏春》\n浣花溪畔废园春，牡丹幻作红衫人。\n杜老诗魂传一脉，陈生痴念结三生。\n花馔夜饮情方炽，道士符飞梦已尘。\n青城别后重相见，溪上呼名泪满巾。"
COUPLET = "对联：\n飞檐斗拱，几回苍烟落照；\n暮鼓晨钟，一枕孤馆秋寒。"
ESSAY_PARAS = ["读完这篇故事，心中久久不能平静。窗外的风掠过枝头，花影摇曳，恍惚间仿佛也看见一位女子立于残垣之间。",
               "这是一个关于情的故事，但它的动人之处，恰恰在于那份情的不可能。她的存在本身，便是诗与花的因缘和合。",
               "而陈生呢？他明知她是异类，却始终无法割舍。这份情，早已超越了色相之惑，成了一种近乎执拗的守护。",
               "就像每年春天，溪畔的花，依旧会开。"]


def test_titles_split_works_and_short_paragraphs_merge():
    doc = "\n\n".join([POEM, COUPLET, "花落花开——读后感"] + ESSAY_PARAS)
    segs = segment_text(doc)
    assert [s.register for s in segs[:2]] == ["zh_poetry", "zh_poetry"]
    assert segs[0].title.startswith("诗·七律") and segs[1].title == "对联："
    assert len({s.block for s in segs}) == 3               # 三篇作品
    essay = [s for s in segs if s.register == "zh"]
    assert len(essay) <= 2 and all(len(s.text) >= 80 for s in essay)   # 短段落合并成窗口
    assert "依旧会开" in essay[-1].text                      # 结尾一句并入上一段，不单独成段
    assert all(s.kind == "body" for s in segs)              # 带标题的诗词是独立作品，照常计入


def test_untitled_poem_inside_modern_paper_is_quotation():
    poem = "浣花溪畔废园春，牡丹幻作红衫人。\n杜老诗魂传一脉，陈生痴念结三生。"
    paper = "\n\n".join([MODERN * 2, poem, MODERN * 2])
    kinds = [(s.register, s.kind) for s in segment_text(paper)]
    assert ("zh_poetry", "quotation") in kinds


def test_english_segments_are_longer_and_split_on_sentences():
    segs = segment_text(ENGLISH * 8)
    assert all(s.register == "en" for s in segs)
    assert all(len(s.text) >= 500 for s in segs)
    assert all(s.text.rstrip().endswith(".") for s in segs)


def test_profiles_and_merge():
    base = {"calibrated": True, "threshold": 0.5, "signals": scoring.DEFAULTS["signals"], "note": "zh"}
    en = dict(base, note="en", threshold=0.7, profile="en")
    merged = scoring.merge_profile(base, en, "en")
    assert merged["note"] == "zh" and merged["profiles"]["en"]["threshold"] == 0.7
    assert scoring.profile_for(merged, "en")[0]["threshold"] == 0.7
    assert scoring.profile_for(merged, "zh")[0]["note"] == "zh"
    # 文言没有专门校准：沿用现代汉语参数，但标记为未校准
    assert scoring.profile_for(merged, "zh_classical") == (merged, False)
    # 英文没有专门校准：退回经验值，不用中文参数
    assert scoring.profile_for(base, "en")[1] is False
    # 替换现代汉语部分时保留其他文体
    again = scoring.merge_profile(merged, dict(base, note="zh2"), "zh")
    assert again["note"] == "zh2" and again["profiles"]["en"]["threshold"] == 0.7


def test_segmenter_excludes_references_and_quotes():
    doc = "\n\n".join([MODERN * 2, CLASSICAL * 2, "“" + MODERN + "”", MODERN, "参考文献",
                       "[1] 脱脱等：《宋史》，北京：中华书局，1977年。", "[2] 李焘：《续资治通鉴长编》，北京：中华书局，2004年。"])
    segs = segment_text(doc)
    kinds = [s.kind for s in segs]
    assert "reference" in kinds and "quotation" in kinds and "body" in kinds
    assert all(s.kind == "reference" for s in segs if "中华书局" in s.text)
    assert sum(len(s.text) for s in segs) >= len(doc.replace("\n", "")) * 0.95


# ---------------- Key ----------------

def test_keys_roundtrip():
    k = keys.issue("测试", days=1, daily_chars=100)
    p = keys.verify(k["key"])
    assert p and p["i"] == k["id"] and p["q"] == 100
    assert keys.verify(k["key"][:-2] + ("AA" if not k["key"].endswith("AA") else "BB")) is None
    assert keys.verify("atc-garbage") is None
    ok, _ = keys.consume(p, 60)
    assert ok
    ok, left = keys.consume(p, 60)
    assert not ok and left == 40
    keys.revoke_runtime(k["id"])
    assert keys.verify(k["key"]) is None


def test_expired_key():
    k = keys.issue("x", days=1)
    import json, base64
    p64 = k["key"][4:].split(".")[0]
    payload = json.loads(base64.urlsafe_b64decode(p64 + "=" * (-len(p64) % 4)))
    payload["e"] = int(time.time()) - 10
    np64 = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).rstrip(b"=").decode()
    forged = f"atc-{np64}.{keys._sign(np64)}"   # 用正确签名构造一个已过期的 Key
    assert keys.verify(forged) is None


# ---------------- 校准 ----------------

def test_calibration_extended_features():
    import random
    rnd = random.Random(1)
    def mk(ai):
        return {"fastdetect": rnd.gauss(2.8 if ai else 0.6, 0.8), "binoculars": rnd.gauss(0.82 if ai else 0.98, 0.06),
                "classifier": rnd.uniform(0.4, 0.95) if ai else rnd.uniform(0.05, 0.6),
                "fastdetect_norm": rnd.gauss(0.2 if ai else 0.05, 0.05), "lrr": rnd.gauss(1.6 if ai else 1.2, 0.2),
                "log_rank": rnd.gauss(0.8 if ai else 1.4, 0.3), "entropy": rnd.gauss(2.0 if ai else 2.6, 0.4),
                "top10": rnd.gauss(0.85 if ai else 0.7, 0.05), "lp_burstiness": rnd.gauss(0.15 if ai else 0.3, 0.08),
                "style_cv": rnd.gauss(0.3 if ai else 0.6, 0.15), "style_phrases": rnd.gauss(3 if ai else 1, 1)}
    human = [mk(False) for _ in range(40)]
    ai = [mk(True) for _ in range(40)]
    res = scoring.calibrate(human, ai, 0.05)
    rep, cal = res["report"], res["calibration"]
    assert rep["cross_validated"] is True
    assert set(scoring.EXTENDED_FEATURES) == set(cal["lr"]["features"])
    assert rep["combined_auroc"] > 0.95 and rep["human_flagged_rate"] <= 0.05 + 1e-9
    assert "lrr" in rep["auroc"]
    # 旧版校准 JSON（特征名为 logit(classifier)）仍然可用
    old = dict(cal); old["lr"] = dict(cal["lr"], features=["fastdetect", "binoculars", "logit(classifier)"],
                                      w=cal["lr"]["w"][:3], mean=cal["lr"]["mean"][:3], std=cal["lr"]["std"][:3])
    assert 0 <= scoring.combine(ai[0], old)["prob"] <= 1


def test_calibration_separates_synthetic():
    import random
    rnd = random.Random(0)
    human = [{"fastdetect": rnd.gauss(0.5, 0.5), "binoculars": rnd.gauss(1.0, 0.05), "classifier": rnd.uniform(0.05, 0.5)} for _ in range(60)]
    ai = [{"fastdetect": rnd.gauss(3.0, 0.7), "binoculars": rnd.gauss(0.8, 0.05), "classifier": rnd.uniform(0.5, 0.98)} for _ in range(60)]
    res = scoring.calibrate(human, ai, 0.05)
    cal, rep = res["calibration"], res["report"]
    assert cal["calibrated"] and "lr" in cal
    assert rep["combined_auroc"] > 0.95
    assert rep["human_flagged_rate"] <= 0.05 + 1e-9
    assert cal["signals"]["binoculars"]["direction"] == -1 and cal["signals"]["fastdetect"]["direction"] == 1
    assert scoring.combine(ai[0], cal)["prob"] > scoring.combine(human[0], cal)["prob"]


# ---------------- 接口 ----------------

@pytest.fixture(scope="module")
def client():
    from app import main
    c = TestClient(main.app)
    for _ in range(300):
        if not main.engine.loading:
            break
        time.sleep(0.1)
    return c


ADMIN = {"X-Admin-Token": "test-admin-pw"}


def issue(client, **kw):
    r = client.post("/admin/api/keys", json={"name": "t", **kw}, headers=ADMIN)
    assert r.status_code == 200, r.text
    return r.json()["key"]


def poll(client, jid, headers, path="/v1/jobs/"):
    for _ in range(3000):
        j = client.get(path + jid, headers=headers).json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.05)
    raise AssertionError("timeout")


def test_health(client):
    h = client.get("/health").json()
    assert h["lm"]["ready"] and h["classifier"]["ready"] and h["classifier_en"]["ready"], h
    assert h["requires_key"] and h["key_signing_configured"]
    assert h["calibration"]["calibrated"] is False


def test_pages(client):
    assert "审读" in client.get("/").text
    assert "管理员登录" in client.get("/admin").text
    assert client.get("/static/app.js").status_code == 200


def test_auth_errors(client):
    assert client.post("/v1/detect", json={"text": MODERN}).json()["error"] == "missing_key"
    assert client.post("/v1/detect", json={"text": MODERN}, headers={"X-API-Key": "atc-bad.bad"}).json()["error"] == "invalid_key"
    assert client.post("/admin/api/keys", json={}, headers={"X-Admin-Token": "wrong"}).status_code == 401


def test_detect_sync(client):
    h = {"Authorization": "Bearer " + issue(client)}
    r = client.post("/v1/detect", json={"text": MODERN * 3}, headers=h)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["status"] == "done"
    s = j["result"]["summary"]
    assert s["methods"] == {"fastdetect": True, "binoculars": True, "classifier": True, "classifier_en": True}
    assert 0 <= s["ai_rate"] <= 1 and s["counted_chars"] > 0
    seg = j["result"]["segments"][0]
    assert set(seg["raw"]) >= {"fastdetect", "binoculars", "classifier", "ppl", "lrr", "log_rank", "entropy", "top10", "style_cv"}
    assert set(s["segments_by_level"]) == {"high", "mid", "light", "low"}
    assert "reliability_notes" in s and any("校准" in n for n in s["reliability_notes"])
    assert 0 <= seg["prob"] <= 1


def test_all_quotation_falls_back_and_excluded_get_reference_value(client):
    h = {"Authorization": "Bearer " + issue(client)}
    # 通篇文言：文言就是正文，照常计入（不再被当作引文排除）
    s = client.post("/v1/detect", json={"text": CLASSICAL * 3}, headers=h).json()["result"]
    assert s["summary"]["fallback_all_counted"] is False
    assert s["summary"]["counted_chars"] > 0 and s["summary"]["ai_rate"] is not None
    assert s["summary"]["main_register"] == "zh_classical"
    assert any("文言" in n for n in s["summary"]["reliability_notes"])
    # 只有参考文献：退回为全部计入，报告仍有数值
    refs = "参考文献\n" + "\n".join(f"[{i}] 脱脱等：《宋史》卷{i}，北京：中华书局，1977年，第{i*3}页。" for i in range(1, 12))
    s = client.post("/v1/detect", json={"text": refs}, headers=h).json()["result"]
    assert s["summary"]["fallback_all_counted"] is True
    assert all(seg["prob"] is not None for seg in s["segments"])
    # 正文 + 文言引文：引文不计入，但有参考值
    r = client.post("/v1/detect", json={"text": MODERN * 2 + "\n\n" + CLASSICAL * 2}, headers=h).json()["result"]
    assert r["summary"]["fallback_all_counted"] is False
    q = [seg for seg in r["segments"] if seg["kind"] == "quotation"]
    assert q and all(seg["prob"] is None and seg["ref_prob"] is not None for seg in q)


def test_pages_versioned_and_not_cached(client):
    r = client.get("/")
    assert "/static/app.js?v=" in r.text and "/static/library.js?v=" in r.text
    assert r.headers["cache-control"] == "no-cache"
    assert client.get("/static/app.js").headers["cache-control"] == "no-cache"


def test_detect_long_async_fast_mode(client):
    h = {"X-API-Key": issue(client)}
    doc = "\n\n".join(MODERN for _ in range(1200))   # 约 16 万字
    t0 = time.time()
    r = client.post("/v1/detect", json={"text": doc, "mode": "fast"}, headers=h)
    assert r.status_code == 202, r.text
    j = poll(client, r.json()["id"], h)
    assert j["status"] == "done", j
    s = j["result"]["summary"]
    assert s["lm_sampled"] and s["lm_scored_segments"] == 60
    assert s["segments"] > 60
    print(f"\n160k chars fast mode (tiny models): {time.time()-t0:.1f}s")


def test_job_owner_isolation(client):
    h1, h2 = {"X-API-Key": issue(client)}, {"X-API-Key": issue(client)}
    r = client.post("/v1/detect", json={"text": MODERN * 30, "wait": False}, headers=h1)
    jid = r.json()["id"]
    assert client.get("/v1/jobs/" + jid, headers=h2).status_code == 404
    assert poll(client, jid, h1)["status"] == "done"


def test_quota(client):
    h = {"X-API-Key": issue(client, daily_chars=500)}
    assert client.post("/v1/detect", json={"text": MODERN * 2}, headers=h).status_code == 200
    r = client.post("/v1/detect", json={"text": MODERN * 2}, headers=h)
    assert r.status_code == 429 and r.json()["error"] == "quota_exceeded"


def test_file_upload(client, tmp_path):
    import zipfile
    h = {"X-API-Key": issue(client)}
    p = tmp_path / "t.docx"
    body = "".join(f"<w:p><w:r><w:t>{MODERN}</w:t></w:r></w:p>" for _ in range(3))
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("word/document.xml", '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
                   + body + "</w:body></w:document>")
    r = client.post("/v1/detect/file", files={"file": ("t.docx", p.read_bytes())}, headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["result"]["summary"]["total_chars"] >= len(MODERN) * 3
    r = client.post("/v1/detect/file", files={"file": ("t.exe", b"xx")}, headers=h)
    assert r.json()["error"] == "bad_file"


def test_calibrate_endpoint_and_apply(client):
    human = [MODERN * 2] * 8
    ai = [("综上所述，该制度在很大程度上体现了中央集权的发展趋势，具有重要意义。此外，值得注意的是，这一变化对后世产生了深远影响，"
           "为理解宋代政治提供了有力支撑。总的来说，其演变是多种因素共同作用的结果。") * 3] * 8
    r = client.post("/admin/api/calibrate", json={"human": human, "ai": ai}, headers=ADMIN)
    assert r.status_code == 200, r.text
    j = poll(client, r.json()["id"], ADMIN, "/admin/api/jobs/")
    assert j["status"] == "done", j
    cal = j["result"]["calibration"]
    assert cal["calibrated"]
    assert client.post("/admin/api/calibration", json={"calibration": cal}, headers=ADMIN).json()["ok"]
    assert client.get("/health").json()["calibration"]["calibrated"] is True


def test_english_uses_english_classifier(client):
    from app import main
    h = {"Authorization": "Bearer " + issue(client)}
    r = client.post("/v1/detect", json={"text": ENGLISH * 6 + "\n\n" + MODERN * 3, "wait": True}, headers=h).json()["result"]
    regs = {seg["register"] for seg in r["segments"]}
    assert regs == {"en", "zh"}, regs
    en_seg = next(seg for seg in r["segments"] if seg["register"] == "en")
    zh_seg = next(seg for seg in r["segments"] if seg["register"] == "zh")
    # 英文段落的分类器分数来自英文分类器，与中文分类器给同一段的分数不同
    assert en_seg["raw"]["classifier"] == round(main.engine.cls_en.predict([en_seg["text"]])[0], 4)
    assert zh_seg["raw"]["classifier"] == round(main.engine.cls.predict([zh_seg["text"]])[0], 4)
    assert set(r["summary"]["chars_by_register"]) == {"en", "zh"}
    assert any("多种文体" in n for n in r["summary"]["reliability_notes"])


def test_calibrate_one_profile_keeps_others(client):
    human = [ENGLISH * 4] * 6
    ai = [("Furthermore, it is worth noting that the administrative landscape of the Song dynasty serves as a testament "
           "to the intricate interplay of central authority and local governance. Moreover, this multifaceted system "
           "played a pivotal role in fostering stability. ") * 6] * 6
    r = client.post("/admin/api/calibrate", json={"human": human, "ai": ai, "profile": "auto"}, headers=ADMIN)
    j = poll(client, r.json()["id"], ADMIN, "/admin/api/jobs/")
    assert j["status"] == "done", j
    cal = j["result"]["calibration"]
    assert cal["profile"] == "en" and j["result"]["report"]["profile_name"] == "英文"
    d = client.post("/admin/api/calibration", json={"calibration": cal}, headers=ADMIN).json()
    assert d["calibration"]["profiles"]["en"]["calibrated"]
    h = client.get("/health").json()["calibration"]["profiles"]
    assert h["en"] is True
