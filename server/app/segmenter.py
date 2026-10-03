"""把长文切成适合检测的段落，并识别不应计入 AI 率的部分（参考文献、以引文为主的段落）。"""
import re
from dataclasses import dataclass, field

from . import config

_SENT_END = re.compile(r"(?<=[。！？!?；;])(?![”’」』\"'])|(?<=[。！？!?；;][”’」』\"'])")
_REF_HEAD = re.compile(
    r"^\s*(?:[一二三四五六七八九十\d]+[、.．\s]*)?(参考文献|参考书目|引用文献|征引文献|主要参考文献|References|Bibliography|Works Cited)\s*[:：]?\s*$",
    re.IGNORECASE,
)
# 参考文献之后又出现这些标题时，恢复计入（例如"附录""致谢"不计，"后记"不计；这里只处理常见情形）
_AFTER_REF_HEAD = re.compile(r"^\s*(附录|致谢|后记|Appendix|Acknowledg)", re.IGNORECASE)
# 参考文献条目的样子：[1] / 1. / (1) 开头，或带"作者. 年份"、期刊卷期页码等
_REF_ENTRY = re.compile(r"^\s*(\[\d+\]|［\d+］|\(\d+\)|\d+[.、]\s)|\b(19|20)\d{2}[a-z]?[.,;)]|\d+\s*[(:（]\s*\d+|pp?\.\s*\d|doi[:.]|https?://",
                        re.IGNORECASE)


def _looks_ref(t: str) -> bool:
    t = t.strip()
    return bool(_REF_ENTRY.search(t[:160])) or bool(re.search(r"\bet al\.|\bJ\.|Press\b|出版社|学报", t))


def _ends_references(line: str, prev_lines: list) -> bool:
    """参考文献之后又开始了新的正文（例如一个文档里放了两篇论文）：
    出现章节标题（摘要 / Abstract / 引言…）；或一段不像参考文献条目的长段正文（≥ 250 字、含 2 句以上）；
    或连续两行都不像参考文献条目（没有年份、卷期、[1] 编号等）的 ≥ 60 字正文行。"""
    s = line.strip()
    if not s or _REF_HEAD.match(s):
        return False
    if is_section_heading(s):
        return True
    looks_ref = _looks_ref
    if len(s) >= 250 and not looks_ref(s):
        return len(re.findall(r"[。！？!?]|\.\s+[A-Z]|\.(?=[A-Z][a-z])", s)) >= 2 or len(s) >= 400
    prev = [p.strip() for p in prev_lines if p.strip() and not _RULE_LINE.match(p.strip())]
    if len(s) >= 100 and not looks_ref(s) and prev and len(prev[-1]) >= 100 and not looks_ref(prev[-1]) \
            and not _REF_HEAD.match(prev[-1]):
        return True
    return False
_CJK = re.compile(r"[㐀-䶿一-鿿]")
_LATIN = re.compile(r"[A-Za-z]")
# 文言常用虚词 / 现代汉语标志词（用于区分文言与白话；阈值用 NLPCC 现代文与 NiuTrans 古文语料测定：
# 现代文误判为文言约 0.7%，文言识别率约 94%）
_CLASSICAL = re.compile(r"[之乎者也矣焉哉曰兮盖乃遂耶欤其而于以]")
_MODERN = re.compile(r"[的了们这那吗呢着么]|他们|我们|就是|因为|但是|没有|一个|进行|通过|发展|问题")
_QUOTED = re.compile(r"“[^”]{4,}”|「[^」]{4,}」|『[^』]{4,}』")
# 英文分句：句末标点后接空白、下一句以大写/引号/括号开头
_EN_SENT_END = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"”’)]))\s+(?=[A-Z\"“‘(])")

REGISTERS = ("zh", "zh_classical", "zh_poetry", "en")
REGISTER_NAMES = {"zh": "现代汉语", "zh_classical": "文言", "zh_poetry": "诗词", "en": "英文"}

# 诗词 / 对联：按标点和换行切成小句，句长整齐（3–7 字）、几乎不用虚词和白话标志词。
# 阈值用 ChangAn 诗词语料与 NiuTrans 古文、NLPCC 现代文测定：诗词识别率约 90%，
# 文言散文误判为诗词约 1.4%，现代文 0%。
_CLAUSE_SPLIT = re.compile(r"[，。！？、；;：:\n,!?　 ]")
_POETRY_FUNC = re.compile(r"[之乎者也矣焉哉曰其而乃遂则]")
# 标题行：书名号、间隔号、破折号、"对联：" 这类短标签、章节编号、Markdown 标题
_TITLE_MARK = re.compile(r"[《》·]|——|^#{1,6}\s|^(第[一二三四五六七八九十百千\d]+[章节回部分卷]|[一二三四五六七八九十]+、|\d+(\.\d+)*[、.．\s])")


@dataclass
class Segment:
    index: int
    text: str
    start: int                       # 在原文中的字符位置
    kind: str = "body"               # body | reference | quotation
    notes: list = field(default_factory=list)
    register: str = "zh"             # zh | zh_classical | zh_poetry | en
    block: int = 0                   # 第几篇作品（遇到标题行加一；平滑不跨作品）
    title: str = ""                  # 本段开头的标题行（如有）

    @property
    def counted(self) -> bool:
        return self.kind == "body"


def classical_ratio(text: str) -> float:
    cjk = len(_CJK.findall(text)) or 1
    return len(_CLASSICAL.findall(text)) / cjk


def modern_ratio(text: str) -> float:
    cjk = len(_CJK.findall(text)) or 1
    return len(_MODERN.findall(text)) / cjk


_EN_TITLE_SKIP = re.compile(r"^(stage|step|part|phase|section|chapter|appendix|table|figure|fig\.|week|month|day)\b", re.I)
_EN_SMALL = {"a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "with", "by", "at", "from", "as", "vs", "via", "into"}


def is_english_title(line: str) -> bool:
    """英文作品标题：单独一行、3–16 个词、不以句号结尾、实词大多首字母大写（Title Case），
    不是编号小节（"2. Methods"）或 "Stage 1: …" 这类分节标签。"""
    s = line.strip().strip("《》\"'")
    if not s or len(s) > 110 or _CJK.search(s) or re.search(r"[.!?;:,]$", s) or ";" in s or _BOX.search(s):
        return False
    if re.match(r"^\d+(\.\d+)*[.)、]?\s", s) or _EN_TITLE_SKIP.match(s):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'’\-]*", s)
    if not 3 <= len(words) <= 16:
        return False
    content = [w for w in words if w.lower() not in _EN_SMALL]
    return bool(content) and sum(w[0].isupper() for w in content) >= 0.8 * len(content)


def is_title(line: str) -> bool:
    s = line.strip()
    if s and len(s) <= 110 and is_english_title(s):
        return True
    if s and 6 <= len(s) <= 24 and re.fullmatch(r"[\u4e00-\u9fff]{3,11}[,，、\s][\u4e00-\u9fff]{3,11}", s):
        return True                                   # "弘扬长征精神，传承红色文化" 这类两句式标题
    if not s or len(s) > 40:
        return False
    if re.search(r"[。！？!?；;，,、]$", s):          # 像句子或诗行的结尾
        return False
    if re.fullmatch(r".{1,12}[：:]", s) and _CJK.search(s):   # "对联：" 这类短标签（英文的 "Weekly Tasks:" 是小节标签，不算新作品）
        return True
    # 单独一行、不带任何标点的短标题（如"黄四娘"）；5 字 / 7 字的可能是没加标点的诗句，不算
    cjk = len(_CJK.findall(s))
    if (2 <= cjk <= 16 and cjk == len(re.sub(r"\s", "", s)) and cjk not in (5, 7)):
        return True
    if ("，" in s or "," in s) and "——" not in s and len(s) > 16:
        return False                                  # 带逗号的长句多半是正文
    return bool(_TITLE_MARK.search(s))


# 论文里的章节标题（"1. Introduction""2.2 Anomalies""一、研究背景""（二）""摘要""References"……）：
# 它们是同一篇文章内部的分节，不是新作品——不应把一篇论文切成十几篇互不相干的"作品"
_SECTION_NUM = re.compile(r"^(\d+(\.\d+)*[.、．]?\s*\S|[IVX]{1,5}[.、]\s*\S|[一二三四五六七八九十]+、|（[一二三四五六七八九十\d]+）|第[一二三四五六七八九十百\d]+[章节部分])")
_SECTION_NAMES = re.compile(
    r"^(#{1,6}\s*)?(abstract|introduction|background|literature review|related work|theoretical framework|method(s|ology)?|"
    r"data( and methods?)?|results?|findings|analysis|discussion|conclusions?|limitations|future work|keywords?|"
    r"acknowledge?ments?|appendix|摘\s*要|关键词|引\s*言|绪\s*论|前\s*言|文献综述|研究方法|研究设计|结\s*论|结\s*语|讨\s*论|致\s*谢|附\s*录)"
    r"\b.{0,50}$", re.I)
_ABSTRACT_HEAD = re.compile(r"^(摘\s*要|内容摘要|内容提要|abstract)\s*([:：]|$|\s)", re.I)
_KEYWORDS_LINE = re.compile(r"^(关键词|关键字|key\s*words?)\s*[:：]", re.I)
_RULE_LINE = re.compile(r"^[-—_*=~·\s]{3,}$")
_BOX = re.compile(r"[\u2500-\u257f\u2580-\u259f]")


def is_drawing_line(line: str) -> bool:
    """用制表符画的框图 / 表格线（┌─┬─┐、│ Smart Home │ …）：不是正文，不参与检测。"""
    s = line.strip()
    n = len(_BOX.findall(s))
    return bool(s) and (n >= 3 or (n >= 1 and (_BOX.match(s[0]) is not None or _BOX.match(s[-1]) is not None)))


def is_section_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 60 or s.startswith("《"):
        return False
    return bool(_SECTION_NUM.match(s) or _SECTION_NAMES.match(s))


def _is_head_line(l: str) -> bool:
    return bool(_KEYWORDS_LINE.match(l.strip()) or is_section_heading(l))


def _body_lines(text: str) -> str:
    """去掉标题行、章节标题、关键词行，只看正文（判断文体时用）。"""
    lines = [l for l in text.splitlines() if l.strip()]
    core = [l for l in lines if not _is_head_line(l)]
    if not core:
        return text
    body = [l for l in core if not is_title(l)]
    return "\n".join(body) if body else "\n".join(core)


def is_poetry(text: str) -> bool:
    if all(_is_head_line(l) for l in text.splitlines() if l.strip()):
        return False                                  # 只有"关键词：……""一、引言"这类行，不是诗
    t = _body_lines(text)
    clauses = [len(_CJK.findall(c)) for c in _CLAUSE_SPLIT.split(t) if _CJK.search(c)]
    if len(clauses) < 4:
        return False
    if any(n < 3 or n > 7 for n in clauses):
        return False
    mean = sum(clauses) / len(clauses)
    sd = (sum((n - mean) ** 2 for n in clauses) / len(clauses)) ** 0.5
    cjk = len(_CJK.findall(t)) or 1
    return sd <= 2.0 and len(_POETRY_FUNC.findall(t)) / cjk <= 0.05 and modern_ratio(t) <= 0.02


def detect_register(text: str) -> str:
    """zh（现代汉语）/ zh_classical（文言）/ zh_poetry（诗词、对联）/ en（英文及其他拉丁字母语言）。"""
    t = _body_lines(text)
    cjk = len(_CJK.findall(t))
    latin = len(_LATIN.findall(t))
    # 拉丁字母为主就是英文（短诗行也算，例如 "A cap of flowers, and a kirtle" 只有 24 个字母）
    if latin >= 8 and latin >= 2 * cjk:
        return "en"
    if cjk >= 12 and is_poetry(t):
        return "zh_poetry"
    if cjk >= 20 and classical_ratio(t) >= config.CLASSICAL_THRESHOLD and modern_ratio(t) <= config.MODERN_MAX_RATIO:
        return "zh_classical"
    return "zh"


def target_chars(register: str) -> int:
    # 英文 1 个 token 约 4 个字母，中文约 1.4 个字；英文段落放长一些，检测才稳定（Turnitin 要求至少 300 词）
    return config.SEGMENT_TARGET_CHARS_EN if register == "en" else config.SEGMENT_TARGET_CHARS


def min_chars(register: str) -> int:
    if register == "zh_poetry":
        return 16
    return config.SEGMENT_MIN_CHARS_EN if register == "en" else config.SEGMENT_MIN_CHARS


def flush_chars(register: str) -> int:
    """遇到段落分隔时，当前窗口至少要这么长才另起一段；更短的相邻段落会合并（同一作品内）。
    短段落信号弱、误差大，合并成一两百字以上的窗口再判断——与 Turnitin、知网按"语篇窗口"判断的做法一致。"""
    return {"en": config.SEGMENT_FLUSH_CHARS_EN, "zh_classical": config.SEGMENT_FLUSH_CHARS_CLASSICAL,
            "zh_poetry": 10 ** 9}.get(register, config.SEGMENT_FLUSH_CHARS)


def _split_sentences(par: str, register: str):
    if register == "en":
        return [s for s in _EN_SENT_END.split(par) if s.strip()]
    return [s for s in _SENT_END.split(par) if s.strip()]


def _split_long(par: str, target: int, register: str = "zh"):
    sents = _split_sentences(par, register)
    out, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) > target * 1.5:
            out.append(cur)
            cur = ""
        cur += s
        if len(cur) >= target:
            out.append(cur)
            cur = ""
    if cur.strip():
        # 末尾剩下的短句并入上一段（英文段落本来就长，剩下不到半段也并入）
        if out and len(cur) < (target * 0.5 if register == "en" else min_chars(register)):
            out[-1] += cur
        else:
            out.append(cur)
    return out


def quotation_ratio(text: str) -> float:
    cjk = len(_CJK.findall(text)) or 1
    quoted = sum(len(m.group(0)) for m in _QUOTED.finditer(text))
    return min(1.0, quoted / cjk)


def segment_text(text: str, exclude_references: bool = True, flag_quotations: bool = True):
    target = config.SEGMENT_TARGET_CHARS
    segments: list[Segment] = []
    in_refs = False
    buf, buf_start = "", 0
    pos = 0
    block = 0
    pending_break = False
    new_block_pending = False
    # 论文题目：下一行就是"摘要 / Abstract"的短行（中文长题目没有书名号、超过 16 字也能认出来）
    lines_all = [l.strip() for l in text.splitlines()]
    nonempty = [l for l in lines_all if l]
    paper_titles = {a for a, b in zip(nonempty, nonempty[1:])
                    if _ABSTRACT_HEAD.match(b) and 4 <= len(a) <= 160 and not _ABSTRACT_HEAD.match(a)
                    and not _SECTION_NAMES.match(a) and not re.search(r"[。！？!?；;，,]$", a)}
    # 中文短标题（"老农的回忆"）：上一行以句末标点结束、本行 2–16 个汉字且无标点、下一行是正文长句
    for prev, a, b in zip(nonempty, nonempty[1:], nonempty[2:]):
        if (re.fullmatch(r"[《]?[\u4e00-\u9fff]{2,16}[》]?", a) and re.search(r"[。！？!?”」…]$", prev)
                and len(b) >= 15 and re.search(r"[。！？，]", b) and not is_section_heading(a)
                and not _SECTION_NAMES.match(a) and not _KEYWORDS_LINE.match(a)):
            paper_titles.add(a)
    # 中文之后紧跟的短英文标题（"Returning Home"）：1–4 个首字母大写的词、下一行是英文长段落、上一行是中文
    for prev, a, b in zip(nonempty, nonempty[1:], nonempty[2:]):
        words = re.findall(r"[A-Za-z][A-Za-z'’\-]*", a)
        if (1 <= len(words) <= 4 and len(a) <= 40 and all(w[0].isupper() for w in words)
                and not re.search(r"[.!?;:,]$", a) and not _CJK.search(a) and _CJK.search(prev)
                and len(b) >= 150 and len(_LATIN.findall(b)) >= 0.6 * len(b) and not is_section_heading(a)
                and not _EN_TITLE_SKIP.match(a)):
            paper_titles.add(a)

    work_heads: set = set()          # 文集里带编号的作品标题（"1. 天坛：圆丘上的沉默""4. Hawaii"）
    blk_paper, blk_numbered = False, False

    def flush(kind_override=None):
        nonlocal buf
        if not buf.strip():
            buf = ""
            return
        reg = detect_register(buf)
        tgt = target_chars(reg)
        first = buf.strip().splitlines()[0].strip()
        title = first if (first in paper_titles or first in work_heads
                          or (is_title(first) and not is_section_heading(first))) else ""
        if reg == "zh_poetry" or len(buf) <= tgt * 1.5:
            pieces = [buf]
        else:
            pieces = _split_long(buf, tgt, reg)
        off = buf_start
        for i, p in enumerate(pieces):
            segments.append(Segment(len(segments), p, off, kind_override or "body", [],
                                    detect_register(p) if len(pieces) > 1 else reg, block, title if i == 0 else ""))
            off += len(p)
        buf = ""

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        line_start = pos
        pos += len(line)
        if exclude_references and _REF_HEAD.match(stripped):
            flush()
            in_refs = True
            pending_break = False
            buf_start = line_start
            buf = line
            continue
        if in_refs and _AFTER_REF_HEAD.match(stripped):
            flush("reference")
            in_refs = False
        elif in_refs and _ends_references(stripped, buf.split("\n")):
            # 参考文献结束、新作品开始：参考文献末尾紧挨着的标题行（新论文题目）归入新作品
            lines = buf.rstrip("\n").split("\n")
            carry = []
            while lines and len(lines) > 1 and (not lines[-1].strip() or _RULE_LINE.match(lines[-1].strip())
                                                 or not _looks_ref(lines[-1])):
                carry.insert(0, lines.pop())
                if len(carry) > 8:
                    break
            carry_text = "\n".join(l for l in carry if l.strip() and not _RULE_LINE.match(l.strip())).strip()
            buf = "\n".join(lines) + "\n" if lines else ""
            flush("reference")
            in_refs = False
            block += 1
            new_block_pending = True
            blk_paper, blk_numbered = False, False
            pending_break = False
            if carry_text and not _REF_HEAD.match(carry_text):
                buf_start = line_start - len(carry_text) - 1
                buf = carry_text + "\n"
        if in_refs:
            if not buf:
                buf_start = line_start
            buf += line
            if len(buf) > target * 4:
                flush("reference")
            continue
        if not stripped or _RULE_LINE.match(stripped) or is_drawing_line(stripped):
            if buf.strip():
                pending_break = True
            continue
        section = is_section_heading(stripped)
        title = stripped in paper_titles or (is_title(stripped) and not section)
        block_before = block
        # 论文内部不带编号的英文小标题（"Risk Factors and Prevention"）是章节，不是新作品
        if title and blk_paper and stripped not in paper_titles and not _CJK.search(stripped):
            title, section = False, True
        # 文集里带编号的作品标题：文体与上一篇不同（中英切换），或当前这篇不是论文、且它自己的标题也带编号
        top = (bool(re.match(r"^\d+\s*[.、．]\s*\S", stripped)) and not re.match(r"^\d+\.\d", stripped)
               and len(stripped) <= 90 and not _SECTION_NAMES.match(re.sub(r"^\d+\s*[.、．]\s*", "", stripped)))
        if top and not title and new_block_pending:
            section, title = False, True          # 参考文献之后紧跟的编号标题，就是下一篇的题目
            work_heads.add(stripped)
        elif top and not title and (buf.strip() or segments):
            h_reg = "zh" if _CJK.search(stripped) else "en"
            prev_text = buf if buf.strip() else segments[-1].text
            cur = "zh" if len(_CJK.findall(prev_text)) * 2 > len(_LATIN.findall(prev_text)) else "en"
            if h_reg != cur or (not blk_paper and blk_numbered):
                section, title = False, True
                work_heads.add(stripped)
        if buf.strip():
            reg = detect_register(buf)
            if title and not (buf.strip() and is_title(buf.strip().splitlines()[-1])):
                # 新作品开始（连续两行标题视为同一个标题块）
                flush()
                block += 1
            elif len(stripped) >= 30 and len(buf.strip()) >= 30 and detect_register(stripped) != reg:
                # 文体切换（如现代文里插入一段文言引文、中文里夹一段英文）时另起一段
                flush()
            elif (pending_break or section) and len(buf) >= flush_chars(reg):
                # 章节标题相当于一次段落分隔：窗口够长就另起一段，否则与下一节合并（仍属同一篇作品）
                flush()
        elif title and segments and not new_block_pending:
            block += 1
        was_pending, new_block_pending = new_block_pending, False
        if block != block_before or was_pending or (not segments and not buf.strip()):
            blk_paper, blk_numbered = False, bool(title and top)
        if _ABSTRACT_HEAD.match(stripped) or _KEYWORDS_LINE.match(stripped):
            blk_paper = True
        pending_break = False
        if not buf:
            buf_start = line_start
        buf += line
        reg = detect_register(buf[:600])
        if reg != "zh_poetry" and len(buf) >= target_chars(reg) * 1.5:
            flush()
    flush("reference" if in_refs else None)
    segments = [s for s in segments if s.text.strip()]
    for s in segments:
        s.text = s.text.strip()
    segments = _merge_short(segments)
    for i, s in enumerate(segments):
        s.index = i
    if flag_quotations:
        _flag_quotations(segments)
    return segments


def _merge_short(segments):
    """过短的段落（如结尾的一两句话）并入同一作品、同一文体的相邻段落。"""
    out: list[Segment] = []
    for s in segments:
        prev = out[-1] if out else None
        if (prev and s.kind == prev.kind == "body" and s.block == prev.block and s.register == prev.register
                and s.register != "zh_poetry" and (len(s.text) < min_chars(s.register) or len(prev.text) < min_chars(prev.register))):
            prev.text = prev.text + "\n" + s.text
            continue
        out.append(s)
    return out


def _flag_quotations(segments):
    """标出"以引文为主"的段落（不计入 AI 率）。按"作品"（标题行分开的每一块）判断，避免把作者自己的文字误当引文：
    - 现代汉语论文里夹的大段文言 → 视为古籍引文；但如果这篇作品以文言为主（如文言小说、仿古文），文言就是正文。
    - 现代汉语文章里没有标题、夹在正文中的诗词 → 视为引用的诗词；带标题的（如"七律《……》"）是独立作品，照常计入。
    - 引号内文字占一半以上的段落 → 视为引文；但如果这类段落超过该作品正文的 30%，或作品以文言为主，多半是小说/对话体，照常计入。"""
    blocks: dict[int, list] = {}
    for s in segments:
        if s.kind == "body":
            blocks.setdefault(s.block, []).append(s)
    for body in blocks.values():
        total = sum(len(s.text) for s in body) or 1
        cjk_body = [s for s in body if s.register != "en"]
        cjk_total = sum(len(s.text) for s in cjk_body) or 1
        classical_chars = sum(len(s.text) for s in cjk_body if s.register == "zh_classical")
        classical_doc = classical_chars >= 0.5 * cjk_total
        modern_doc = sum(len(s.text) for s in cjk_body if s.register == "zh") >= 0.5 * cjk_total
        quoted = [(s, quotation_ratio(s.text)) for s in cjk_body]
        quote_heavy = [s for s, q in quoted if q >= 0.5]
        dialogue_doc = classical_doc or sum(len(s.text) for s in quote_heavy) >= 0.3 * total
        for s, q in quoted:
            if q >= 0.5 and not dialogue_doc:
                s.kind, s.notes = "quotation", [f"引号内文字约占 {q:.0%}"]
            elif s.register == "zh_classical" and not classical_doc and len(s.text) >= 40:
                s.kind, s.notes = "quotation", [f"文言段落（文言虚词 {classical_ratio(s.text):.0%}），疑为古籍引文"]
            elif s.register == "zh_poetry" and modern_doc and not s.title:
                s.kind, s.notes = "quotation", ["正文中引用的诗词（无标题）"]


# 古籍语料（NiuTrans）排印时不用引号，而 AI 写的文言几乎都带引号：如果把引号送进模型，
# 模型会学成"有引号就是 AI"，从而冤枉带引号的真人文言。文言段落送进模型前一律去掉引号，只看文字本身。
_QUOTES = re.compile(r"[“”\"「」『』‘’＂]")


def normalize_classical(text: str) -> str:
    return _QUOTES.sub("", text)


# 从网页 / 聊天窗口复制到 Word 时，英文句号后的空格常被吞掉（"income.However"），分词会因此变得很怪，
# 语言模型和分类器都把它当成"不常见的写法"而偏向人写。英文段落送进模型前补回句间空格（小数、缩写不受影响）。
_EN_GLUED = re.compile(r"(?<=[a-z%\)\]])([.!?;:])(?=[A-Z][a-z])")
_EN_NUM_HEAD = re.compile(r"(?m)^(\d+(?:\.\d+)*\.)(?=[A-Z])")


def normalize_english(text: str) -> str:
    text = _EN_GLUED.sub(r"\1 ", text)
    return _EN_NUM_HEAD.sub(r"\1 ", text)


_EN_PAPER_HEAD = re.compile(
    r"^\s*(\d+(\.\d+)*\.?\s*)?(abstract|keywords?|introduction|materials? and methods|methods?|methodology|"
    r"results?( and (analysis|discussion))?|discussion|conclusions?|references|literature review)\b", re.I)


def is_english_paper(text: str) -> bool:
    """英文学术论文：至少有 3 个不同的常见章节标题（Abstract、Introduction、Methods、Results、Conclusion、References 等）。"""
    found = set()
    for line in text.splitlines():
        if len(line.strip()) > 80:
            continue
        m = _EN_PAPER_HEAD.match(line.strip())
        if m:
            found.add(m.group(3).lower().split()[0])
    return len(found) >= 3
