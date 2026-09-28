"""把长文切成适合检测的段落，并识别不应计入 AI 率的部分（参考文献、以引文为主的段落）。"""
import re
from dataclasses import dataclass, field

from . import config

_SENT_END = re.compile(r"(?<=[。！？!?；;])")
_REF_HEAD = re.compile(
    r"^\s*(?:[一二三四五六七八九十\d]+[、.．\s]*)?(参考文献|参考书目|引用文献|征引文献|主要参考文献|References|Bibliography|Works Cited)\s*[:：]?\s*$",
    re.IGNORECASE,
)
# 参考文献之后又出现这些标题时，恢复计入（例如"附录""致谢"不计，"后记"不计；这里只处理常见情形）
_AFTER_REF_HEAD = re.compile(r"^\s*(附录|致谢|后记|Appendix|Acknowledg)", re.IGNORECASE)
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


def is_title(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 40:
        return False
    if re.search(r"[。！？!?；;，,、]$", s):          # 像句子或诗行的结尾
        return False
    if re.fullmatch(r".{1,12}[：:]", s):              # "对联：" 这类短标签
        return True
    if ("，" in s or "," in s) and "——" not in s and len(s) > 16:
        return False                                  # 带逗号的长句多半是正文
    return bool(_TITLE_MARK.search(s))


def _body_lines(text: str) -> str:
    """去掉标题行，只看正文（判断文体时用）。"""
    lines = [l for l in text.splitlines() if l.strip()]
    body = [l for l in lines if not is_title(l)]
    return "\n".join(body) if body else text


def is_poetry(text: str) -> bool:
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
    if latin >= 30 and latin >= 2 * cjk:
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

    def flush(kind_override=None):
        nonlocal buf
        if not buf.strip():
            buf = ""
            return
        reg = detect_register(buf)
        tgt = target_chars(reg)
        title = buf.strip().splitlines()[0].strip() if is_title(buf.strip().splitlines()[0]) else ""
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
        if in_refs:
            if not buf:
                buf_start = line_start
            buf += line
            if len(buf) > target * 4:
                flush("reference")
            continue
        if not stripped:
            if buf.strip():
                pending_break = True
            continue
        title = is_title(stripped)
        if buf.strip():
            reg = detect_register(buf)
            if title and not (buf.strip() and is_title(buf.strip().splitlines()[-1])):
                # 新作品开始（连续两行标题视为同一个标题块）
                flush()
                block += 1
            elif len(stripped) >= 30 and len(buf.strip()) >= 30 and detect_register(stripped) != reg:
                # 文体切换（如现代文里插入一段文言引文、中文里夹一段英文）时另起一段
                flush()
            elif pending_break and len(buf) >= flush_chars(reg):
                flush()
        elif title and segments:
            block += 1
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
    """标出"以引文为主"的段落（不计入 AI 率）。按整篇文稿判断，避免把作者自己的文字误当引文：
    - 现代汉语论文里夹的大段文言 → 视为古籍引文；但如果全文以文言为主（如文言小说、仿古文），文言就是正文。
    - 现代汉语文章里没有标题、夹在正文中的诗词 → 视为引用的诗词；带标题的（如"七律《……》"）是独立作品，照常计入。
    - 引号内文字占一半以上的段落 → 视为引文；但如果这类段落超过正文的 30%，或全文以文言为主，多半是小说/对话体，照常计入。"""
    body = [s for s in segments if s.kind == "body"]
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
