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

REGISTERS = ("zh", "zh_classical", "en")
REGISTER_NAMES = {"zh": "现代汉语", "zh_classical": "文言", "en": "英文"}


@dataclass
class Segment:
    index: int
    text: str
    start: int                       # 在原文中的字符位置
    kind: str = "body"               # body | reference | quotation
    notes: list = field(default_factory=list)
    register: str = "zh"             # zh | zh_classical | en

    @property
    def counted(self) -> bool:
        return self.kind == "body"


def classical_ratio(text: str) -> float:
    cjk = len(_CJK.findall(text)) or 1
    return len(_CLASSICAL.findall(text)) / cjk


def modern_ratio(text: str) -> float:
    cjk = len(_CJK.findall(text)) or 1
    return len(_MODERN.findall(text)) / cjk


def detect_register(text: str) -> str:
    """zh（现代汉语）/ zh_classical（文言）/ en（英文及其他拉丁字母语言）。"""
    cjk = len(_CJK.findall(text))
    latin = len(_LATIN.findall(text))
    if latin >= 30 and latin >= 2 * cjk:
        return "en"
    if cjk >= 20 and classical_ratio(text) >= config.CLASSICAL_THRESHOLD and modern_ratio(text) <= config.MODERN_MAX_RATIO:
        return "zh_classical"
    return "zh"


def target_chars(register: str) -> int:
    # 英文 1 个 token 约 4 个字母，中文约 1.4 个字；英文段落放长一些，检测才稳定（Turnitin 要求至少 300 词）
    return config.SEGMENT_TARGET_CHARS_EN if register == "en" else config.SEGMENT_TARGET_CHARS


def min_chars(register: str) -> int:
    return config.SEGMENT_MIN_CHARS_EN if register == "en" else config.SEGMENT_MIN_CHARS


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

    def flush(kind_override=None):
        nonlocal buf
        if not buf.strip():
            buf = ""
            return
        reg = detect_register(buf)
        tgt = target_chars(reg)
        pieces = _split_long(buf, tgt, reg) if len(buf) > tgt * 1.5 else [buf]
        off = buf_start
        for p in pieces:
            segments.append(Segment(len(segments), p, off, kind_override or "body", [], detect_register(p)))
            off += len(p)
        buf = ""

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        line_start = pos
        pos += len(line)
        if exclude_references and _REF_HEAD.match(stripped):
            flush()
            in_refs = True
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
            if len(buf) >= min_chars(detect_register(buf[:600])):
                flush()
            continue
        # 文体切换（如现代文里插入一段文言引文、中文里夹一段英文）时另起一段
        if buf.strip() and len(stripped) >= 30 and len(buf.strip()) >= 30 \
                and detect_register(stripped) != detect_register(buf[-600:]):
            flush()
        if not buf:
            buf_start = line_start
        buf += line
        if len(buf) >= target_chars(detect_register(buf[:600])):
            flush()
    flush("reference" if in_refs else None)
    segments = [s for s in segments if s.text.strip()]
    for i, s in enumerate(segments):
        s.index = i
        s.text = s.text.strip()
    if flag_quotations:
        _flag_quotations(segments)
    return segments


def _flag_quotations(segments):
    """标出"以引文为主"的段落（不计入 AI 率）。按整篇文稿判断，避免把作者自己的文字误当引文：
    - 现代汉语论文里夹的大段文言 → 视为古籍引文；但如果全文以文言为主（如文言小说、仿古文），文言就是正文。
    - 引号内文字占一半以上的段落 → 视为引文；但如果这类段落超过正文的 30%，或全文以文言为主，多半是小说/对话体，照常计入。"""
    body = [s for s in segments if s.kind == "body"]
    total = sum(len(s.text) for s in body) or 1
    cjk_body = [s for s in body if s.register != "en"]
    cjk_total = sum(len(s.text) for s in cjk_body) or 1
    classical_chars = sum(len(s.text) for s in cjk_body if s.register == "zh_classical")
    classical_doc = classical_chars >= 0.5 * cjk_total
    quoted = [(s, quotation_ratio(s.text)) for s in cjk_body]
    quote_heavy = [s for s, q in quoted if q >= 0.5]
    dialogue_doc = classical_doc or sum(len(s.text) for s in quote_heavy) >= 0.3 * total
    for s, q in quoted:
        if q >= 0.5 and not dialogue_doc:
            s.kind, s.notes = "quotation", [f"引号内文字约占 {q:.0%}"]
        elif s.register == "zh_classical" and not classical_doc and len(s.text) >= 40:
            s.kind, s.notes = "quotation", [f"文言段落（文言虚词 {classical_ratio(s.text):.0%}），疑为古籍引文"]
