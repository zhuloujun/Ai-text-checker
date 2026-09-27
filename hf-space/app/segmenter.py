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
# 文言常用虚词（用于粗略识别以古籍引文为主的段落）
_CLASSICAL = re.compile(r"[之乎者也矣焉哉曰兮盖乃遂耶欤焉]")
_QUOTED = re.compile(r"“[^”]{4,}”|「[^」]{4,}」|『[^』]{4,}』")


@dataclass
class Segment:
    index: int
    text: str
    start: int                       # 在原文中的字符位置
    kind: str = "body"               # body | reference | quotation
    notes: list = field(default_factory=list)

    @property
    def counted(self) -> bool:
        return self.kind == "body"


def _split_long(par: str, target: int):
    sents = [s for s in _SENT_END.split(par) if s.strip()]
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
        if out and len(cur) < config.SEGMENT_MIN_CHARS:
            out[-1] += cur
        else:
            out.append(cur)
    return out


def quotation_ratio(text: str) -> float:
    cjk = len(_CJK.findall(text)) or 1
    quoted = sum(len(m.group(0)) for m in _QUOTED.finditer(text))
    return min(1.0, quoted / cjk)


def classical_ratio(text: str) -> float:
    cjk = len(_CJK.findall(text)) or 1
    return len(_CLASSICAL.findall(text)) / cjk


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
        pieces = _split_long(buf, target) if len(buf) > target * 1.5 else [buf]
        off = buf_start
        for p in pieces:
            kind = kind_override or "body"
            notes = []
            if kind == "body" and flag_quotations:
                q = quotation_ratio(p)
                c = classical_ratio(p)
                if q >= 0.5:
                    kind, notes = "quotation", [f"引号内文字约占 {q:.0%}"]
                elif c >= config.CLASSICAL_THRESHOLD and len(p) >= 40:
                    kind, notes = "quotation", [f"文言虚词密度 {c:.0%}，疑为古籍引文"]
            segments.append(Segment(len(segments), p, off, kind, notes))
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
            if len(buf) >= config.SEGMENT_MIN_CHARS:
                flush()
            continue
        if not buf:
            buf_start = line_start
        buf += line
        if len(buf) >= target:
            flush()
    flush("reference" if in_refs else None)
    for i, s in enumerate(segments):
        s.index = i
        s.text = s.text.strip()
    return [s for s in segments if s.text]
