"""服务端文档解析（供 API 直接上传文件使用；网页端在浏览器里解析，不走这里）。"""
from __future__ import annotations

import io
import re
import zipfile
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _docx_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    paras = []
    for p in root.iter(f"{W}p"):
        parts = []
        for node in p.iter():
            if node.tag == f"{W}t" and node.text:
                parts.append(node.text)
            elif node.tag == f"{W}tab":
                parts.append("\t")
            elif node.tag in (f"{W}br", f"{W}cr"):
                parts.append("\n")
        # 跳过修订中被删除的文字（w:delText 不是 w:t，天然不会被收集）
        paras.append("".join(parts))
    return "\n".join(paras)


def _pdf_text(data: bytes) -> str:
    from pdfminer.high_level import extract_text
    return extract_text(io.BytesIO(data))


def _txt(data: bytes) -> str:
    for enc in ("utf-8-sig", "gb18030", "utf-16"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def extract(filename: str, data: bytes) -> str:
    name = (filename or "").lower()
    if name.endswith(".docx"):
        text = _docx_text(data)
    elif name.endswith(".pdf"):
        text = _pdf_text(data)
    elif name.endswith((".txt", ".md")):
        text = _txt(data)
    else:
        raise ValueError("只支持 .docx / .pdf / .txt / .md 文件")
    # PDF 提取常见的"汉字之间多余空格 / 行内断行"做轻度清理
    text = re.sub(r"(?<=[一-鿿]) (?=[一-鿿])", "", text)
    return text.replace("\r\n", "\n").replace("\x0c", "\n")
