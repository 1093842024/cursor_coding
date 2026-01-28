# -*- coding: utf-8 -*-
"""
阶段1：长篇小说按章/按段解析。
将上传的 txt 切分为 segments，每项含 segment_id、chapter_title、content、summary。
"""
import os
import re
from typing import List, Dict, Any, Optional, Tuple


# 常见章节标题正则（支持中文与英文）
CHAPTER_PATTERNS = [
    re.compile(r"^第[一二三四五六七八九十百千\d]+章\s*[^\n]*$", re.MULTILINE),
    re.compile(r"^Chapter\s+[\dIVXLCDMivxlcdm]+\s*[^\n]*$", re.MULTILINE),
    re.compile(r"^【[^】]+】\s*$", re.MULTILINE),
    re.compile(r"^第[一二三四五六七八九十百千\d]+节\s*[^\n]*$", re.MULTILINE),
    re.compile(r"^[\d]+[\.、．]\s*[^\n]+$", re.MULTILINE),  # 1. xxx 或 1、xxx
]


def _detect_encoding(filepath: str) -> str:
    """尝试检测文件编码。"""
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb2312", "big5"):
        try:
            with open(filepath, "r", encoding=enc) as f:
                f.read()
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return "utf-8"


def split_by_chapter(content: str) -> List[Tuple[Optional[str], str]]:
    """
    按章节标题切分正文。返回 [(chapter_title, segment_content), ...]。
    chapter_title 为 None 表示无章节标题或为首段。
    """
    if not content or not content.strip():
        return []

    # 找到所有匹配的章节标题及其位置
    matches = []
    for pat in CHAPTER_PATTERNS:
        for m in pat.finditer(content):
            start = m.start()
            title = m.group(0).strip()
            matches.append((start, title))
    matches.sort(key=lambda x: x[0])

    # 去重、合并重叠（同一位置只保留一个）
    seen = set()
    unique_matches = []
    for start, title in matches:
        if start in seen:
            continue
        seen.add(start)
        unique_matches.append((start, title))

    if not unique_matches:
        # 无章节：整体作一段，或按双换行粗分
        parts = re.split(r"\n\s*\n", content.strip())
        return [(None, p.strip()) for p in parts if p.strip()]

    segments = []
    for i, (start, title) in enumerate(unique_matches):
        end = unique_matches[i + 1][0] if i + 1 < len(unique_matches) else len(content)
        block = content[start:end]
        # 去掉首行章节标题，只保留正文
        first_line = block.split("\n")[0].strip()
        if first_line == title or first_line.startswith(title[:min(10, len(title))]):
            block = "\n".join(block.split("\n")[1:])
        segments.append((title, block.strip()))
    return segments


def parse_novel_file(
    filepath: str,
    max_segment_chars: Optional[int] = 2000,
    fallback_split_by_paragraphs: bool = True,
) -> List[Dict[str, Any]]:
    """
    解析小说文件为 segments 列表。
    每项为 {"segment_id": int, "chapter_title": str|None, "content": str, "summary": str}。
    """
    if not os.path.isfile(filepath):
        return []

    enc = _detect_encoding(filepath)
    with open(filepath, "r", encoding=enc) as f:
        content = f.read()

    raw = split_by_chapter(content)
    if not raw:
        return []

    segments = []
    seg_id = 0
    for chapter_title, block in raw:
        if not block:
            continue
        if max_segment_chars and len(block) > max_segment_chars and fallback_split_by_paragraphs:
            # 按段落或按长度再切
            paras = re.split(r"\n\s*\n", block)
            buf = []
            buf_len = 0
            for p in paras:
                p = p.strip()
                if not p:
                    continue
                if buf_len + len(p) <= max_segment_chars or not buf:
                    buf.append(p)
                    buf_len += len(p)
                else:
                    segments.append({
                        "segment_id": seg_id,
                        "chapter_title": chapter_title if seg_id == 0 and chapter_title else None,
                        "content": "\n\n".join(buf),
                        "summary": "",
                    })
                    seg_id += 1
                    buf = [p]
                    buf_len = len(p)
                    chapter_title = None  # 仅第一段保留章节名
            if buf:
                segments.append({
                    "segment_id": seg_id,
                    "chapter_title": chapter_title,
                    "content": "\n\n".join(buf),
                    "summary": "",
                })
                seg_id += 1
        else:
            segments.append({
                "segment_id": seg_id,
                "chapter_title": chapter_title,
                "content": block,
                "summary": "",
            })
            seg_id += 1

    return segments


def segments_to_full_story(segments: List[Dict[str, Any]]) -> str:
    """将 segments 的 content 拼接为全文，用于全局角色/场景抽取。"""
    return "\n\n".join(s.get("content", "") for s in segments if s.get("content"))
