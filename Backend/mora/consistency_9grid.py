# -*- coding: utf-8 -*-
"""
阶段4：角色/场景一致性 9 宫格生成。
接口：单参考图 -> 9 张多视角/多景别的一致性图。
后端可通过环境变量 USE_CONSISTENCY_BACKEND=nano_banana|local 选择。
"""
from __future__ import annotations

import os
from typing import List, Optional, Any, Callable

# 9 宫格中每格对应的景别/视角描述（英文），用于本地生成时的 prompt 后缀
GRID_9_PROMPT_SUFFIXES = [
    "close-up face, portrait",
    "medium shot, upper body",
    "full body, standing",
    "profile view, side angle",
    "from behind, back view",
    "wide shot, in environment",
    "slightly low angle",
    "slightly high angle",
    "centered, front view",
]


def generate_9grid_from_reference(
    ref_image_path: str,
    subject_type: str,
    subject_desc: str,
    options: Optional[dict] = None,
    backend: Optional[str] = None,
    local_generate_fn: Optional[Callable[[str, str], Any]] = None,
) -> List[Any]:
    """
    从一张参考图生成 9 宫格（3x3）一致性图。
    - ref_image_path: 参考图路径（角色或场景的首图）
    - subject_type: "character" | "scene"
    - subject_desc: 简短描述，用于构造各格 prompt
    - options: 可含 style, size, seed 等
    - backend: "nano_banana" | "local" | None(从环境变量读)
    - local_generate_fn: 仅 local 时使用，签名为 (prompt, negative_prompt) -> PIL.Image
    返回 9 张图的列表，顺序与 GRID_9_PROMPT_SUFFIXES 一致。
    """
    options = options or {}
    backend = backend or os.environ.get("USE_CONSISTENCY_BACKEND", "local")
    if backend == "nano_banana":
        return _generate_9grid_nano_banana(ref_image_path, subject_type, subject_desc, options)
    return _generate_9grid_local(
        ref_image_path, subject_type, subject_desc, options, local_generate_fn
    )


def _generate_9grid_local(
    ref_image_path: str,
    subject_type: str,
    subject_desc: str,
    options: dict,
    local_generate_fn: Optional[Callable] = None,
) -> List[Any]:
    """本地占位：按 9 种景别/视角各生成一张。若未传入 local_generate_fn 则返回空列表。"""
    if not local_generate_fn:
        return []
    base = subject_desc.strip()
    out = []
    neg = options.get("negative_prompt", "bad hands, bad face, blurry, low quality, watermark")
    for suf in GRID_9_PROMPT_SUFFIXES:
        prompt = f"{base}, {suf}"
        img = local_generate_fn(prompt, neg)
        if img is not None:
            out.append(img)
    return out


def _generate_9grid_nano_banana(
    ref_image_path: str,
    subject_type: str,
    subject_desc: str,
    options: dict,
) -> List[Any]:
    """调用 Nano Banana Pro / 火山等 API。若未配置或不可用则返回空列表。"""
    try:
        from mora.nano_banana_client import request_9grid
        return request_9grid(ref_image_path, subject_type, subject_desc, options)
    except Exception:
        return []
