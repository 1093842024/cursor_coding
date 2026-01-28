# -*- coding: utf-8 -*-
"""
阶段4：Nano Banana Pro / 火山引擎等「单参考图 + 多 prompt -> 9 宫格」的 API 封装。
配置通过环境变量：NANO_BANANA_API_KEY, NANO_BANANA_BASE_URL 等。
当前为占位实现，返回空列表；接入时在此实现 request_9grid。
"""
from __future__ import annotations

import os
from typing import List, Any, Optional


def request_9grid(
    ref_image_path: str,
    subject_type: str,
    subject_desc: str,
    options: Optional[dict] = None,
) -> List[Any]:
    """
    请求 API 生成 9 张与 ref_image_path 一致的多视角/多景别图。
    - subject_type: "character" | "scene"
    - subject_desc: 文本描述
    - options: style, seed, width, height 等
    返回 List[PIL.Image] 或空列表。
    """
    # 占位：未接入真实 API 时直接返回空，由 consistency_9grid 回退到 local
    _ = ref_image_path, subject_type, subject_desc, options
    if os.environ.get("NANO_BANANA_API_KEY"):
        # TODO: 发 HTTP 请求，上传 ref_image_path，按 9 种景别/视角请求 9 次或一次 9 宫格
        pass
    return []
