# -*- coding: utf-8 -*-
"""
五大生成环节 UI 包。
入口 build_app(t2imodel) 返回组装好的 Gradio Blocks。
各 tab 支持 build(demo, ctx) 与 run_standalone() 单独运行调试。
"""
from .app import build_app
from . import shared

__all__ = ["build_app", "shared"]
