# -*- coding: utf-8 -*-
"""统一配置加载：从环境变量或 .env 读取。各模块 api_key、base_url、默认 model 由此获取，便于部署与安全审查。"""
import os
from typing import Tuple

# 可选：从 .env 加载（若已安装 python-dotenv）
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.isfile(_env_path):
        load_dotenv(_env_path)
except ImportError:
    pass


def get_llm_config(provider: str) -> Tuple[str, str, str]:
    """返回 (api_key, base_url, model_name)。从环境变量读取，例如 MOONSHOT_API_KEY、MOONSHOT_BASE_URL、MOONSHOT_MODEL。"""
    p = provider.upper().replace("-", "_")
    api_key = (os.environ.get(f"LLM_{p}_API_KEY") or os.environ.get(f"{p}_API_KEY") or "").strip()
    base_url = (os.environ.get(f"LLM_{p}_BASE_URL") or os.environ.get(f"{p}_BASE_URL") or "").strip()
    model_name = (os.environ.get(f"LLM_{p}_MODEL") or os.environ.get(f"{p}_MODEL") or "").strip()
    return api_key, base_url, model_name
