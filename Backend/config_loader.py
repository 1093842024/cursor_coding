# -*- coding: utf-8 -*-
"""统一配置加载：从环境变量或 .env 读取；环境变量缺失时可从 config.toml 补全。各模块 api_key、base_url、默认 model 由此获取。"""
import os
from typing import Any, Dict, Tuple

# 可选：从 .env 加载（若已安装 python-dotenv）
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.isfile(_env_path):
        load_dotenv(_env_path)
except ImportError:
    pass

# 可选：从 config.toml 加载（环境变量缺失时补全）
_config_toml: Dict[str, Any] = {}
_config_path = os.path.join(os.path.dirname(__file__), "config.toml")
if os.path.isfile(_config_path):
    try:
        import tomllib
        with open(_config_path, "rb") as f:
            _config_toml = tomllib.load(f)
    except ImportError:
        try:
            import tomli as tomllib
            with open(_config_path, "rb") as f:
                _config_toml = tomllib.load(f)
        except ImportError:
            pass


def get_llm_config(provider: str) -> Tuple[str, str, str]:
    """返回 (api_key, base_url, model_name)。优先环境变量，缺失时从 config.toml 的 [llm.<provider>] 补全。"""
    p = provider.upper().replace("-", "_")
    pk = provider.lower().replace("-", "_")
    api_key = (os.environ.get(f"LLM_{p}_API_KEY") or os.environ.get(f"{p}_API_KEY") or "").strip()
    base_url = (os.environ.get(f"LLM_{p}_BASE_URL") or os.environ.get(f"{p}_BASE_URL") or "").strip()
    model_name = (os.environ.get(f"LLM_{p}_MODEL") or os.environ.get(f"{p}_MODEL") or "").strip()
    if not api_key or not base_url or not model_name:
        llm = (_config_toml.get("llm") or {})
        section = (llm.get(pk) or llm.get(provider) or {})
        if isinstance(section, dict):
            api_key = api_key or str((section.get("api_key") or "")).strip()
            base_url = base_url or str((section.get("base_url") or "")).strip()
            model_name = model_name or str((section.get("model") or "")).strip()
    return api_key, base_url, model_name


def get_t2i_config(model_id: str) -> Tuple[str, str, str]:
    """返回 (api_key, base_url, model_name)。供商用 T2I 后端使用。优先环境变量，缺失时从 config.toml 的 [t2i]<table> 或 [t2i.<model_id>] 补全。"""
    p = model_id.upper().replace("-", "_")
    mk = model_id.lower().replace("-", "_")
    api_key = (os.environ.get(f"T2I_{p}_API_KEY") or os.environ.get(f"{p}_API_KEY") or "").strip()
    base_url = (os.environ.get(f"T2I_{p}_BASE_URL") or os.environ.get(f"{p}_BASE_URL") or "").strip()
    model_name = (os.environ.get(f"T2I_{p}_MODEL") or os.environ.get(f"{p}_MODEL") or "").strip()
    if not api_key or not base_url or not model_name:
        t2i = (_config_toml.get("t2i") or {})
        section = (t2i.get(mk) or t2i.get(model_id) or {})
        if isinstance(section, dict):
            api_key = api_key or str((section.get("api_key") or "")).strip()
            base_url = base_url or str((section.get("base_url") or "")).strip()
            model_name = model_name or str((section.get("model") or "")).strip()
    return api_key, base_url, model_name
