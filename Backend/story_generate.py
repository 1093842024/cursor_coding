# -*- coding: utf-8 -*-
"""
故事可视化视频生产线 — 入口。五大生成环节已拆分到 story_ui 各模块，此处仅做组装与启动。
运行时应以 Backend 为工作目录：cd Backend && python story_generate.py
"""
import os
import sys

# 保证 Backend 在 path，便于 story_ui 内从 tool/llm 等导入
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

os.environ.setdefault("no_proxy", "localhost,0.0.0.0,:8082")

from story_ui.app import build_app


def generate_image_gr_demo(t2imodel):
    """返回由 story_ui 组装的五 Tab Gradio 应用。"""
    return build_app(t2imodel)


if __name__ == "__main__":
    from mora.generate_image import get_t2i_model, sd15_model

    try:
        t2imodel = get_t2i_model(
            backend=os.environ.get("T2I_BACKEND"),
            model_id=os.environ.get("T2I_MODEL", "ipadapter"),
        )
    except Exception as e:
        print("t2i error", e)
        t2imodel = sd15_model()

    generate_image_gr_demo(t2imodel).queue().launch(
        max_threads=15,
        show_api=True,
        share=False,
        server_name="0.0.0.0",
        server_port=8082,
    )
