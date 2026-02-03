# -*- coding: utf-8 -*-
"""
五大环节 Gradio 应用组装入口。根据 REFACTOR_PLAN 将 Tab1～Tab5 挂到同一 Blocks 下。
"""
import os
import sys
import gradio as gr

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from . import tab1_story, tab2_characters, tab3_refs_shots, tab4_voice_video, tab5_export


def build_app(t2imodel=None):
    """
    组装完整五 Tab 应用。t2imodel 为文生图模型实例，可为 None（tab3 内按 ctx 参数调用 get_t2i_model）。
    返回 gr.Blocks 实例，调用方负责 .queue().launch(...)。
    """
    demo = gr.Blocks(title="安全管理部-视觉生成技术")
    with demo:
        state_full_story = gr.State(value="")
        state_segments = gr.State(value=[])
        state_global_characters = gr.State(value=[])
        state_global_scenes = gr.State(value=[])
        state_storyboards = gr.State(value=[])
        state_shot_images = gr.State(value=[])
        state_shot_audios = gr.State(value=[])
        state_shot_videos = gr.State(value=[])

        ctx = {
            "state_full_story": state_full_story,
            "state_segments": state_segments,
            "state_global_characters": state_global_characters,
            "state_global_scenes": state_global_scenes,
            "state_storyboards": state_storyboards,
            "state_shot_images": state_shot_images,
            "state_shot_audios": state_shot_audios,
            "state_shot_videos": state_shot_videos,
            "t2imodel": t2imodel,
        }

        gr.Markdown(
            "**流程**：Tab1 故事与剧本 → Tab2 角色与场景 → Tab3 角色/场景图与分镜画面 → Tab4 配音与分镜视频 → Tab5 成片导出。各 Tab 可单独「本页测试」。"
        )
        tab1_story.build(ctx)
        tab2_characters.build(ctx)
        tab3_refs_shots.build(ctx)
        tab4_voice_video.build(ctx)
        tab5_export.build(ctx)

    return demo
