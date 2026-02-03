# -*- coding: utf-8 -*-
"""
环节1：故事与剧本。主题/梗概扩写、小说解析、按段摘要、按段标注角色场景。
"""
import os
import sys
import gradio as gr

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from llm import (
    _generate_response,
    expand_story_from_outline,
    summarize_segment,
    get_segment_characters_scenes,
)
from tool.novel_parser import parse_novel_file, segments_to_full_story
try:
    from . import shared
except ImportError:
    import importlib.util
    _spec = importlib.util.spec_from_file_location("shared", os.path.join(os.path.dirname(__file__), "shared.py"))
    shared = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(shared)

try:
    from mora.generate_image import (
        T2I_BACKEND_OPEN_SOURCE,
        T2I_BACKEND_COMMERCIAL,
        T2I_OPEN_SOURCE_MODEL_IDS,
        T2I_COMMERCIAL_MODEL_IDS,
    )
except Exception:
    T2I_BACKEND_OPEN_SOURCE = "open_source"
    T2I_BACKEND_COMMERCIAL = "commercial"
    T2I_OPEN_SOURCE_MODEL_IDS = ["sdxl_lightning", "sd15"]
    T2I_COMMERCIAL_MODEL_IDS = []


def _segments_from_story(story_text_val):
    if not story_text_val or not story_text_val.strip():
        return []
    parts = [p.strip() for p in story_text_val.strip().split("\n\n") if p.strip()]
    return [{"segment_id": i, "chapter_title": None, "content": p, "summary": ""} for i, p in enumerate(parts)]


def llm_story(prompt, modeltype, input_mode):
    if input_mode == "主题/梗概" and prompt and len(prompt.strip()) < 250:
        return expand_story_from_outline(prompt.strip(), modeltype, None)
    return _generate_response(prompt, modeltype)


def llm_story_and_sync(prompt, modeltype, input_mode):
    story = llm_story(prompt, modeltype, input_mode)
    segs = _segments_from_story(story)
    return story, story, segs


def on_parse_novel(novel_file_obj):
    if not novel_file_obj:
        return "", "", [], "请先上传 txt 文件"
    if isinstance(novel_file_obj, (list, tuple)) and novel_file_obj:
        novel_file_obj = novel_file_obj[0]
    path = getattr(novel_file_obj, "name", None) or str(novel_file_obj)
    if not path or not os.path.isfile(path):
        return "", "", [], "请先上传有效的 txt 文件"
    segs = parse_novel_file(path)
    full = segments_to_full_story(segs)
    lines = []
    for s in segs:
        tit = s.get("chapter_title") or f"段{s['segment_id']+1}"
        lines.append(f"[{tit}] {s.get('summary', '(未提炼)')}")
    return full, full, segs, "\n".join(lines)


def on_summarize_all(segs, modeltype):
    if not segs:
        return [], "当前无段落数据，请先上传并解析小说。"
    out = []
    lines = []
    for s in segs:
        c = s.get("content", "")
        summ = summarize_segment(c, modeltype, None) if c else ""
        snew = dict(s)
        snew["summary"] = summ
        out.append(snew)
        tit = snew.get("chapter_title") or f"段{snew['segment_id']+1}"
        lines.append(f"[{tit}] {summ or '(未提炼)'}")
    return out, "\n".join(lines)


def on_label_segment_characters_scenes(segs, chars, scenes, modeltype):
    if not segs:
        return [], "无段落数据"
    cnames = [c["name"] for c in (chars or [])]
    snames = [s["name"] for s in (scenes or [])]
    out = []
    for s in segs:
        s2 = dict(s)
        cn, sc = get_segment_characters_scenes(s.get("content", ""), cnames, snames, modeltype, None)
        s2["character_names"] = cn
        s2["scene_names"] = sc
        out.append(s2)
    return out, "已标注 %d 段" % len(out)


def test_tab1_story(llm_modeltype):
    outline = "两只小青蛙，一只坐井观天，一只勇敢探索外面的世界。"
    story = expand_story_from_outline(outline, llm_modeltype or "moonshot", None)
    segs = _segments_from_story(story)
    return story, story, segs


def build(ctx):
    """在 ctx 所在 Blocks 下挂载「1、故事与剧本」Tab。需在 with demo 内调用。"""
    with gr.Tab("1、故事与剧本"):
        gr.Markdown("**故事与剧本**：主题/梗概扩写、小说解析、按段摘要。")
        with gr.Tab("故事创作"):
            input_mode = gr.Radio(
                choices=["主题/梗概", "长篇小说"],
                value="主题/梗概",
                label="输入模式",
            )
            with gr.Row():
                story_prompt = gr.Textbox(
                    value=shared.Default_story_prompt,
                    label="故事prompt（主题/梗概时使用）",
                    scale=3,
                    max_lines=3,
                )
                generate_story_text = gr.Button("生成故事设定与描述", variant="primary")
            story_text = gr.Textbox(label="总体故事text", max_lines=8)
            gr.Markdown("---\n**长篇小说**：上传 txt 后点击「解析为章/段」，再可「一键提炼各段」生成摘要；全文将用于角色/场景抽取。")
            with gr.Row():
                novel_file = gr.File(label="上传小说 (txt)", file_types=[".txt"])
                btn_parse_novel = gr.Button("解析为章/段", variant="secondary")
            segment_display = gr.Textbox(label="各段摘要（解析或提炼后更新）", lines=6, max_lines=12)
            btn_summarize_all = gr.Button("一键提炼各段", variant="secondary")
            gr.Markdown("**阶段2**：在「角色场景创作」中生成角色场景（详细版）后，可对当前段落标注涉及的角色/场景：")
            btn_label_segment_cs = gr.Button("按段标注角色场景", variant="secondary")
            segment_status = gr.Textbox(label="按段标注状态", value="", interactive=False)
            with gr.Row():
                btn_test_tab1 = gr.Button("本页测试", variant="secondary")
        with gr.Row():
            with gr.Column(scale=1):
                llm_modeltype = gr.Dropdown(
                    choices=["moonshot", "gpt3.5-turbo", "glm"],
                    value="moonshot",
                    label="语言模型",
                )
            with gr.Column(scale=9):
                with gr.Accordion("图像生成参数设置", open=False):
                    with gr.Row():
                        t2i_backend = gr.Dropdown(
                            choices=[T2I_BACKEND_OPEN_SOURCE, T2I_BACKEND_COMMERCIAL],
                            value=T2I_BACKEND_OPEN_SOURCE,
                            label="文生图后端（商用/开源）",
                        )
                        t2i_model_id = gr.Dropdown(
                            choices=T2I_OPEN_SOURCE_MODEL_IDS + T2I_COMMERCIAL_MODEL_IDS,
                            value="sdxl_lightning",
                            label="文生图模型",
                        )
                        seed = gr.Slider(0, 10000000000000, 0, label="seed", step=1)
                        guidance = gr.Slider(0, 10, 1, label="guidance", step=0.5)
                        width = gr.Slider(480, 1920, 1024, label="width", step=24)
                        height = gr.Slider(480, 1920, 1024, label="height", step=24)
                        num_inference_steps = gr.Slider(1, 8, 4, label="infer steps", step=1)
                        num_img_per_prompt = gr.Slider(1, 4, 3, label="imgnum/prompt", step=1)

        state_full_story = ctx["state_full_story"]
        state_segments = ctx["state_segments"]

        generate_story_text.click(
            llm_story_and_sync,
            inputs=[story_prompt, llm_modeltype, input_mode],
            outputs=[story_text, state_full_story, state_segments],
        )
        btn_test_tab1.click(
            test_tab1_story,
            inputs=[llm_modeltype],
            outputs=[story_text, state_full_story, state_segments],
        )
        btn_parse_novel.click(
            on_parse_novel,
            inputs=[novel_file],
            outputs=[story_text, state_full_story, state_segments, segment_display],
        )
        btn_summarize_all.click(
            on_summarize_all,
            inputs=[state_segments, llm_modeltype],
            outputs=[state_segments, segment_display],
        )
        btn_label_segment_cs.click(
            on_label_segment_characters_scenes,
            inputs=[state_segments, ctx["state_global_characters"], ctx["state_global_scenes"], llm_modeltype],
            outputs=[state_segments, segment_status],
        )

    ctx["story_text"] = story_text
    ctx["llm_modeltype"] = llm_modeltype
    ctx["t2i_backend"] = t2i_backend
    ctx["t2i_model_id"] = t2i_model_id
    ctx["seed"] = seed
    ctx["guidance"] = guidance
    ctx["width"] = width
    ctx["height"] = height
    ctx["num_inference_steps"] = num_inference_steps
    ctx["num_img_per_prompt"] = num_img_per_prompt


def run_standalone(server_name="0.0.0.0", server_port=8082):
    """仅启动本环节页面，便于单独调试。"""
    demo = gr.Blocks(title="1、故事与剧本（独立调试）")
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
        }
        build(ctx)
        gr.Markdown("当前为 **Tab1 独立调试**，仅包含故事与剧本环节。")
    demo.queue().launch(server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    run_standalone()
