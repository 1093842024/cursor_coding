# -*- coding: utf-8 -*-
"""
环节2：角色与场景。角色/场景抽取、按段标注、电影化分镜；fenjin_parse 供 Tab4 使用。
"""
import os
import sys
import gradio as gr

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from llm import (
    _generate_response,
    extract_characters_scenes_detailed,
    generate_cinematic_storyboard_for_segment,
)
try:
    from . import shared
except ImportError:
    import importlib.util
    _spec = importlib.util.spec_from_file_location("shared", os.path.join(os.path.dirname(__file__), "shared.py"))
    shared = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(shared)

Max_fenjin_num = shared.Max_fenjin_num


def _parse_person_scene_detailed(text):
    chars, scenes = [], []
    block = None
    for line in text.split("\n"):
        line = line.strip()
        if "[Characters]" in line or line == "[Characters]":
            block = "char"
            continue
        if "[Scenes]" in line or line == "[Scenes]":
            block = "scene"
            continue
        parts = [p.strip() for p in line.split(";") if p.strip()]
        if len(parts) < 2:
            continue
        name = parts[0].split(":")[-1].strip() if ":" in parts[0] else parts[0]
        desc = parts[1].split(":")[-1].strip() if ":" in parts[1] else parts[1]
        detailed = parts[2].split(":")[-1].strip() if len(parts) > 2 and ":" in parts[2] else desc
        if block == "char" and ("角色" in line or "外形" in str(parts[0])):
            chars.append({"name": name, "desc": desc, "detailed": detailed})
        elif block == "scene" and ("场景" in line or "画面" in str(parts[0])):
            scenes.append({"name": name, "desc": desc, "detailed": detailed})
    return chars, scenes


def llm_person_scene(prompt, story, modeltype):
    newprompt = prompt.replace("{story}", story)
    return _generate_response(newprompt, modeltype)


def llm_person_scene_detailed(story, modeltype):
    raw = extract_characters_scenes_detailed(story or "", modeltype, None)
    chars, scenes = _parse_person_scene_detailed(raw)
    lines = ["[Characters]"]
    for i, c in enumerate(chars):
        lines.append(f"角色{i}名称:{c['name']};外形描述:{c['desc']}")
    lines.append("[Scenes]")
    for j, s in enumerate(scenes):
        lines.append(f"场景{j}名称:{s['name']};画面描述:{s['desc']}")
    return "\n".join(lines), chars, scenes


def llm_fenjing(prompt, story, person_scene, modeltype):
    newprompt = prompt.replace("{person_scene}", person_scene).replace("{story}", story)
    return _generate_response(newprompt, modeltype)


def _parse_cinematic_shot_line(line):
    d = {"narration": "", "characters": "", "scene": "", "prompt": "", "shot_type": "", "scale": ""}
    if "->" not in line:
        return d
    _, rest = line.split("->", 1)
    for part in rest.split(";"):
        part = part.strip()
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        v = v.strip().rstrip(".")
        k = k.strip()
        if "旁白" in k or k == "旁白内容":
            d["narration"] = v
        elif "角色" in k or k == "角色":
            d["characters"] = v
        elif "场景" in k or k == "场景":
            d["scene"] = v
        elif "画面prompt" in k or k == "画面prompt":
            d["prompt"] = v
        elif "镜头类型" in k or k == "镜头类型":
            d["shot_type"] = v
        elif "景别" in k or k == "景别":
            d["scale"] = v
    return d


def on_generate_cinematic_storyboards(segs, modeltype):
    if not segs:
        return "", [], "无段落数据"
    all_boards = []
    legacy_lines = []
    for seg_idx, s in enumerate(segs):
        content = s.get("content", "") or s.get("summary", "")
        cnames = s.get("character_names") or []
        snames = s.get("scene_names") or []
        raw = generate_cinematic_storyboard_for_segment(content, cnames, snames, modeltype, None)
        shots = []
        for line in (raw or "").split("\n"):
            line = line.strip()
            if not line or "->" not in line:
                continue
            d = _parse_cinematic_shot_line(line)
            shots.append(d)
            legacy_lines.append(
                "分镜%i画面%i->旁白内容:%s;角色:%s;场景:%s;画面prompt:%s."
                % (seg_idx + 1, len(shots), d["narration"], d["characters"], d["scene"], d["prompt"])
            )
        all_boards.append(shots)
    return "\n".join(legacy_lines), all_boards, "已生成 %d 段分镜" % len(all_boards)


def person_scene_text_parse(person_scene):
    persons, scenes = [], []
    for line in (person_scene or "").split("\n"):
        if "角色" in line and "外形" in line:
            parts = line.split(";")
            name = parts[0].split(":")[-1].strip() if parts else ""
            desc = parts[1].split(":")[-1].strip() if len(parts) > 1 else ""
            if name or desc:
                persons.append(name + ":" + desc)
        if "场景" in line and "画面" in line:
            parts = line.split(";")
            name = parts[0].split(":")[-1].strip() if parts else ""
            desc = parts[1].split(":")[-1].strip() if len(parts) > 1 else ""
            if name or desc:
                scenes.append(name + ":" + desc)
    return (
        gr.update(choices=persons, value=persons[0] if persons else None),
        gr.update(choices=scenes, value=scenes[0] if scenes else None),
    )


def fenjin_parse(content_text):
    """解析分镜文本，得到各镜的旁白、角色、场景、画面 prompt 等，供 Tab4 绑定。"""
    fenjing_pangbai = []
    fenjin_person = []
    fenjing_scene = []
    fenjing_prompt = []
    fenjin_contents = {}
    for line in (content_text or "").split("\n"):
        if "->" not in line:
            continue
        startcontent, infocontent = line.split("->", 1)
        if "分镜" in startcontent and "画面" in startcontent:
            try:
                i = int(startcontent.split("分镜")[-1].split("画面")[0])
                fenjin_contents.setdefault(i, []).append(infocontent)
            except (ValueError, IndexError):
                pass
    num = len(fenjin_contents)
    for i in range(Max_fenjin_num):
        if i < num:
            contents = fenjin_contents.get(i + 1, [])
            pangbai_all, person_all, scene_all, prompt_all = "", [], [], []
            for cnt in contents:
                parts = cnt.split(";")
                if len(parts) >= 4:
                    pangbai_all += (parts[0].split(":")[-1] if ":" in parts[0] else parts[0]) + "."
                    person_all.append(parts[1].split(":")[-1] if ":" in parts[1] else parts[1])
                    scene_all.append(parts[2].split(":")[-1] if ":" in parts[2] else parts[2])
                    prompt_all.append(parts[3].split(":")[-1] if ":" in parts[3] else parts[3])
            fenjing_pangbai.append(gr.update(value=pangbai_all, visible=True))
            fenjin_person.append(gr.update(choices=person_all, value=person_all[0] if person_all else None, visible=True))
            fenjing_scene.append(gr.update(choices=scene_all, value=scene_all[0] if scene_all else None, visible=True))
            fenjing_prompt.append(gr.update(choices=prompt_all, value=prompt_all[0] if prompt_all else None, visible=True))
        else:
            fenjing_pangbai.append(gr.update(value=None, visible=False))
            fenjin_person.append(gr.update(choices=[], value=None, visible=False))
            fenjing_scene.append(gr.update(choices=[], value=None, visible=False))
            fenjing_prompt.append(gr.update(choices=[], value=None, visible=False))
    return fenjing_pangbai + fenjin_person + fenjing_scene + fenjing_prompt


def test_tab2_characters_scenes(llm_modeltype):
    short = "井底有两只青蛙。小蓝总看天，小绿想出去。小绿跳出井口，看到草地与河流，回来告诉小蓝。最后两只青蛙一起跳出了井。"
    raw = extract_characters_scenes_detailed(short, llm_modeltype or "moonshot", None)
    chars, scenes = _parse_person_scene_detailed(raw)
    lines = ["[Characters]"]
    for i, c in enumerate(chars):
        lines.append("角色%d名称:%s;外形描述:%s" % (i, c.get("name", ""), c.get("desc", "")))
    lines.append("[Scenes]")
    for j, s in enumerate(scenes):
        lines.append("场景%d名称:%s;画面描述:%s" % (j, s.get("name", ""), s.get("desc", "")))
    person_scene_text = "\n".join(lines)
    cnames = [c.get("name") for c in chars if c.get("name")]
    snames = [s.get("name") for s in scenes if s.get("name")]
    raw_sb = generate_cinematic_storyboard_for_segment(short[:150], cnames, snames, llm_modeltype or "moonshot", None)
    boards = []
    legacy = []
    for line in (raw_sb or "").split("\n"):
        line = line.strip()
        if not line or "->" not in line:
            continue
        d = _parse_cinematic_shot_line(line)
        boards.append(d)
        legacy.append("分镜1画面%d->旁白内容:%s;角色:%s;场景:%s;画面prompt:%s." % (len(boards), d["narration"], d["characters"], d["scene"], d["prompt"]))
    return person_scene_text, chars, scenes, "\n".join(legacy), [boards], "本页测试已生成角色/场景与分镜"


def build(ctx):
    """挂载「2、角色与场景」Tab。依赖 ctx 内 story_text、llm_modeltype、state_*。"""
    with gr.Tab("2、角色与场景"):
        gr.Markdown("**角色与场景**：角色/场景抽取、按段标注、电影化分镜。输入默认来自 Tab1 的 story_text / state。")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    person_scene_prompt = gr.Textbox(value=shared.Default_person_scene_prompt, label="角色场景prompt", scale=3, max_lines=3)
                    generate_ps_text = gr.Button("生成角色场景text", variant="primary")
                    generate_ps_detailed_btn = gr.Button("生成角色场景（详细版）", variant="secondary")
                person_scene_text = gr.Textbox(value=shared.DeFault_person_scene_text, label="角色场景text", max_lines=8)
            with gr.Column():
                with gr.Row():
                    content_prompt = gr.Textbox(value=shared.DeFault_fenjin_prompt, label="分镜prompt", scale=3, max_lines=3)
                    generate_content_text = gr.Button("生成分镜描述", variant="primary", scale=1)
                    btn_cinematic_sb = gr.Button("按段生成电影化分镜", variant="secondary")
                content_text = gr.Textbox(value=shared.DeFault_fenjin_text, label="分镜text", max_lines=8)
                cinematic_sb_status = gr.Textbox(label="电影化分镜状态", value="", interactive=False)
                with gr.Row():
                    btn_test_tab2 = gr.Button("本页测试", variant="secondary")

        story_text = ctx["story_text"]
        llm_modeltype = ctx["llm_modeltype"]
        state_segments = ctx["state_segments"]
        state_global_characters = ctx["state_global_characters"]
        state_global_scenes = ctx["state_global_scenes"]
        state_storyboards = ctx["state_storyboards"]

        generate_ps_text.click(
            llm_person_scene,
            inputs=[person_scene_prompt, story_text, llm_modeltype],
            outputs=[person_scene_text],
        )
        generate_ps_detailed_btn.click(
            llm_person_scene_detailed,
            inputs=[story_text, llm_modeltype],
            outputs=[person_scene_text, state_global_characters, state_global_scenes],
        )
        generate_content_text.click(
            llm_fenjing,
            inputs=[content_prompt, story_text, person_scene_text, llm_modeltype],
            outputs=[content_text],
        )
        btn_cinematic_sb.click(
            on_generate_cinematic_storyboards,
            inputs=[state_segments, llm_modeltype],
            outputs=[content_text, state_storyboards, cinematic_sb_status],
        )
        btn_test_tab2.click(
            test_tab2_characters_scenes,
            inputs=[llm_modeltype],
            outputs=[person_scene_text, state_global_characters, state_global_scenes, content_text, state_storyboards, cinematic_sb_status],
        )

    ctx["person_scene_text"] = person_scene_text
    ctx["content_text"] = content_text
    ctx["fenjin_parse"] = fenjin_parse


def run_standalone(server_name="0.0.0.0", server_port=8083):
    demo = gr.Blocks(title="2、角色与场景（独立调试）")
    with demo:
        state_full_story = gr.State(value="")
        state_segments = gr.State(value=[])
        state_global_characters = gr.State(value=[])
        state_global_scenes = gr.State(value=[])
        state_storyboards = gr.State(value=[])
        state_shot_images = gr.State(value=[])
        state_shot_audios = gr.State(value=[])
        state_shot_videos = gr.State(value=[])
        story_text = gr.Textbox(label="总体故事text（占位）", value="井底有两只青蛙。小蓝总看天，小绿想出去。", visible=True)
        llm_modeltype = gr.Dropdown(choices=["moonshot", "gpt3.5-turbo", "glm"], value="moonshot", label="语言模型")
        ctx = {
            "state_full_story": state_full_story,
            "state_segments": state_segments,
            "state_global_characters": state_global_characters,
            "state_global_scenes": state_global_scenes,
            "state_storyboards": state_storyboards,
            "state_shot_images": state_shot_images,
            "state_shot_audios": state_shot_audios,
            "state_shot_videos": state_shot_videos,
            "story_text": story_text,
            "llm_modeltype": llm_modeltype,
        }
        build(ctx)
        gr.Markdown("当前为 **Tab2 独立调试**。")
    demo.queue().launch(server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    run_standalone()
