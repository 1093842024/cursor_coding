# -*- coding: utf-8 -*-
"""
环节3：角色/场景图与分镜画面。设定图、9 宫格、分镜选图；提供 generate_single_fenjin_img 等供 Tab4 使用。
"""
import os
import sys
import random
import time
from PIL import Image
import numpy as np
import gradio as gr

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

try:
    from . import shared
except ImportError:
    import importlib.util
    _spec = importlib.util.spec_from_file_location("shared", os.path.join(os.path.dirname(__file__), "shared.py"))
    shared = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(shared)

from mora.generate_image import get_t2i_model, T2I_BACKEND_OPEN_SOURCE, T2I_OPEN_SOURCE_MODEL_IDS, T2I_COMMERCIAL_MODEL_IDS
from mora.sd_prompt.sdxl_styles import sdxl_style_template, sdxl_styles

try:
    from mora.consistency_9grid import generate_9grid_from_reference
except Exception:
    generate_9grid_from_reference = None

MAX_SEED = shared.MAX_SEED
DEFAULT_NEG_PROMPT = shared.DEFAULT_NEG_PROMPT
Max_fenjin_num = shared.Max_fenjin_num


def _fenjin_img_from_9grid(ip, scene, prompt, grid_index=0):
    base_scene = "set_lib/scene/"
    if scene:
        path = base_scene + scene.replace("/", "_") + "/grid_9/%d.png" % grid_index
        if os.path.isfile(path):
            return [(Image.open(path).convert("RGB"), prompt or "")]
    base_char = "set_lib/character/"
    for name in (ip or "").split(","):
        name = name.strip()
        if not name or name.lower() == "none":
            continue
        path = base_char + name.replace("/", "_") + "/grid_9/%d.png" % grid_index
        if os.path.isfile(path):
            return [(Image.open(path).convert("RGB"), prompt or "")]
    return []


USE_LEGACY_CONSISTENCY = False


def build(ctx):
    """挂载「3、角色/场景图与分镜画面」Tab。ctx 需含 t2imodel、state_global_characters/scenes、person_scene_text、以及 seed/guidance/width/height 等。"""
    t2imodel = ctx.get("t2imodel")

    def _get_model():
        return t2imodel or get_t2i_model(
            backend=ctx.get("t2i_backend") or T2I_BACKEND_OPEN_SOURCE,
            model_id=ctx.get("t2i_model_id") or "sdxl_lightning",
        )

    def generate_no_ref(prompt_neg, prompt_pos, seed, num_inference_steps, guidance, width, height, num_img_per_prompt=1):
        model = _get_model()
        all_images = []
        s = seed
        for _ in range(num_img_per_prompt):
            if s == 0:
                s = random.randint(1, MAX_SEED)
            img = model.generate_face_style(None, None, prompt_pos, prompt_neg, steps=num_inference_steps, seed=s, guidance=guidance, width=width, height=height)
            all_images.append((img, prompt_pos))
            s = random.randint(1, MAX_SEED)
        return all_images

    def generate_adapter_person(prompt_neg, prompt_pos, person_ref, seed, num_inference_steps, guidance, width, height, num_img_per_prompt, scale):
        model = _get_model()
        if person_ref and isinstance(person_ref, list):
            person_ref = [Image.open(pr[0]).convert("RGB") if isinstance(pr, (tuple, list)) else pr for pr in person_ref]
        all_images = []
        s = seed
        for _ in range(num_img_per_prompt):
            if s == 0:
                s = random.randint(1, MAX_SEED)
            img = model.generate_face_style(person_ref, None, prompt_pos, prompt_neg, steps=num_inference_steps, seed=s, guidance=guidance, width=width, height=height, face_scale=scale, bgstyle_scale=scale)
            all_images.append((img, prompt_neg))
            s = random.randint(1, MAX_SEED)
        return all_images

    def generate_adapter_scene(scene_prompt_neg, scene_prompt_pos, style_ref, seed, num_inference_steps, guidance, width, height, num_img_per_prompt, scale):
        model = _get_model()
        if style_ref and isinstance(style_ref, list):
            style_ref = [Image.open(sr[0]).convert("RGB") if isinstance(sr, (tuple, list)) else sr for sr in style_ref]
        all_images = []
        s = seed
        for _ in range(num_img_per_prompt):
            if s == 0:
                s = random.randint(1, MAX_SEED)
            img = model.generate_face_style(None, style_ref, scene_prompt_pos, scene_prompt_neg, steps=num_inference_steps, seed=s, guidance=guidance, width=width, height=height, face_scale=scale, bgstyle_scale=scale)
            all_images.append((img, scene_prompt_pos))
            s = random.randint(1, MAX_SEED)
        return all_images

    def batch_generate_character_scene_refs(chars, scenes, stylename, seed_val, steps, guid, w, h, t2i_backend, t2i_model_id):
        if not chars and not scenes:
            return "无角色/场景数据，请先在「角色场景创作」中生成角色场景（详细版）。"
        model = get_t2i_model(backend=t2i_backend or T2I_BACKEND_OPEN_SOURCE, model_id=t2i_model_id or "sdxl_lightning")
        base_char, base_scene = "set_lib/character/", "set_lib/scene/"
        for d in (base_char, base_scene):
            os.makedirs(d, exist_ok=True)
        report = []
        for c in (chars or []):
            name, desc = c.get("name", ""), c.get("desc", "") or c.get("detailed", "")
            if not name or not desc:
                continue
            try:
                _, pos, _ = sdxl_style_template.get_name_style_prompt(stylename or "No Style", desc)
                s = seed_val if seed_val else random.randint(1, MAX_SEED)
                img = model.generate_face_style(None, None, pos, DEFAULT_NEG_PROMPT, steps=steps, seed=s, guidance=guid, width=w, height=h)
                savedir = base_char + name.replace("/", "_") + "/"
                os.makedirs(savedir, exist_ok=True)
                img.save(savedir + str(time.time()) + ".png")
                report.append("角色: " + name)
            except Exception as e:
                report.append("角色 %s 失败: %s" % (name, str(e)))
        for s in (scenes or []):
            name, desc = s.get("name", ""), s.get("desc", "") or s.get("detailed", "")
            if not name or not desc:
                continue
            try:
                _, pos, _ = sdxl_style_template.get_name_style_prompt(stylename or "No Style", desc)
                s = seed_val if seed_val else random.randint(1, MAX_SEED)
                img = model.generate_face_style(None, None, pos, DEFAULT_NEG_PROMPT, steps=steps, seed=s, guidance=guid, width=w, height=h)
                savedir = base_scene + name.replace("/", "_") + "/"
                os.makedirs(savedir, exist_ok=True)
                img.save(savedir + str(time.time()) + ".png")
                report.append("场景: " + name)
            except Exception as e:
                report.append("场景 %s 失败: %s" % (name, str(e)))
        return "\n".join(report) if report else "未生成任何图片"

    def batch_generate_9grid_for_characters_scenes(chars, scenes, seed_val, steps, guid, w, h, t2i_backend, t2i_model_id):
        if not generate_9grid_from_reference:
            return "未加载 consistency_9grid 模块"
        model = get_t2i_model(backend=t2i_backend or T2I_BACKEND_OPEN_SOURCE, model_id=t2i_model_id or "sdxl_lightning")
        base_char, base_scene = "set_lib/character/", "set_lib/scene/"
        opts = {"negative_prompt": DEFAULT_NEG_PROMPT, "seed": seed_val}

        def local_gen(prompt, neg):
            s = seed_val or random.randint(1, MAX_SEED)
            return model.generate_face_style(None, None, prompt, neg, steps=steps, seed=s, guidance=guid, width=w, height=h)

        report = []
        for c in (chars or []):
            name = c.get("name", "")
            desc = (c.get("desc") or c.get("detailed", ""))
            if not name:
                continue
            dirname = base_char + name.replace("/", "_") + "/"
            if not os.path.isdir(dirname):
                report.append("角色 %s 尚无首图目录，请先一键生成首图" % name)
                continue
            imgs = [os.path.join(dirname, f) for f in os.listdir(dirname) if f.endswith(".png") and "grid" not in f]
            ref = (imgs[0] if imgs else None) or os.path.join(dirname, "ref.png")
            if not imgs and not os.path.isfile(ref):
                report.append("角色 %s 无参考图，跳过 9 宫格" % name)
                continue
            ref = imgs[0] if imgs else ref
            grid_dir = os.path.join(dirname, "grid_9")
            os.makedirs(grid_dir, exist_ok=True)
            try:
                out = generate_9grid_from_reference(ref, "character", desc, opts, backend="local", local_generate_fn=local_gen)
                for i, img in enumerate(out):
                    if i < 9:
                        img.save(os.path.join(grid_dir, "%d.png" % i))
                report.append("角色 %s: 已写 grid_9 (%d 张)" % (name, len(out)))
            except Exception as e:
                report.append("角色 %s 失败: %s" % (name, str(e)))
        for s in (scenes or []):
            name = s.get("name", "")
            desc = (s.get("desc") or s.get("detailed", ""))
            if not name:
                continue
            dirname = base_scene + name.replace("/", "_") + "/"
            if not os.path.isdir(dirname):
                report.append("场景 %s 尚无首图目录" % name)
                continue
            imgs = [os.path.join(dirname, f) for f in os.listdir(dirname) if f.endswith(".png") and "grid" not in f]
            ref = (imgs[0] if imgs else None) or os.path.join(dirname, "ref.png")
            if not imgs and not os.path.isfile(ref):
                report.append("场景 %s 无参考图，跳过" % name)
                continue
            ref = imgs[0] if imgs else ref
            grid_dir = os.path.join(dirname, "grid_9")
            os.makedirs(grid_dir, exist_ok=True)
            try:
                out = generate_9grid_from_reference(ref, "scene", desc, opts, backend="local", local_generate_fn=local_gen)
                for i, img in enumerate(out):
                    if i < 9:
                        img.save(os.path.join(grid_dir, "%d.png" % i))
                report.append("场景 %s: 已写 grid_9 (%d 张)" % (name, len(out)))
            except Exception as e:
                report.append("场景 %s 失败: %s" % (name, str(e)))
        return "\n".join(report) if report else "未处理任何角色/场景"

    def generate_single_fenjin_img(gallery_person, gallery_scene, seed, num_inference_steps, guidance, width, height, num_img_per_prompt, mask_img_select, person_scale, bg_scale, ip, scene, prompt):
        if not prompt or not ip or not scene:
            return None
        if not USE_LEGACY_CONSISTENCY:
            images = _fenjin_img_from_9grid(ip, scene, prompt)
            if images:
                return gr.update(value=images, visible=True)
        return None

    def generate_all_fenjin_imgs(gallery_person, gallery_scene, seed, num_inference_steps, guidance, width, height, num_img_per_prompt, *ip_scene_prompt):
        if seed == 0:
            seed = random.randint(1, MAX_SEED)
        n = len(ip_scene_prompt) // 3
        ips = list(ip_scene_prompt[:n])
        scenes = list(ip_scene_prompt[n : 2 * n])
        prompts = list(ip_scene_prompt[2 * n : 3 * n])
        fenjin_images = [gr.update(value=None, visible=False)] * Max_fenjin_num
        for i, (ip, scene, prompt) in enumerate(zip(ips, scenes, prompts)):
            if prompt and ip and scene and not USE_LEGACY_CONSISTENCY:
                images = _fenjin_img_from_9grid(ip, scene, prompt)
                if images:
                    fenjin_images[i] = gr.update(value=images, visible=True)
            yield fenjin_images

    # ---- UI ----
    with gr.Tab("3、角色/场景图与分镜画面"):
        gr.Markdown("**角色/场景图与分镜画面**：设定图、9 宫格、每镜选图。")
        gr.Markdown("批量生成全部角色/场景首图并写入角色库/场景库；可再为每个角色/场景生成 9 宫格。")
        with gr.Row():
            btn_batch_refs = gr.Button("一键生成全部角色/场景首图", variant="primary")
            btn_test_tab3 = gr.Button("本页测试", variant="secondary")
            batch_refs_report = gr.Textbox(label="批量首图生成结果", lines=4, interactive=False)
        with gr.Row():
            btn_batch_9grid = gr.Button("为全部角色/场景生成 9 宫格", variant="secondary")
            batch_9grid_report = gr.Textbox(label="9 宫格生成结果", lines=3, interactive=False)
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    stylename = gr.Dropdown(choices=sdxl_style_template.get_style_name(), value="No Style", label="prompt风格类型", scale=1)
                    template_type = gr.Dropdown(choices=["cat", "man", "woman", "natural_scene"], value="cat", label="模版主体", scale=1)
                person_name = gr.Dropdown(choices=[], label="角色名称", allow_custom_value=True)
                scene_name = gr.Dropdown(choices=[], label="场景名称", allow_custom_value=True)
            with gr.Column(scale=3):
                style_example_img = gr.Gallery(sdxl_style_template.get_exampleimg_path(), label="风格模板图",  columns=8, height=360)

        def change_template_type(template_type):
            if template_type == "cat":
                return gr.update(choices=sdxl_style_template.get_style_name()), gr.update(value=sdxl_style_template.get_exampleimg_path())
            tpl = sdxl_styles(defaultdir=template_type)
            return gr.update(choices=tpl.get_style_name()), gr.update(value=tpl.get_exampleimg_path())

        def apply_select_stylename(stylename, person_text, scene_text):
            _, neg1, _ = sdxl_style_template.get_name_style_prompt(stylename, (person_text or "").split(":")[-1])
            pos1 = (person_text or "").split(":")[-1]
            _, neg2, _ = sdxl_style_template.get_name_style_prompt(stylename, (scene_text or "").split(":")[-1])
            pos2 = (scene_text or "").split(":")[-1]
            return neg1, pos1, neg2, pos2, pos1, pos2

        def get_select_template_prompt(person_text, scene_text, evt: gr.SelectData):
            stylename = evt.value["caption"]
            _, neg1, _ = sdxl_style_template.get_name_style_prompt(stylename, (person_text or "").split(":")[-1])
            pos1 = (person_text or "").split(":")[-1]
            _, neg2, _ = sdxl_style_template.get_name_style_prompt(stylename, (scene_text or "").split(":")[-1])
            pos2 = (scene_text or "").split(":")[-1]
            return neg1, pos1, neg2, pos2, stylename, pos1, pos2

        def get_select_name_template_prompt(text, stylename):
            _, neg, _ = sdxl_style_template.get_name_style_prompt(stylename, (text or "").split(":")[-1])
            pos = (text or "").split(":")[-1]
            return neg, pos, pos

        person_scene_text = ctx["person_scene_text"]
        template_type.select(change_template_type, inputs=[template_type], outputs=[stylename, style_example_img])
        try:
            from .tab2_characters import person_scene_text_parse
        except ImportError:
            import importlib.util
            _tab2_path = os.path.join(os.path.dirname(__file__), "tab2_characters.py")
            _spec = importlib.util.spec_from_file_location("tab2_characters", _tab2_path)
            _tab2 = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_tab2)
            person_scene_text_parse = _tab2.person_scene_text_parse
        person_scene_text.change(person_scene_text_parse, inputs=[person_scene_text], outputs=[person_name, scene_name])

        with gr.Tab("角色设定图"):
            with gr.Row():
                with gr.Column(scale=3):
                    person_prompt_neg = gr.Textbox(label="角色neg prompt(模板词)")
                    person_prompt_pos = gr.Textbox(label="角色pos prompt(模板词)")
                    generate_person = gr.Button("模板生成角色图", variant="primary")
                    gr.Markdown("点击生成图片可自动移动到参考区")
                with gr.Column(scale=2):
                    person_img = gr.Gallery(label="角色图", columns=1,  height=300)
                    register_person = gr.Button("注册角色图", variant="primary")
                with gr.Column(scale=3):
                    person_simple_neg = gr.Textbox(value=DEFAULT_NEG_PROMPT, label="角色neg prompt(参考图)")
                    person_simple_pos = gr.Textbox(label="角色pos prompt(参考图)")
                    person_scale_simple = gr.Slider(0, 1, 0.5, label="参考权重", step=0.1)
                    generate_person_simple = gr.Button("参考生成角色图", variant="primary")
                with gr.Column(scale=2):
                    gr.Markdown("点击图片自动删除")
                    person_ref = gr.Gallery(label="角色 ref", columns=1,  height=300)
                person_img.select(shared.get_select_to_candi_image, inputs=[person_ref], outputs=[person_ref])
                person_ref.select(shared.get_select_to_remove, inputs=[person_ref], outputs=[person_ref])

        with gr.Tab("场景设定图"):
            with gr.Row():
                with gr.Column(scale=3):
                    scene_prompt_neg = gr.Textbox(label="场景neg prompt(模板词)")
                    scene_prompt_pos = gr.Textbox(label="场景pos prompt(模板词)")
                    generate_scene = gr.Button("模版生成场景图", variant="primary")
                    gr.Markdown("点击生成图片可自动移动到参考区")
                with gr.Column(scale=2):
                    scene_img = gr.Gallery(label="场景图", columns=1,  height=300)
                    register_scene = gr.Button("注册场景图", variant="primary")
                with gr.Column(scale=3):
                    scene_simple_neg = gr.Textbox(value=DEFAULT_NEG_PROMPT, label="场景neg prompt(参考图)")
                    scene_simple_pos = gr.Textbox(label="场景pos prompt(参考图)")
                    scene_scale_simple = gr.Slider(0, 1, 0.5, label="参考权重", step=0.1)
                    generate_scene_simple = gr.Button("参考生成场景图", variant="primary")
                with gr.Column(scale=2):
                    gr.Markdown("点击图片自动删除")
                    style_ref = gr.Gallery(label="场景 ref", columns=1,  height=300)
                scene_img.select(shared.get_select_to_candi_image, inputs=[style_ref], outputs=[style_ref])
                style_ref.select(shared.get_select_to_remove, inputs=[style_ref], outputs=[style_ref])

        with gr.Tab("动作设定图"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("点击图片可自动移动到选定区")
                    action_img = gr.Gallery(label="动作图", columns=1,  height=300)
                    select_action_img = gr.Image(type="pil", label="动作选定图",  height=300)
                with gr.Column(scale=2):
                    action_prompt_neg = gr.Textbox(value=DEFAULT_NEG_PROMPT, label="动作neg prompt(模板词)")
                    action_prompt_pos = gr.Textbox(label="动作pos prompt(模板词)")
                    generate_action = gr.Button("生成动作图", variant="primary")
                    register_action_name = gr.Textbox(label="动作注册名")
                    register_action = gr.Button("注册动作图", variant="primary")
                with gr.Column(scale=2):
                    action_fg_mask = gr.Gallery(label="动作前景与mask", columns=1,  height=600)
                action_img.select(shared.get_select_to_candi_image2, inputs=None, outputs=[select_action_img])
                action_fg_mask.select(shared.get_select_to_remove, inputs=[action_fg_mask], outputs=[action_fg_mask])

        seed = ctx["seed"]
        guidance = ctx["guidance"]
        width = ctx["width"]
        height = ctx["height"]
        num_inference_steps = ctx["num_inference_steps"]
        num_img_per_prompt = ctx["num_img_per_prompt"]
        t2i_backend = ctx["t2i_backend"]
        t2i_model_id = ctx["t2i_model_id"]

        stylename.select(apply_select_stylename, inputs=[stylename, person_name, scene_name], outputs=[person_prompt_neg, person_prompt_pos, scene_prompt_neg, scene_prompt_pos, person_simple_pos, scene_simple_pos])
        style_example_img.select(get_select_template_prompt, inputs=[person_name, scene_name], outputs=[person_prompt_neg, person_prompt_pos, scene_prompt_neg, scene_prompt_pos, stylename, person_simple_pos, scene_simple_pos])
        person_name.change(get_select_name_template_prompt, inputs=[person_name, stylename], outputs=[person_prompt_neg, person_prompt_pos, person_simple_pos])
        scene_name.change(get_select_name_template_prompt, inputs=[scene_name, stylename], outputs=[scene_prompt_neg, scene_prompt_pos, scene_simple_pos])

        generate_person.click(generate_no_ref, inputs=[person_prompt_neg, person_prompt_pos, seed, num_inference_steps, guidance, width, height, num_img_per_prompt], outputs=[person_img])
        generate_person_simple.click(generate_adapter_person, inputs=[person_simple_neg, person_simple_pos, person_ref, seed, num_inference_steps, guidance, width, height, num_img_per_prompt, person_scale_simple], outputs=[person_img])
        generate_scene.click(generate_no_ref, inputs=[scene_prompt_neg, scene_prompt_pos, seed, num_inference_steps, guidance, width, height, num_img_per_prompt], outputs=[scene_img])
        generate_scene_simple.click(generate_adapter_scene, inputs=[scene_simple_neg, scene_simple_pos, style_ref, seed, num_inference_steps, guidance, width, height, num_img_per_prompt, scene_scale_simple], outputs=[scene_img])

        btn_batch_refs.click(
            batch_generate_character_scene_refs,
            inputs=[ctx["state_global_characters"], ctx["state_global_scenes"], stylename, seed, num_inference_steps, guidance, width, height, t2i_backend, t2i_model_id],
            outputs=[batch_refs_report],
        )
        btn_batch_9grid.click(
            batch_generate_9grid_for_characters_scenes,
            inputs=[ctx["state_global_characters"], ctx["state_global_scenes"], seed, num_inference_steps, guidance, width, height, t2i_backend, t2i_model_id],
            outputs=[batch_9grid_report],
        )

        def test_tab3(tb, tm):
            prompt = "a cute frog in a green meadow, cartoon style"
            try:
                model = get_t2i_model(backend=tb or T2I_BACKEND_OPEN_SOURCE, model_id=tm or "sdxl_lightning")
                img = model.generate_face_style(None, None, prompt, DEFAULT_NEG_PROMPT, steps=4, seed=random.randint(1, MAX_SEED), guidance=1.0, width=512, height=512)
                return [(img, prompt)]
            except Exception as e:
                return [(None, "本页测试失败: %s" % str(e))]

        btn_test_tab3.click(test_tab3, inputs=[t2i_backend, t2i_model_id], outputs=[person_img])

    ctx["generate_single_fenjin_img"] = generate_single_fenjin_img
    ctx["person_img"] = person_img
    ctx["scene_img"] = scene_img
    ctx["person_name"] = person_name
    ctx["scene_name"] = scene_name
    ctx["register_person"] = register_person
    ctx["register_scene"] = register_scene


def run_standalone(t2imodel=None, server_name="0.0.0.0", server_port=8084):
    demo = gr.Blocks(title="3、角色/场景图与分镜画面（独立调试）")
    with demo:
        state_global_characters = gr.State(value=[])
        state_global_scenes = gr.State(value=[])
        person_scene_text = gr.Textbox(value=shared.DeFault_person_scene_text, label="角色场景text（占位）")
        seed = gr.Slider(0, 10000000000000, 0, label="seed", step=1)
        guidance = gr.Slider(0, 10, 1, label="guidance", step=0.5)
        width = gr.Slider(480, 1920, 1024, label="width", step=24)
        height = gr.Slider(480, 1920, 1024, label="height", step=24)
        num_inference_steps = gr.Slider(1, 8, 4, label="infer steps", step=1)
        num_img_per_prompt = gr.Slider(1, 4, 3, label="imgnum/prompt", step=1)
        t2i_backend = gr.Dropdown(choices=[T2I_BACKEND_OPEN_SOURCE, "commercial"], value=T2I_BACKEND_OPEN_SOURCE, label="文生图后端")
        t2i_model_id = gr.Dropdown(choices=T2I_OPEN_SOURCE_MODEL_IDS + T2I_COMMERCIAL_MODEL_IDS, value="sdxl_lightning", label="文生图模型")
        ctx = {
            "state_global_characters": state_global_characters,
            "state_global_scenes": state_global_scenes,
            "person_scene_text": person_scene_text,
            "seed": seed,
            "guidance": guidance,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "num_img_per_prompt": num_img_per_prompt,
            "t2i_backend": t2i_backend,
            "t2i_model_id": t2i_model_id,
            "t2imodel": t2imodel,
        }
        build(ctx)
        gr.Markdown("当前为 **Tab3 独立调试**。")
    demo.queue().launch(server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    run_standalone()
