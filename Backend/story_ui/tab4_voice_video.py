# -*- coding: utf-8 -*-
"""
环节4：配音与分镜视频。分镜旁白 TTS、单镜/全部分镜图与分镜视频、角色/场景/动作库选择与保存。
"""
import os
import sys
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

from tts_utils import CH_LANGUAGE_ID, EN_LANGUAGE_ID, translate_en_to_ch, generate_text_audio
from video_gen import generate_shot_video

Max_fenjin_num = shared.Max_fenjin_num
Default_mask_img = shared.Default_mask_img


def _generate_single_tts(pb_tts_id, ps_tts_id, language, rate, audiotext, tts_backend, tts_model_id):
    if not audiotext or len(audiotext) < 2:
        return None, None
    if language != "en":
        tmpat = translate_en_to_ch(audiotext)
        if len(tmpat) > 1:
            audiotext = tmpat
    audiofile, subtitlefile = generate_text_audio(audiotext, pb_tts_id, "audio", rate, tts_backend=tts_backend, tts_model_id=tts_model_id)
    return gr.update(value=audiofile, visible=True), gr.update(value=subtitlefile, visible=True)


def _generate_tts(pb_tts_id, ps_tts_id, language, rate, tts_backend, tts_model_id, *audiotext):
    addata = [gr.update(value=None, visible=False)] * Max_fenjin_num
    adst = [gr.update(value=None, visible=False)] * Max_fenjin_num
    for i, at in enumerate(audiotext):
        if at and len(at) > 1:
            if language != "en":
                tmpat = translate_en_to_ch(at)
                if len(tmpat) > 1:
                    at = tmpat
            audiofile, subtitlefile = generate_text_audio(at, pb_tts_id, f"audio_{i}", rate, tts_backend=tts_backend, tts_model_id=tts_model_id)
            addata[i] = gr.update(value=audiofile, visible=True)
            adst[i] = gr.update(value=subtitlefile, visible=True)
            yield addata + adst


def _generate_single_fenjin_video(fadeinfadeout, audio, subtitle, img, video_backend, video_model_id, outputdir="data/video/", savename_suffix="0"):
    if not audio or not subtitle or not img or len(audio) == 0 or len(subtitle) == 0:
        return None
    imagefiles = []
    for data in img:
        imgpath, _ = data
        imagefiles.append(imgpath)
    allimagefiles = ";".join(imagefiles)
    os.makedirs(outputdir, exist_ok=True)
    video = generate_shot_video(
        allimagefiles, audio, subtitle,
        output_dir=outputdir, savename=f"video_{savename_suffix}.mp4",
        backend=(video_backend or "slideshow").strip().lower(),
        model_id=(video_model_id or "").strip() or None,
        fadetime=float(fadeinfadeout or 1),
    )
    return gr.update(value=video, visible=True)


def build(ctx):
    """挂载「4、配音与分镜视频」Tab。依赖 ctx 内 content_text、fenjin_parse、generate_single_fenjin_img、person_img/scene_img/register_*（来自 tab3）、seed/guidance 等。"""
    content_text = ctx["content_text"]
    fenjin_parse_fn = ctx["fenjin_parse"]
    generate_single_fenjin_img = ctx.get("generate_single_fenjin_img")
    person_img = ctx.get("person_img")
    person_name = ctx.get("person_name")
    scene_img = ctx.get("scene_img")
    scene_name = ctx.get("scene_name")
    register_person_btn = ctx.get("register_person")
    register_scene_btn = ctx.get("register_scene")
    seed = ctx["seed"]
    num_inference_steps = ctx["num_inference_steps"]
    guidance = ctx["guidance"]
    width = ctx["width"]
    height = ctx["height"]
    num_img_per_prompt = ctx["num_img_per_prompt"]

    with gr.Tab("4、配音与分镜视频"):
        with gr.Row():
            language = gr.Radio(choices=["en", "zh"], value="en", label="旁白字幕语言", visible=False)
            with gr.Column():
                pangbai_tts_id = gr.Dropdown(choices=CH_LANGUAGE_ID + EN_LANGUAGE_ID, value=CH_LANGUAGE_ID[0] if CH_LANGUAGE_ID else None, label="旁白声personid")
                person_tts_id = gr.Dropdown(choices=CH_LANGUAGE_ID + EN_LANGUAGE_ID, value=CH_LANGUAGE_ID[2] if len(CH_LANGUAGE_ID) > 2 else CH_LANGUAGE_ID[0], label="角色声音id")
                tts_backend = gr.Dropdown(choices=["edgetts", "qwen3_tts"], value="edgetts", label="TTS 引擎")
                tts_model_id = gr.Textbox(value="", label="TTS 模型（可选）", placeholder="qwen3-tts-flash 等")
            with gr.Column():
                generate_fenjin_tts = gr.Button("生成所有分镜旁白", variant="primary")
                btn_test_tab4 = gr.Button("本页测试", variant="secondary")
            with gr.Column():
                rate = gr.Slider(-100, 100, 0, label="语速", step=1)
                fadeinfadeout = gr.Slider(0, 5, 1, label="渐入渐出时长", step=0.5)
            with gr.Column():
                video_backend = gr.Dropdown(choices=["slideshow", "i2v", "t2v"], value="slideshow", label="分镜视频后端")
                video_model_id = gr.Dropdown(choices=["svd", "wan2.2"], value="svd", label="I2V/T2V 模型")
            with gr.Column():
                person_scale = gr.Slider(0, 1, 1, label="人物mask权重", step=0.1)
                bg_scale = gr.Slider(0, 1, 1, label="背景mask权重", step=0.1)
            with gr.Column():
                generate_fenjin = gr.Button("生成所有分镜图", variant="secondary", visible=False)
                generate_fenjin_video = gr.Button("生成所有分镜视频", variant="secondary", visible=False)

        audiotext = [None] * Max_fenjin_num
        audiodata = [None] * Max_fenjin_num
        audiosubtitle = [None] * Max_fenjin_num
        videoclip = [None] * Max_fenjin_num
        fenjin_imgs = [None] * Max_fenjin_num
        fenjin_imgs_select = [None] * Max_fenjin_num
        IP_type = [None] * Max_fenjin_num
        scene_type = [None] * Max_fenjin_num
        fenjin_img_prompt = [None] * Max_fenjin_num
        fenjin_audio_generate = [None] * Max_fenjin_num
        fenjin_img_generate = [None] * Max_fenjin_num
        fenjin_video_generate = [None] * Max_fenjin_num

        for i in range(Max_fenjin_num):
            visual = i < 1
            with gr.Tab(f"分镜{i}"):
                with gr.Row():
                    audiotext[i] = gr.Textbox(label="分镜剧情旁白+角色说话", visible=visual, scale=2)
                    IP_type[i] = gr.Dropdown(label="角色", visible=visual, allow_custom_value=True, scale=1)
                    scene_type[i] = gr.Dropdown(label="场景", visible=visual, allow_custom_value=True, scale=0)
                    fenjin_img_prompt[i] = gr.Dropdown(label="分镜图prompt", visible=visual, allow_custom_value=True, scale=2)
                    with gr.Column():
                        fenjin_audio_generate[i] = gr.Button("优化分镜音频", variant="primary")
                        fenjin_img_generate[i] = gr.Button("优化分镜图", variant="primary")
                        fenjin_video_generate[i] = gr.Button("优化分镜视频", variant="primary")
                with gr.Tab("分镜音视频"):
                    with gr.Row():
                        audiodata[i] = gr.Audio(type="filepath", label=f"分镜音频{i}", visible=visual, )
                        audiosubtitle[i] = gr.File(label=f"分镜音频字幕{i}", visible=visual)
                        videoclip[i] = gr.Video(label=f"分镜视频{i}",  visible=visual)
                with gr.Tab("分镜图片"):
                    with gr.Row():
                        fenjin_imgs[i] = gr.Gallery(columns=2, height=300, label=f"分镜图{i}",  visible=visual, scale=2)
                        fenjin_imgs_select[i] = gr.Gallery(columns=2, height=300, label=f"选定分镜图{i}",  visible=visual, scale=2)

        with gr.Row():
            mask_img_lib = gr.Gallery(value=Default_mask_img, label="mask池", columns=5,  height=190, scale=5)
            mask_img_select = gr.Gallery(label="mask选择池", columns=3,  height=190, scale=3)
            mask_img_lib.select(shared.get_select_to_candi_image, inputs=[mask_img_select], outputs=[mask_img_select])
            mask_img_select.select(shared.get_select_to_remove, inputs=[mask_img_select], outputs=[mask_img_select])

        gr.Markdown("注意：点击设定池中图片自动删除(假操作)")
        with gr.Row():
            with gr.Column():
                person_img_select = gr.Gallery(label="角色图注册池", columns=3,  height=320)
                with gr.Row():
                    select_person_lib = gr.Dropdown(choices=["all"], label="选择角色加载", multiselect=True)
                    save_person_lib = gr.Button("更新/存储角色库", variant="primary")
            with gr.Column():
                scene_img_select = gr.Gallery(label="场景图注册池", columns=3,  height=320)
                with gr.Row():
                    select_scene_lib = gr.Dropdown(choices=["all"], label="选择场景加载", multiselect=True)
                    save_scene_lib = gr.Button("更新/存储场景库", variant="primary")
            with gr.Column():
                action_img_select = gr.Gallery(label="动作图注册池", columns=3,  height=320)
                with gr.Row():
                    select_action_lib = gr.Dropdown(choices=["all"], label="选择动作加载", multiselect=True)
                    save_action_lib = gr.Button("更新/存储动作库", variant="primary")

        person_img_select.select(shared.get_select_to_remove, inputs=[person_img_select], outputs=[person_img_select])
        scene_img_select.select(shared.get_select_to_remove, inputs=[scene_img_select], outputs=[scene_img_select])
        action_img_select.select(shared.get_select_to_remove, inputs=[action_img_select], outputs=[action_img_select])
        save_person_lib.click(shared.saveperson_lib, inputs=[person_img_select], outputs=[select_person_lib])
        save_scene_lib.click(shared.savescene_lib, inputs=[scene_img_select], outputs=[select_scene_lib])
        save_action_lib.click(shared.saveaction_lib, inputs=[action_img_select], outputs=[select_action_lib])
        select_person_lib.change(shared.selectperson_lib, inputs=[select_person_lib], outputs=[person_img_select])
        select_scene_lib.change(shared.selectscene_lib, inputs=[select_scene_lib], outputs=[scene_img_select])
        select_action_lib.change(shared.selectaction_lib, inputs=[select_action_lib], outputs=[action_img_select])

        if register_person_btn is not None and person_img is not None and person_name is not None:
            register_person_btn.click(shared.register_img_to_candi, inputs=[person_img, person_name, person_img_select], outputs=[person_img_select])
        if register_scene_btn is not None and scene_img is not None and scene_name is not None:
            register_scene_btn.click(shared.register_img_to_candi, inputs=[scene_img, scene_name, scene_img_select], outputs=[scene_img_select])

        content_text.change(
            fenjin_parse_fn,
            inputs=[content_text],
            outputs=audiotext + IP_type + scene_type + fenjin_img_prompt,
        )

        generate_fenjin_tts.click(
            _generate_tts,
            inputs=[pangbai_tts_id, person_tts_id, language, rate, tts_backend, tts_model_id] + audiotext,
            outputs=audiodata + audiosubtitle,
        )

        for i in range(Max_fenjin_num):
            fenjin_audio_generate[i].click(
                _generate_single_tts,
                inputs=[pangbai_tts_id, person_tts_id, language, rate, audiotext[i], tts_backend, tts_model_id],
                outputs=[audiodata[i], audiosubtitle[i]],
            )

            def _make_video_fn(idx):
                def _fn(f, a, s, img, vb, vm):
                    return _generate_single_fenjin_video(f, a, s, img, vb, vm, savename_suffix=str(idx))
                return _fn
            fenjin_video_generate[i].click(
                _make_video_fn(i),
                inputs=[fadeinfadeout, audiodata[i], audiosubtitle[i], fenjin_imgs_select[i], video_backend, video_model_id],
                outputs=[videoclip[i]],
            )

            if generate_single_fenjin_img is not None:
                fenjin_img_generate[i].click(
                    generate_single_fenjin_img,
                    inputs=[person_img_select, scene_img_select, seed, num_inference_steps, guidance, width, height, num_img_per_prompt,
                            mask_img_select, person_scale, bg_scale,
                            IP_type[i], scene_type[i], fenjin_img_prompt[i]],
                    outputs=[fenjin_imgs[i]],
                )
            fenjin_imgs[i].select(shared.get_select_to_candi_image, inputs=[fenjin_imgs_select[i]], outputs=[fenjin_imgs_select[i]])
            fenjin_imgs_select[i].select(shared.get_select_to_remove, inputs=[fenjin_imgs_select[i]], outputs=[fenjin_imgs_select[i]])

        def test_tab4(pb, r, tb, tm):
            text = "这是一段本页测试旁白，用于验证 TTS 与分镜视频链路。"
            voice = pb or (CH_LANGUAGE_ID[0] if CH_LANGUAGE_ID else "zh-CN-YunxiNeural")
            outdir = "data/audio/"
            os.makedirs(outdir, exist_ok=True)
            try:
                audio_path, sub_path = generate_text_audio(text, voice, "test_tab4", rate=r or 0, outputdir=outdir, tts_backend=tb, tts_model_id=tm or None)
            except Exception:
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            vidir = "data/video/"
            os.makedirs(vidir, exist_ok=True)
            video_path = None
            for p in ["data/image/dragon_baby1.png", "data/image/boy.png"]:
                if os.path.isfile(p):
                    video_path = generate_shot_video(p, audio_path, sub_path, output_dir=vidir, savename="test_tab4.mp4", backend="slideshow", fadetime=0.5)
                    break
            return gr.update(value=audio_path, visible=True), gr.update(value=sub_path, visible=True), gr.update(value=video_path, visible=video_path is not None)

        btn_test_tab4.click(
            test_tab4,
            inputs=[pangbai_tts_id, rate, tts_backend, tts_model_id],
            outputs=[audiodata[0], audiosubtitle[0], videoclip[0]],
        )

    ctx["videoclip"] = videoclip


def run_standalone(server_name="0.0.0.0", server_port=8085):
    try:
        from .tab2_characters import fenjin_parse
    except ImportError:
        import importlib.util
        _tab2_path = os.path.join(os.path.dirname(__file__), "tab2_characters.py")
        _spec = importlib.util.spec_from_file_location("tab2_characters", _tab2_path)
        _tab2 = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_tab2)
        fenjin_parse = _tab2.fenjin_parse
    demo = gr.Blocks(title="4、配音与分镜视频（独立调试）")
    with demo:
        gr.Markdown("**Tab4 独立调试**：仅包含配音与分镜视频。")
        content_text = gr.Textbox(value=shared.DeFault_fenjin_text, label="分镜text（占位）", max_lines=6)
        ctx = {
            "content_text": content_text,
            "fenjin_parse": fenjin_parse,
            "generate_single_fenjin_img": None,
            "person_img": None,
            "person_name": None,
            "scene_img": None,
            "scene_name": None,
            "register_person": None,
            "register_scene": None,
            "seed": gr.Slider(0, 10000000000000, 0, label="seed"),
            "num_inference_steps": gr.Slider(1, 8, 4, label="steps"),
            "guidance": gr.Slider(0, 10, 1, label="guidance"),
            "width": gr.Slider(480, 1920, 1024, label="width"),
            "height": gr.Slider(480, 1920, 1024, label="height"),
            "num_img_per_prompt": gr.Slider(1, 4, 3, label="num_img"),
        }
        build(ctx)
    demo.queue().launch(server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    run_standalone()
