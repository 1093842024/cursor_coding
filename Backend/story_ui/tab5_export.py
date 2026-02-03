# -*- coding: utf-8 -*-
"""
环节5：成片导出。分镜视频串联、片头片尾、BGM、标题与结尾字。
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

from video_utils import comb_video

DeFault_BGM_path = getattr(shared, "DeFault_BGM_path", [])


def _generate_final_video(start_img, end_img, BGM, volumn, starttext, storyname, endtext, *videoclip):
    outputvideopath = "data/video/"
    valideclip = [c for c in videoclip if c is not None]
    if len(valideclip) == 0:
        return None
    comb_video.generate_final_video(
        valideclip, start_img, end_img, outputvideopath, "finalvideo.mp4",
        bgm=BGM, bgmvolume=volumn, starttext=starttext or "", storyname=storyname or "", endtext=endtext or "",
    )
    return outputvideopath + "finalvideo.mp4"


def _test_tab5():
    vdir = "data/video/"
    cand = [os.path.join(vdir, "test_tab4.mp4"), os.path.join(vdir, "video_0.mp4")]
    existing = [p for p in cand if os.path.isfile(p)]
    if not existing:
        raise gr.Error("暂无分镜视频。请先在「配音与分镜视频」Tab 点击「本页测试」或生成分镜视频后再导出。")
    outpath = os.path.join(vdir, "finalvideo_test.mp4")
    try:
        comb_video.generate_final_video(existing, None, None, vdir, "finalvideo_test.mp4", bgm=None, bgmvolume=0)
        return outpath if os.path.isfile(outpath) else existing[0]
    except Exception as e:
        raise gr.Error("成片合成失败: %s" % str(e))


def build(ctx):
    """挂载「5、成片导出」Tab。依赖 ctx 内 videoclip 列表。"""
    videoclip = ctx.get("videoclip") or []

    with gr.Tab("5、成片导出"):
        gr.Markdown("**成片导出**：片头片尾、BGM、分镜列表合成。")
        with gr.Row():
            with gr.Column(scale=1):
                start_img = gr.Image("data/image/dragon_baby1.png", type="filepath", label="片头", )
                end_img = gr.Image("data/image/goodnight1.png", type="filepath", label="片尾", )
            with gr.Column(scale=2):
                _bgm0 = shared.get_bgm_song(DeFault_BGM_path[0]) if DeFault_BGM_path else None
                BGM = gr.Audio(value=_bgm0, type="filepath", label="背景音乐", )
                with gr.Row():
                    bgmlist = gr.Dropdown(choices=DeFault_BGM_path or [""], value=(DeFault_BGM_path[0] if DeFault_BGM_path else None), label="可选bgm")
                    volumn = gr.Slider(0, 2, 1, label="背景音量", step=0.1)
                with gr.Row():
                    starttext = gr.Textbox(value="龙宝的睡前故事", label="片头字")
                    storyname = gr.Textbox(value="", label="故事名")
                    endtext = gr.Textbox(value="～晚安好梦～", label="片尾字")
            with gr.Column(scale=2):
                generate_allvideo = gr.Button("生成完整视频", variant="primary")
                btn_test_tab5 = gr.Button("本页测试", variant="secondary")
                allvideo = gr.Video(label="完整视频", )
        bgmlist.select(shared.get_bgm_song, inputs=[bgmlist], outputs=[BGM])
        generate_allvideo.click(
            _generate_final_video,
            inputs=[start_img, end_img, BGM, volumn, starttext, storyname, endtext] + videoclip,
            outputs=[allvideo],
        )
        btn_test_tab5.click(_test_tab5, inputs=[], outputs=[allvideo])


def run_standalone(server_name="0.0.0.0", server_port=8086):
    demo = gr.Blocks(title="5、成片导出（独立调试）")
    with demo:
        gr.Markdown("**Tab5 独立调试**：用 data/video/ 下已有分镜视频合成成片。")
        ctx = {"videoclip": []}
        build(ctx)
    demo.queue().launch(server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    run_standalone()
