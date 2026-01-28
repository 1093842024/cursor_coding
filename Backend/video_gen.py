# -*- coding: utf-8 -*-
"""
分镜视频生成统一入口：backend in ["slideshow", "i2v", "t2v"]。
- slideshow：图+音+字幕合成为幻灯片式分镜（沿用 video_utils）。
- i2v / t2v：调用真实 I2V/T2V 模型再合成；开源主选 Wan2.2，商用可按 backend+model_id 接入。
"""
import os
from typing import List, Union


def generate_shot_video(
    images: Union[List[str], str],
    audio_path: str,
    subtitle_path: str,
    output_path: str = None,
    output_dir: str = "data/video/",
    savename: str = "shot.mp4",
    backend: str = "slideshow",
    model_id: str = None,
    fadetime: float = 1.0,
    **kwargs,
) -> str:
    """
    生成单条分镜视频。返回最终视频文件路径。
    images: 图片路径列表或 "path1;path2;..." 字符串。
    backend: slideshow | i2v | t2v。
    model_id: 可选，如 wan2.2、svd、cogvideox 等，用于 i2v/t2v。
    """
    backend = (backend or "slideshow").strip().lower()
    if isinstance(images, (list, tuple)):
        image_str = ";".join(str(p) for p in images)
    else:
        image_str = (images or "").strip()

    if not image_str or not audio_path or not subtitle_path:
        raise ValueError("generate_shot_video 需要 images、audio_path、subtitle_path 均非空")

    os.makedirs(output_dir, exist_ok=True)
    out_file = output_path or os.path.join(output_dir, savename)

    if backend == "slideshow":
        from video_utils import comb_video
        return comb_video.image_to_video_with_audio_subtitle(
            image_str,
            audio_path,
            subtitle_path,
            output_dir,
            os.path.basename(out_file),
            fadetime=fadetime,
        )

    if backend in ("i2v", "t2v"):
        # 扩展点：接入 Wan2.2、SVD、CogVideoX 等。先做占位，避免未实现时静默失败。
        _run_i2v_t2v(
            image_str=image_str,
            audio_path=audio_path,
            subtitle_path=subtitle_path,
            out_path=out_file,
            backend=backend,
            model_id=model_id,
            fadetime=fadetime,
            **kwargs,
        )
        return out_file

    raise ValueError("backend 须为 slideshow、i2v、t2v 之一，当前: %s" % backend)


def _run_i2v_t2v(
    image_str: str,
    audio_path: str,
    subtitle_path: str,
    out_path: str,
    backend: str,
    model_id: str = None,
    fadetime: float = 1.0,
    **kwargs,
) -> None:
    """内部：执行 i2v/t2v 生成并写文件到 out_path。支持 model_id=svd；wan2.2 暂以 svd 代用。"""
    model_id = (model_id or "wan2.2").strip().lower()
    first_image = (image_str or "").split(";")[0].strip() if ";" in (image_str or "") else (image_str or "").strip()
    if not first_image or not os.path.isfile(first_image):
        raise FileNotFoundError("I2V 需要有效首图路径: %s" % first_image)

    out_dir = os.path.dirname(out_path)
    out_name = os.path.basename(out_path)
    out_dir = (out_dir or ".") + os.sep
    raw_path = os.path.join(os.path.dirname(out_path) or ".", "_i2v_raw_" + out_name)

    if model_id in ("svd", "wan2.2"):
        try:
            from mora.iterate_generate_video import svd_image_to_video
            svd_image_to_video(first_image, raw_path, **{k: v for k, v in kwargs.items() if k in ("num_frames", "fps", "seed")})
        except Exception as e:
            if os.path.isfile(raw_path):
                try:
                    os.remove(raw_path)
                except OSError:
                    pass
            raise RuntimeError("I2V SVD 生成失败: %s" % e)
        try:
            from video_utils import comb_video
            comb_video.video_audio_subtitle_to_video(raw_path, audio_path, subtitle_path, out_dir, out_name)
        finally:
            if os.path.isfile(raw_path):
                try:
                    os.remove(raw_path)
                except OSError:
                    pass
        return

    raise NotImplementedError(
        "I2V/T2V 仅支持 model_id=svd 或 wan2.2（暂用 SVD）。请使用 backend='slideshow' 或 model_id=svd。"
    )
