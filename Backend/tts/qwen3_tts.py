# -*- coding: utf-8 -*-
"""Qwen3-TTS（阿里云百炼 Qwen-TTS 实时语音）对接。需配置 DASHSCOPE_API_KEY 或 QWEN_TTS_API_KEY。"""
import os
import re


def synthesize(
    text: str,
    voice_or_speaker_id: str,
    rate: float = 0,
    output_path: str = None,
    subtitle_path: str = None,
    model_id: str = None,
) -> tuple:
    """
    使用 Qwen3-TTS 合成语音。
    voice_or_speaker_id: 音色名，如 Cherry、Xiaoyan 等（与 Edge-TTS 的 voice 格式不同）。
    rate: 语速调整，暂可忽略或映射到 API 参数。
    返回 (output_path, subtitle_path)。Qwen3 不提供精确字幕时间戳时，subtitle_path 可为 None。
    """
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_TTS_API_KEY", "")
    if not api_key:
        raise ValueError("Qwen3-TTS 需要配置 DASHSCOPE_API_KEY 或 QWEN_TTS_API_KEY")

    try:
        import dashscope
    except ImportError:
        raise RuntimeError("Qwen3-TTS 需要安装 dashscope: pip install dashscope")

    model = model_id or "qwen3-tts-flash"
    voice = (voice_or_speaker_id or "Cherry").split(":")[0].strip()

    dashscope.api_key = api_key
    resp = dashscope.MultiModalConversation.call(
        model=model,
        text=text,
        voice=voice,
        language_type="Auto",
    )

    if resp.status_code != 200 or not getattr(resp.output, "audio", None):
        msg = getattr(resp, "message", str(resp))
        raise RuntimeError("Qwen3-TTS 调用失败: %s" % msg)

    audio_url = resp.output.audio.url
    import requests
    r = requests.get(audio_url, timeout=30)
    r.raise_for_status()
    if not output_path:
        output_path = "tts_out.mp3"
    with open(output_path, "wb") as f:
        f.write(r.content)

    # 若需要字幕，可生成简单整段字幕（起止时间需另行估算或由上层用 Edge 生成）
    if subtitle_path and text:
        # 占位：整段 0 到估计时长的一行 srt
        try:
            import subprocess
            proc = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", output_path],
                capture_output=True, text=True, timeout=5
            )
            dur = float(proc.stdout.strip() or "1.0")
        except Exception:
            dur = max(1.0, len(text) * 0.1)
        with open(subtitle_path, "w", encoding="utf-8") as f:
            f.write("1\n00:00:00,000 --> %s\n%s\n" % (_sec_to_srt(dur), text))
    else:
        subtitle_path = None

    return output_path, subtitle_path


def _sec_to_srt(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec % 1) * 1000)
    return "%02d:%02d:%02d,%03d" % (h, m, s, ms)
