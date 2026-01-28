import torch
import pandas as pd
import os,sys,cv2,time,random,re
from urllib import parse
import html
import numpy as np
import requests
from PIL import Image,ImageDraw
import requests as req
from io import BytesIO
import hashlib,shutil
import time,datetime,json
from moviepy.editor import *


# sys.path.append(...)
# from tts.OpenVoice_en.openvoice_demo import openvoice_tts
from tts.edgetts_demo import edgetts_api, EN_LANGUAGE_ID, CH_LANGUAGE_ID


def translate(text, to_language="en", text_language="zh-CN"):
    GOOGLE_TRANSLATE_URL = 'http://translate.google.com/m?q=%s&tl=%s&sl=%s'
    text = parse.quote(text)
    url = GOOGLE_TRANSLATE_URL % (text,to_language,text_language)
    response = requests.get(url)
    data = response.text
    expr = r'(?s)class="(?:t0|result-container)">(.*?)<'
    result = re.findall(expr, data)
    if (len(result) == 0):
        return ""
    return html.unescape(result[0])
 
def convert_vtt_srt_ast(subtitlefile1,subtitlefile2):
    '''
        *.vtt   -> *.srt
        *.lyric -> *.srt
        *.ass   -> *.srt
        *.srt   -> *.ass
        ...
    '''
    os.system(f'ffmpeg -y -i {subtitlefile1} {subtitlefile2}')   

def combine_audiopaths(audiopaths,savepath):
    paths=[]
    for current_tts_path in audiopaths:
        audio_clip = AudioFileClip(current_tts_path)
        #print(audio_clip)
        paths.append(audio_clip)
    final_audio = concatenate_audioclips(paths)
    final_audio.write_audiofile(savepath)
    
def get_every_script(scripfile,trans=True):
    scene_title_content=[]
    scene_title_content_en=[]
    idx=1
    lines=open(scripfile,'r',encoding='utf-8').readlines()
    contents=''.join(lines)
    scenes=contents.split('##')
    for scene in scenes:
        print(scene)
        items=scene.replace('\n','').split('/')
        if len(items)==4:
            title,content,subtitle,_=items
        else:
            title,content,subtitle=items
        scene_title_content.append(f'{title}～～～{content}')
        if trans:
            title_en=translate(f'{title}', "en","zh-CN")
            content_en=translate(content,"en","zh-CN")
            scene_title_content_en.append(f'{title_en}～～～{content_en}')
            print(idx,title_en,content_en)
        idx+=1
        #break
    return scene_title_content,scene_title_content_en
        

def generate_script_audio(scripfile,outputdir='data/audio/',tts_type='edgetts'):
    if tts_type=='openvoice':
        tts_tool=openvoice_tts()
    elif tts_type=='edgetts':
        tts_tool=edgetts_api()
    
    if not os.path.exists(outputdir):os.makedirs(outputdir)
    scene_title_content,scene_title_content_en=get_every_script(scripfile)
    for idx,title_content in enumerate(scene_title_content_en):
        if tts_type=='openvoice':
            tts_tool.base_tts(title_content,f'{outputdir}script{idx+1}.mp3',
                              speaker='friendly',
                              speed=0.75)
        elif tts_type=='edgetts':
            tts_tool.base_tts(title_content,f'data/audio/script{idx+1}.mp3',
                              webvitfile=f'{outputdir}script{idx+1}.vtt',
                              voice='zh-CN-YunxiNeural',
                              rate='+0%')
            convert_vtt_srt_ast(f'{outputdir}script{idx+1}.vtt',f'{outputdir}script{idx+1}.srt')
            os.remove(f'{outputdir}script{idx+1}.vtt')


_edge_tts_tool = edgetts_api()


def synthesize(
    text: str,
    voice_or_speaker_id: str,
    rate: float = 0,
    output_path: str = None,
    subtitle_path: str = None,
    backend: str = None,
    model_id: str = None,
) -> tuple:
    """
    统一 TTS 入口。返回 (audio_path, subtitle_path)，subtitle_path 可为 None。
    backend: edgetts | qwen3_tts | qwen_tts，默认 edgetts。
    model_id: 可选，如 qwen3-tts-flash。
    """
    backend = (backend or "edgetts").strip().lower()
    if backend in ("qwen3_tts", "qwen_tts", "qwen3-tts"):
        from tts.qwen3_tts import synthesize as _qwen_synth
        return _qwen_synth(
            text=text,
            voice_or_speaker_id=voice_or_speaker_id,
            rate=rate,
            output_path=output_path,
            subtitle_path=subtitle_path,
            model_id=model_id,
        )
    # 默认 Edge-TTS
    if not output_path:
        output_path = "data/audio/tts_out.mp3"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    base = os.path.splitext(output_path)[0]
    vtt_path = base + ".vtt"
    srt_path = subtitle_path or (base + ".srt")
    voice = (voice_or_speaker_id or "zh-CN-YunxiNeural").split(":")[0]
    rate_str = f"+{int(rate)}%" if rate >= 0 else f"{int(rate)}%"
    _edge_tts_tool.sync_tts(text, output_path, webvitfile=vtt_path, voice=voice, rate=rate_str)
    convert_vtt_srt_ast(vtt_path, srt_path)
    if os.path.isfile(vtt_path):
        os.remove(vtt_path)
    return output_path, srt_path


def generate_text_audio(
    text,
    voice,
    datafilename,
    rate=0,
    outputdir="data/audio/",
    tts_backend: str = None,
    tts_model_id: str = None,
):
    """生成单段配音。voice 为 Edge 的 voice 或 Qwen 音色名。增加 tts_backend、tts_model_id 下传到 synthesize。"""
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    if tts_backend and tts_backend.lower() in ("qwen3_tts", "qwen_tts", "qwen3-tts"):
        out_mp3 = os.path.join(outputdir, f"{datafilename}.mp3")
        out_srt = os.path.join(outputdir, f"{datafilename}.srt")
        return synthesize(
            text, voice, rate,
            output_path=out_mp3,
            subtitle_path=out_srt,
            backend=tts_backend,
            model_id=tts_model_id,
        )
    if voice not in EN_LANGUAGE_ID and voice not in CH_LANGUAGE_ID:
        return None, None
    out_mp3 = os.path.join(outputdir, f"{datafilename}.mp3")
    out_srt = os.path.join(outputdir, f"{datafilename}.srt")
    return synthesize(
        text, voice, rate,
        output_path=out_mp3,
        subtitle_path=out_srt,
        backend=tts_backend or "edgetts",
        model_id=tts_model_id,
    )

def translate_en_to_ch(text):
    return translate(text,'zh-CN','en')

def translate_ch_to_en(text):
    return translate(text)




if __name__=='__main__':
    print(translate_en_to_ch('i love you'))
    #print(get_every_script('data/script/白雪公主与七个小矮人2.txt'))
    #generate_script_audio('data/script/白雪公主与七个小矮人2.txt')
    #generate_text_audio('这是一个测试内容','zh-CN-YunyangNeural','1')
    pass