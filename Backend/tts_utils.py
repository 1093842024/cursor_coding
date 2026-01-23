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


#sys.path.append('/Users/glennge/work/cv/diffusion/MoneyPrinter/Backend/OpenVoice_en/')
#from tts.OpenVoice_en.openvoice_demo import openvoice_tts
from tts.edgetts_demo import edgetts_api,EN_LANGUAGE_ID,CH_LANGUAGE_ID


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


edge_tts_tool=edgetts_api()

def generate_text_audio(text,voice,datafilename,rate=0,outputdir='data/audio/'):
    if not os.path.exists(outputdir):os.makedirs(outputdir)
    if voice not in EN_LANGUAGE_ID and voice not in CH_LANGUAGE_ID:
        return None,None
    else:
        edge_tts_tool.sync_tts(text,f'{outputdir}{datafilename}.mp3',
                            webvitfile=f'{outputdir}{datafilename}.vtt',
                            voice=voice,rate=rate)
        convert_vtt_srt_ast(f'{outputdir}{datafilename}.vtt',f'{outputdir}{datafilename}.srt')
        os.remove(f'{outputdir}{datafilename}.vtt')
        return f'{outputdir}{datafilename}.mp3',f'{outputdir}{datafilename}.srt'

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