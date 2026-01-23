import os,sys,shutil
import uuid
import requests
from typing import List
from moviepy.editor import *

from datetime import timedelta
from moviepy.video.fx.all import crop
from moviepy.video.tools.subtitles import SubtitlesClip



def search_for_stock_videos(query: str, api_key: str, it: int, min_dur: int):
    """
    Searches for stock videos based on a query.

    Args:
        query (str): The query to search for.
        api_key (str): The API key to use.

    Returns:
        List[str]: A list of stock videos.
    """
    # Build headers
    headers = {"Authorization": api_key}
    # Build URL
    qurl = f"https://api.pexels.com/videos/search?query={query}&per_page={it}"
    # Send the request
    r = requests.get(qurl, headers=headers)
    # Parse the response
    response = r.json()
    # Parse each video
    raw_urls = []
    video_url = []
    video_res = 0
    try:
        # loop through each video in the result
        for i in range(it):
            #check if video has desired minimum duration
            if response["videos"][i]["duration"] < min_dur:
                continue
            raw_urls = response["videos"][i]["video_files"]
            temp_video_url = ""
            
            # loop through each url to determine the best quality
            for video in raw_urls:
                # Check if video has a valid download link
                if ".com/external" in video["link"]:
                    # Only save the URL with the largest resolution
                    if (video["width"]*video["height"]) > video_res:
                        temp_video_url = video["link"]
                        video_res = video["width"]*video["height"]
                        
            # add the url to the return list if it's not empty
            if temp_video_url != "":
                video_url.append(temp_video_url)
                
    except Exception as e:
        print(colored("[-] No Videos found.", "red"))
        print(colored(e, "red"))

    # Let user know
    print(colored(f"\t=> \"{query}\" found {len(video_url)} Videos", "cyan"))

    # Return the video url
    return video_url



def __generate_subtitles_locally(sentences: List[str], audio_clips: List[AudioFileClip]):
    """
    Generates subtitles from a given audio file and returns the path to the subtitles.

    Args:
        sentences (List[str]): all the sentences said out loud in the audio clips
        audio_clips (List[AudioFileClip]): all the individual audio clips which will make up the final audio track
    Returns:
        str: The generated subtitles
    """

    def convert_to_srt_time_format(total_seconds):
        # Convert total seconds to the SRT time format: HH:MM:SS,mmm
        if total_seconds == 0:
            return "0:00:00,0"
        return str(timedelta(seconds=total_seconds)).rstrip('0').replace('.', ',')

    start_time = 0
    subtitles = []

    for i, (sentence, audio_clip) in enumerate(zip(sentences, audio_clips), start=1):
        duration = audio_clip.duration
        end_time = start_time + duration

        # Format: subtitle index, start time --> end time, sentence
        subtitle_entry = f"{i}\n{convert_to_srt_time_format(start_time)} --> {convert_to_srt_time_format(end_time)}\n{sentence}\n"
        subtitles.append(subtitle_entry)

        start_time += duration  # Update start time for the next subtitle

    return "\n".join(subtitles)



def save_video(video_url: str, directory: str = "../temp") :
    """
    Saves a video from a given URL and returns the path to the video.

    Args:
        video_url (str): The URL of the video to save.
        directory (str): The path of the temporary directory to save the video to

    Returns:
        str: The path to the saved video.
    """
    video_id = uuid.uuid4()
    video_path = f"{directory}/{video_id}.mp4"
    try:
        with open(video_path, "wb") as f:
            f.write(requests.get(video_url).content)
        return video_path
    except Exception as e:
        print('download error',video_url,e)
        return None


def combine_videos(video_paths: List[str], max_duration: int, max_clip_duration: int, threads: int):
    """
    Combines a list of videos into one video and returns the path to the combined video.

    Args:
        video_paths (List): A list of paths to the videos to combine.
        max_duration (int): The maximum duration of the combined video.
        max_clip_duration (int): The maximum duration of each clip.
        threads (int): The number of threads to use for the video processing.

    Returns:
        str: The path to the combined video.
    """
    video_id = uuid.uuid4()
    combined_video_path = f"../temp/{video_id}.mp4"
    
    # Required duration of each clip
    req_dur = max_duration / len(video_paths)

    print(("[+] Combining videos...", "blue"))
    print((f"[+] Each clip will be maximum {req_dur} seconds long.", "blue"))

    clips = []
    tot_dur = 0
    # Add downloaded clips over and over until the duration of the audio (max_duration) has been reached
    while tot_dur < max_duration:
        for video_path in video_paths:
            clip = VideoFileClip(video_path)
            clip = clip.without_audio()
            # Check if clip is longer than the remaining audio
            if (max_duration - tot_dur) < clip.duration:
                clip = clip.subclip(0, (max_duration - tot_dur))
            # Only shorten clips if the calculated clip length (req_dur) is shorter than the actual clip to prevent still image
            elif req_dur < clip.duration:
                clip = clip.subclip(0, req_dur)
            clip = clip.set_fps(30)

            # Not all videos are same size,
            # so we need to resize them
            if round((clip.w/clip.h), 4) < 0.5625:
                clip = crop(clip, width=clip.w, height=round(clip.w/0.5625), \
                            x_center=clip.w / 2, \
                            y_center=clip.h / 2)
            else:
                clip = crop(clip, width=round(0.5625*clip.h), height=clip.h, \
                            x_center=clip.w / 2, \
                            y_center=clip.h / 2)
            clip = clip.resize((1080, 1920))

            if clip.duration > max_clip_duration:
                clip = clip.subclip(0, max_clip_duration)

            clips.append(clip)
            tot_dur += clip.duration

    final_clip = concatenate_videoclips(clips)
    final_clip = final_clip.set_fps(30)
    final_clip.write_videofile(combined_video_path, threads=threads)

    return combined_video_path

def combine_videos_simple(video_paths,outputpath, threads=4):
    clips=[]
    for video_path in video_paths:
        clip = VideoFileClip(video_path)
        clips.append(clip)
    audio_clips = [clip.audio for clip in clips]
    video_clips = [clip.without_audio() for clip in clips]
    # 使用concatenate_videoclips合并视频文件
    final_video_clip = concatenate_videoclips(video_clips)
    # 使用concatenate_videoclips合并音频文件
    final_audio_clip = CompositeAudioClip(audio_clips)
    # 将音频添加到视频中
    final_clip = final_video_clip.set_audio(final_audio_clip)
    #final_clip = final_clip.set_fps(30)
    final_clip.write_videofile(outputpath, codec='libx264', threads=threads)


def generate_video(combined_video_path: str, tts_path: str, subtitles_path: str, threads: int, subtitles_position: str,  text_color : str):
    """
    This function creates the final video, with subtitles and audio.

    Args:
        combined_video_path (str): The path to the combined video.
        tts_path (str): The path to the text-to-speech audio.
        subtitles_path (str): The path to the subtitles.
        threads (int): The number of threads to use for the video processing.
        subtitles_position (str): The position of the subtitles.

    Returns:
        str: The path to the final video.
    """
    # Make a generator that returns a TextClip when called with consecutive
    generator = lambda txt: TextClip(
        txt,
        font="../fonts/bold_font.ttf",
        fontsize=100,
        color=text_color,
        stroke_color="black",
        stroke_width=5,
    )

    # Split the subtitles position into horizontal and vertical
    horizontal_subtitles_position, vertical_subtitles_position = subtitles_position.split(",")

    # Burn the subtitles into the video
    subtitles = SubtitlesClip(subtitles_path, generator)
    result = CompositeVideoClip([
        VideoFileClip(combined_video_path),
        subtitles.set_pos((horizontal_subtitles_position, vertical_subtitles_position))
    ])

    # Add the audio
    audio = AudioFileClip(tts_path)
    result = result.set_audio(audio)

    result.write_videofile("../temp/output.mp4", threads=threads or 2)

    return "output.mp4"

def video_fadein_fadeout(videopath,outputpath,fadein_second=1,fadeout_second=1):
    if fadein_second is None and fadeout_st is None:return
    fadeinstr=''
    fadeoutstr=''
    videoclip=VideoFileClip(videopath)
    duration=videoclip.duration
    if fadein_second is not None:
        fadeinstr=f'fade=in:0:d={fadein_second}'
    if fadeout_second is not None:
        fadeoutstr=f'fade=out:st={duration-fadeout_second}:d={fadeout_second}'
    if len(fadeinstr)>0 and len(fadeoutstr)>0:
        totalstr=f'{fadeinstr},{fadeoutstr}'
    else:
        totalstr=f'{fadeinstr}{fadeoutstr}'
    
    os.system(f'ffmpeg -y -i {videopath} -vf {totalstr} {outputpath}')


def image_to_video(imagepath,fps,seconds,outputpath,audiopath,vol=1,size=None):
    '''
    ffmpeg -r 25 -loop 1 -i ~/IMG_8679.JPG -pix_fmt yuv420p -vcodec libx264 -b:v 600k -r:v 25 -preset medium -crf 30 -s 720x576 -vframes 250 -r 25 -t 10 
    ffmpeg -r 15 -f image2 -loop 1 -i 图片.png -i 音频.mp3 -s 1920x1080 -pix_fmt yuvj420p -t 278 -vcodec libx264 输出.mp4
    参数介绍
        -r 15 为读取输入文件的时候帧率为15帧每秒
        -loop 1 ：因为只有一张图片所以必须加入这个参数（循环这张图片）
        -pix_fmt：指定图片输入格式(有yuv420,yuv444等各种格式)
        -t ：图片转换成视频的持续时长，单位是秒（S），必须指定该值，否则会无限制生成视频
        -s ：指定视频的分辨率
        -vcodec libx264：生成视频的编码格式，这里指定的是x264
    '''
    if size is None:
        os.system(f'ffmpeg -y -r {fps} -loop 1  -f image2 -i {imagepath} -i {audiopath} -pix_fmt yuvj420p -c:v libx264 -t {seconds} -af "volume={vol}" {outputpath}')
    else:
        os.system(f'ffmpeg -y -r {fps} -loop 1  -f image2 -i {imagepath} -i {audiopath} -s {size} -pix_fmt yuvj420p -c:v libx264 -t {seconds} -af "volume={vol}" {outputpath}')


def image_to_video_with_audio(imagepaths,fps,outputpath,audiopath,size=None):
    audio_clip = AudioFileClip(audiopath)
    seconds=audio_clip.duration

    imagepaths=imagepaths.split(';')
    if len(imagepaths)>1:
        t_duration=round(seconds/len(imagepaths),4)
        tmpfilename='tmpimagepaths.txt'
        tmpfile=open(tmpfilename,'w')
        for imagepath in imagepaths:
            print(f"file '{imagepath}'\nduration {t_duration}\n")
            tmpfile.write(f"file '{imagepath}'\nduration {t_duration}\n") 
        tmpfile.write(f"file '{imagepaths[-1]}'\nduration {t_duration}\n") 
        tmpfile.close()   
        tmpoutput=outputpath.replace('.mp4','_tmp.mp4')
        os.system(f'ffmpeg -y -f concat -safe 0 -i {tmpfilename} -r 30 -c:v libx264 -pix_fmt yuv420p {tmpoutput}')
        os.remove(tmpfilename)

        os.system(f'ffmpeg -y -i {tmpoutput} -i {audiopath} -c:v copy -c:a aac -map 0:v -map 1:a -shortest {outputpath}')
        os.remove(tmpoutput)
    else:
        imagepath=imagepaths[0]
        if size is None:
            os.system(f'ffmpeg -y -r {fps} -loop 1 -f image2 -i {imagepath} -i {audiopath} -pix_fmt yuvj420p -c:v libx264 -t {seconds} {outputpath}')
        else:
            os.system(f'ffmpeg -y -r {fps} -loop 1 -f image2 -i {imagepath} -i {audiopath} -s {size} -pix_fmt yuvj420p -c:v libx264 -t {seconds} {outputpath}')

def convert_vtt_srt_ast(subtitlefile1,subtitlefile2):
    '''
        *.vtt   -> *.srt
        *.lyric -> *.srt
        *.ass   -> *.srt
        *.srt   -> *.ass
        ...
    '''
    os.system(f'ffmpeg -y -i {subtitlefile1} {subtitlefile2}')

def add_hard_srt_to_video(videopath,subtitlefile,outputpath):
    '''
    ch_en.srt 
        1
        00:00:0,000 --> 00:00:5,000
        这是0到2秒显示的字幕
        Here is 0 to 2 seconds showed subtitle
        
        2
        00:00:2,000 --> 00:00:4,000
        这是2至4秒显示的字幕
        Here is 2 to 4 seconds showed subtitle
    '''
    if '.srt' in subtitlefile:
        #os.system(f'ffmpeg -y -i {videopath} -vf subtitles={subtitlefile} {outputpath}')
        os.system(f'''ffmpeg -y -i {videopath} -vf "subtitles={subtitlefile}" {outputpath}''')
        #os.system(f"""ffmpeg -y -i {videopath} -vf "subtitles={subtitlefile}:fontsdir=/mnt/glennge/MoneyPrinter/source/fonts" -c:v libx264 -crf 23 -c:a copy -metadata:s:s:0 language=chi {outputpath}""")
    elif '.ass' in subtitlefile:
        os.system(f'ffmpeg -y -i {videopath} -vf ass={subtitlefile} {outputpath}')

def add_soft_srt_to_video(videopath,subtitlefile,outputpath):
    os.system(f'ffmpeg -y -i {videopath} -i {subtitlefile} -c copy -c:s mov_text -metadata:s:s:0 language=chi {outputpath}')

def add_soft_multisrt_to_video(videopath,subtitlefile1,subtitlefile2,outputpath):
    '''
    ch_eng.srt 
        1
        00:00:0,000 --> 00:00:5,000
        这是0到2秒显示的字幕
        Here is 0 to 2 seconds showed subtitle
        
        2
        00:00:2,000 --> 00:00:4,000
        这是2至4秒显示的字幕
        Here is 2 to 4 seconds showed subtitle
    eng.srt
        1
        00:00:0,000 --> 00:00:5,000
        Here is 0 to 2 seconds showed subtitle
        
        2
        00:00:2,000 --> 00:00:4,000
        Here is 2 to 4 seconds showed subtitle
    '''
    os.system(f'ffmpeg -y -i {videopath} -i {subtitlefile1} -i {subtitlefile2} -map 0 -map 1 -map 2 -c copy -c:s mov_text -metadata:s:s:0 language=chi_eng -metadata:s:s:1 language=eng {outputpath}')

def soft_subtitle_to_hard_subtitle(videopath,outputpath):
    os.system(f'ffmpeg -y -i {videopath} -vf "subtitles={videopath}" -c:v libx264 -crf 23 -c:a copy {outputpath}')

def merge_video_audio(videopath,audiopath,outputpath):
    os.system(f'ffmpeg -y -i {videopath} -i {audiopath} -c:v copy -c:a copy {outputpath}')

def concate_videos(videopaths,outputpath):
    tmpfilename='tmpvideopaths.txt'
    tmpfile=open(tmpfilename,'w')
    for videopath in videopaths:
        tmpfile.write(f"file '{videopath}'\n") 
    tmpfile.close()   
    #os.system(f'ffmpeg -y -f concat -safe 0 -i {tmpfilename} -c copy -map 0:v -map 0:a -map 0:s {outputpath}')
    os.system(f'ffmpeg -y -f concat -safe 0 -i {tmpfilename} {outputpath}')
    os.remove(tmpfilename)


def add_imagevideo_watermark_to_video(videopath,imagevideopath,outputpath,position='topleft',x=0,y=0,
                                  alpha=1.0,
                                  startoffset=1,
                                  scale_x_y='100:100',
                                  endoffset=6):
    if position is not None:
        if position=='topleft':
            os.system(f"""ffmpeg -y -i {videopath} -i {imagevideopath} -filter_complex "scale={scale_x_y},overlay=x=0:y=0:alpha={alpha}:enable='between(t,{startoffset},{endoffset})'" {outputpath}""")
        elif position=='topright':
            os.system(f"""ffmpeg -y -i {videopath} -i {imagevideopath} -filter_complex "scale={scale_x_y},overlay=x=W-w:y=0:alpha={alpha}:enable='between(t,{startoffset},{endoffset})'" {outputpath}""")
        elif position=='bottomleft':
            os.system(f"""ffmpeg -y -i {videopath} -i {imagevideopath} -filter_complex "scale={scale_x_y},overlay=x=0:y=H-y:alpha={alpha}:enable='between(t,{startoffset},{endoffset})'" {outputpath}""")
        elif position=='bottomright':
            os.system(f"""ffmpeg -y -i {videopath} -i {imagevideopath} -filter_complex "scale={scale_x_y},overlay=x=W-w:y=H-y:alpha={alpha}:enable='between(t,{startoffset},{endoffset})'" {outputpath}""")
    else:
        os.system(f"""ffmpeg -y -i {videopath} -i {imagevideopath} -filter_complex "scale={scale_x_y},overlay=x={x}:y={y}:alpha={alpha}:enable='between(t,{startoffset},{endoffset})'" {outputpath}""")


def add_text_watermark_to_video(videopath,text,outputpath,position='topleft',x=0,y=0,
                                  fontsize=50,
                                  fontfile='FreeSerif.ttf',
                                  fontcolor='white',
                                  startoffset=1,
                                  endoffset=6):
    os.system(f"""ffmpeg -y -i {videopath} -vf "drawtext=fontsize={fontsize}:fontfile={fontfile}:text='{text}':fontcolor={fontcolor}:x={x}:y={y}:enable='between(t,{startoffset},{endoffset})'" {outputpath}""")


def add_bgm_to_video(videopath,audiopath,outputpath,v0=1,v1=0.5):
    #os.system(f"""ffmpeg -y -i {videopath} -stream_loop -1 -i {audiopath} -filter_complex ”[0:a]volume={v0}[a0];[1:a]volume={v1}[a1];[a0][a1]amix=inputs=2[a]“ {outputpath}""")
    os.system(f'''ffmpeg -y -i {videopath} -stream_loop -1 -i {audiopath} -filter_complex "[0:a]volume={v0}[a0];[1:a]volume={v1}[a1];[a0][a1]amix=inputs=2:duration=first:dropout_transition=1[a]" -map 0:v -map "[a]" -c:v copy -c:a aac {outputpath}''')

def video_add_audio_and_length(videofile,audiofile,outputvideopath):
    pass

def video_add_start_endimg(videofile,outputvideopath,savename,bgm,startimg=None,endimg=None,second=5):
    allvideofiles=[]

    if startimg is not None:
        image_to_video(startimg,12,second,outputvideopath+'start.mp4',bgm,vol=0.0001)
        video_fadein_fadeout(outputvideopath+'start.mp4',outputvideopath+'start_fade.mp4',1,1)
        allvideofiles.append(outputvideopath+'start_fade.mp4')
        os.remove(outputvideopath+'start.mp4')
    allvideofiles.append(videofile)
    if endimg is not None:
        image_to_video(endimg,12,second,outputvideopath+'end.mp4',bgm,vol=0.0001)
        video_fadein_fadeout(outputvideopath+'end.mp4',outputvideopath+'end_fade.mp4',1,1)
        allvideofiles.append(outputvideopath+'end_fade.mp4')
        os.remove(outputvideopath+'end.mp4')
    if len(allvideofiles)>1:
        #concate_videos(allvideofiles,outputvideopath+savename,True) #有问题
        combine_videos_simple(allvideofiles,outputvideopath+savename)
    else:
        shutil.copy(videofile,outputvideopath+savename)

def video_add_start_endimg2(videofile,outputvideopath,savename,startimg,endimg,second=5,bgm=None):    
    # 使用ImageClip加载图片文件，并设置持续时间为5秒
    #start_image_clip = ImageClip(startimg, duration=second)
    #end_image_clip = ImageClip(endimg, duration=second)
    image_to_video(startimg,12,second,outputvideopath+'start.mp4',bgm,vol=0.0001)
    start_image_clip=VideoFileClip(outputvideopath+'start.mp4')
    image_to_video(endimg,12,second,outputvideopath+'end.mp4',bgm,vol=0.0001)
    end_image_clip=VideoFileClip(outputvideopath+'end.mp4')
    
    # 使用VideoFileClip加载视频文件
    video_clip = VideoFileClip(videofile)
    # 将图片片段与视频片段合并
    final_clip = concatenate_videoclips([start_image_clip,video_clip,end_image_clip])

    # 保存合并后的视频文件
    final_clip.write_videofile(outputvideopath+savename, codec='libx264')

def add_bgm_to_video2(videopath,audiopath,outputpath,va=0.5):
    # 加载视频和音频文件
    video_clip = VideoFileClip(videopath)
    audio_clip = AudioFileClip(audiopath)
    # 获取视频和音频的长度
    video_duration = video_clip.duration
    audio_duration = audio_clip.duration

    # 根据视频和音频长度调整音频
    if video_duration > audio_duration:
        extend_duration=video_duration-audio_duration
        # 计算需要循环的次数
        loop_times = int(extend_duration / audio_duration) + 1
        # 创建一个与需要延长时间相等的音频片段，其内容是原始音频的循环
        extended_audio = concatenate_audioclips([audio_clip] * loop_times)
        # 如果视频长度大于音频长度，让背景音频循环
        adjusted_audio = concatenate_audioclips([audio_clip, extended_audio.subclip(0, extend_duration)])
    else:
        # 如果视频长度小于音频长度，截取和视频长度一致的音频
        adjusted_audio = audio_clip.subclip(0, video_duration)

    # 调整音频音量
    adjusted_audio = adjusted_audio.volumex(va)  # 将音量调整为原音量的50%

    # 为背景音频添加渐入渐出效果
    fade_in_duration = 2  # 渐入时长，单位为秒
    fade_out_duration = 2  # 渐出时长，单位为秒
    adjusted_audio = adjusted_audio.audio_fadein(fade_in_duration).audio_fadeout(fade_out_duration)

    # 将原有音频与调整后的背景音频合并
    combined_audio = CompositeAudioClip([video_clip.audio, adjusted_audio])
    # 将合并后的音频与视频合并
    final_clip = video_clip.set_audio(combined_audio)
    # 导出合并后的视频
    final_clip.write_videofile(outputpath, codec='libx264')

def add_text_watermask(videopath,outputpath,starttext='',storyname='',endtext='',second=5):
    def get_watermask(text,starttime=0,endtime=5,fontsize=96,color='white',position='center'):
        # 添加文字水印
        watermark = TextClip(
            txt=text,  # 水印文本
            font="/mnt/glennge/MoneyPrinter/source/fonts/Dengb.ttf",  # 字体文件的路径
            fontsize=fontsize,  # 字体大小
            color=color,  # 文本颜色
            stroke_color='black',
            )
        watermark = watermark.set_position(position)  # 例如：(x,y) x: left center right, y: top middle down
        watermark = watermark.set_start(starttime).set_end(endtime)
        return watermark
    video_clip = VideoFileClip(videopath)
    duration=video_clip.duration

    start_watermask=get_watermask(starttext,0,second,position=lambda t: ('center', t+512-104))
    storyname_watermask=get_watermask(storyname,0,second,color='yellow',position=lambda t: ('center', t+512))
    middle_watermask=get_watermask('公众号:进击智能的雨林君',0,duration,fontsize=24,position=('right','top')) 
    end_watermask=get_watermask(endtext,duration-5,duration)
    # 将文字水印添加到视频上
    final_clip = CompositeVideoClip([video_clip, start_watermask,storyname_watermask,middle_watermask,end_watermask], size=video_clip.size)
    #final_clip = CompositeVideoClip([video_clip, start_watermask,middle_watermask,end_watermask], size=video_clip.size)
    final_clip.write_videofile(outputpath, codec="libx264")
    # 释放资源
    final_clip.close()
    video_clip.close()


class CombineVideo():
    def __init__(self):
        pass
    
    def image_to_video_with_audio_subtitle(self,imagefiles,audiofile,subtitlefile,outputvideopath,savename,fadetime=1):
        print(imagefiles,audiofile,subtitlefile,outputvideopath,savename,fadetime)
        image_to_video_with_audio(imagefiles,12,outputvideopath+'tmp_audio.mp4',audiofile)
        add_soft_srt_to_video(outputvideopath+'tmp_audio.mp4',subtitlefile,outputvideopath+'tmp_subtitle.mp4')
        soft_subtitle_to_hard_subtitle(outputvideopath+'tmp_subtitle.mp4',outputvideopath+'tmp_subtitle_hard.mp4')
        #add_hard_srt_to_video(outputvideopath+'tmp_audio.mp4',subtitlefile,outputvideopath+'tmp_subtitle.mp4')
        os.remove(outputvideopath+'tmp_audio.mp4')
        os.remove(outputvideopath+'tmp_subtitle.mp4')
        video_fadein_fadeout(outputvideopath+'tmp_subtitle_hard.mp4',outputvideopath+savename,fadetime,fadetime)
        os.remove(outputvideopath+'tmp_subtitle_hard.mp4')
        return outputvideopath+savename
    
    def images_to_video_with_audios_subtitles(self,imagefiles,audiofiles,subtitlefiles,outputvideopath,savename):
        videopaths=[]
        for i,(imagefile,audiofile,subtitlefile) in enumerate(imagefiles,audiofiles,subtitlefiles):
            videopath=self.image_to_video_with_audio_subtitle(imagefile,audiofile,subtitlefile,outputvideopath,f'{i}_{savename}')
            videopaths.append(videopath)
        concate_videos(videopaths,outputvideopath+savename)
        
    def video_audio_subtitle_to_video(self,videofile,audiofile,subtitlefile,outputvideopath,savename):
        video_add_audio_and_length(videofile,audiofile,outputvideopath+'tmp_video_audio.mp4')
        add_soft_srt_to_video(outputvideopath+'tmp_video_audio.mp4',subtitlefile,outputvideopath+'tmp_video_audio_subtitle.mp4')
        soft_subtitle_to_hard_subtitle(outputvideopath+'tmp_video_audio_subtitle.mp4',outputvideopath+'tmp_video_audio_subtitle_hard.mp4')
        os.remove(outputvideopath+'tmp_video_audio.mp4')
        os.remove(outputvideopath+'tmp_video_audio_subtitle.mp4')
        video_fadein_fadeout(outputvideopath+'tmp_video_audio_subtitle_hard.mp4',outputvideopath+savename,1,1)
        os.remove(outputvideopath+'tmp_video_audio_subtitle_hard.mp4')
        return outputvideopath+savename
    
    def videos_audios_subtitles_to_video(self,videofiles,audiofiles,subtitlefiles,outputvideopath,savename):
        videopaths=[]
        for i,(videofile,audiofile,subtitlefile) in enumerate(videofiles,audiofiles,subtitlefiles):
            videopath=self.video_audio_subtitle_to_video(videofile,audiofile,subtitlefile,outputvideopath,f'{i}_{savename}')
            videopaths.append(videopath)
        concate_videos(videopaths,outputvideopath+savename)
    
    def generate_final_video(self,videoclipfiles,startimg,endimg,outputvideopath,savename,bgm,bgmvolume=0.5,starttext='',storyname='',endtext=''):
        concate_videos(videoclipfiles,outputvideopath+'tmpvideoclip.mp4')
        if startimg is not None and endimg is not None:
            #video_add_start_endimg(outputvideopath+'tmpvideoclip.mp4',outputvideopath,'start_videoclip_end.mp4',bgm,startimg=startimg,endimg=endimg,second=5)
            video_add_start_endimg2(outputvideopath+'tmpvideoclip.mp4',outputvideopath,'start_videoclip_end.mp4',startimg=startimg,endimg=endimg,second=5,bgm=bgm)
            add_text_watermask(outputvideopath+'start_videoclip_end.mp4',outputvideopath+'start_videoclip_end_watermask.mp4',starttext,storyname,endtext,5)
            #add_bgm_to_video(outputvideopath+'start_videoclip_end.mp4',bgm,outputvideopath+savename,v0=1,v1=bgmvolume)
            add_bgm_to_video2(outputvideopath+'start_videoclip_end_watermask.mp4',bgm,outputvideopath+savename,bgmvolume)
        else:
            #add_bgm_to_video(outputvideopath+'tmpvideoclip.mp4',bgm,outputvideopath+savename,v0=1,v1=bgmvolume)
            add_bgm_to_video2(outputvideopath+'start_videoclip_end.mp4',bgm,outputvideopath+savename,bgmvolume)
            
            




comb_video=CombineVideo()
        


if __name__=='__main__':
    #add_bgm_to_video('data/video/output.mp4','../source/Songs/output000.mp3','data/video/bgm.mp4')
    #add_soft_srt_to_video('data/video/video_0.mp4','data/audio/audio_0.srt','data/video/tmp_subtitle.mp4')
    #soft_subtitle_to_hard_subtitle('data/video/tmp_subtitle.mp4','data/video/tmp_subtitle_hard.mp4')

    #combine_videos_simple(['data/video/start_fade.mp4','data/video/tmpvideoclip.mp4','data/video/end_fade.mp4'],'data/video/output.mp4')

    #video_add_start_endimg2('data/video/tmpvideoclip.mp4','data/video/','output.mp4','data/image/boy.png','data/image/boy.png',5,'../source/Songs/output000.mp3')
    #add_text_watermask('data/video/output.mp4','data/video/output_watermask.mp4','龙宝的睡前故事','晚安好梦')
    #add_bgm_to_video2('data/video/output_watermask.mp4','../source/Songs/output000.mp3','data/video/bgm.mp4',0.8)

    add_text_watermask('data/video/video_0.mp4','data/video/video_0_watermask.mp4','龙宝的睡前故事','--井底之蛙','晚安好梦',5)
    
    #image_to_video_with_audio(f'data/image/boy.png;data/image/room.png;data/image/boy.png',25,f'data/video/output.mp4',f'data/audio/audio_0.mp3')

    #concate_videos(['data/video/video_0.mp4','data/video/video_1.mp4','data/video/video_2.mp4'],'data/video/output.mp4')

    #concate_videos(['data/video/start_fade.mp4','data/video/tmpvideoclip.mp4','data/video/end_fade.mp4'],'data/video/output.mp4')

    #image_to_video('data/image/boy.png',12,5,'data/video/start.mp4','../source/Songs/output000.mp3')

    #add_bgm_to_video('data/video/allscript.mp4','../source/Songs/output000.mp3','bgm.mp4')

    #add_bgm_to_video('data/video/output.mp4','../source/Songs/output000.mp3','bgm.mp4')

    #add_hard_srt_to_video('data/video/video_0.mp4','data/audio/audio_0.srt','tmp_subtitle.mp4')
    #add_soft_srt_to_video('data/video/video_0.mp4','data/audio/audio_0.srt','tmp_subtitle.mp4')
    #image_to_video('data/image/儿童绘本-白雪公主的秘密冒险01_00001_.png',25,5,'data/video/output1.mp4')
    #image_to_video('data/image/儿童绘本-白雪公主的秘密冒险02_00001_.png',25,5,'data/video/output2.mp4')
    sys.exit()
    #video_fadein_fadeout('data/video/output1.mp4','data/video/output1_fadeinout.mp4',1,1s)
    videopaths=[]
    for idx in range(1,17):
        if idx<10:
            idx_s=f'0{idx}'
        else:
            idx_s=f'{idx}'
        image_to_video_with_audio(f'data/image/儿童绘本-白雪公主的秘密冒险{idx_s}_00001_.png',25,f'data/video/script{idx}.mp4',f'data/audio/script{idx}.mp3')
        add_soft_srt_to_video(f'data/video/script{idx}.mp4',f'data/audio/script{idx}.srt',f'data/video/subtitle_script{idx}.mp4')
        os.remove(f'data/video/script{idx}.mp4')
        video_fadein_fadeout(f'data/video/subtitle_script{idx}.mp4',f'data/video/fade_script{idx}.mp4',1,1)
        os.remove(f'data/video/subtitle_script{idx}.mp4')
        videopaths.append(f'data/video/fade_script{idx}.mp4')
    concate_videos(videopaths,'data/video/allscript.mp4')
    pass
