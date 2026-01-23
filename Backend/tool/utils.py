import os
import threading
from typing import Any
from loguru import logger
import json
from uuid import uuid4
import urllib3
import sys,random
import logging
import zipfile
import requests
from termcolor import colored

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# TikTok Session ID
# Obtain your session ID by logging into TikTok and copying the sessionid cookie.
TIKTOK_SESSION_ID="689835499"

# ImageMagick Binary Path
# Download ImageMagick from https://imagemagick.org/script/download.php
IMAGEMAGICK_BINARY="/opt/homebrew/Cellar/imagemagick/7.1.1-29_1/bin/magick"

# Pexels API Key
# Register at https://www.pexels.com/api/ to get your API key.
PEXELS_API_KEY="sd0X5UD45qXzPr2yzQspDfwwBRUudile2fkYeqcbZmgw0Po0ObwV5r4x"

# Optional API Keys
# -----------------

# OpenAI API Key
# Visit https://openai.com/api/ for details on obtaining an API key.
OPENAI_API_KEY=""

# AssemblyAI API Key
# Sign up at https://www.assemblyai.com/ to receive an API key.
ASSEMBLY_AI_API_KEY=""

# Google API Key
# Generate your API key through https://makersuite.google.com/app/apikey
GOOGLE_API_KEY=""


def clean_dir(path: str):
    """
    Removes every file in a directory.

    Args:
        path (str): Path to directory.

    Returns:
        None
    """
    try:
        if not os.path.exists(path):
            os.mkdir(path)
            logger.info(f"Created directory: {path}")

        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            os.remove(file_path)
            logger.info(f"Removed file: {file_path}")

        logger.info(colored(f"Cleaned {path} directory", "green"))
    except Exception as e:
        logger.error(f"Error occurred while cleaning directory {path}: {str(e)}")

def fetch_songs(zip_url: str):
    """
    Downloads songs into songs/ directory to use with geneated videos.

    Args:
        zip_url (str): The URL to the zip file containing the songs.

    Returns:
        None
    """
    try:
        logger.info(colored(f" => Fetching songs...", "magenta"))

        files_dir = "../Songs"
        if not os.path.exists(files_dir):
            os.mkdir(files_dir)
            logger.info(colored(f"Created directory: {files_dir}", "green"))
        else:
            # Skip if songs are already downloaded
            return

        # Download songs
        response = requests.get(zip_url)

        # Save the zip file
        with open("../Songs/songs.zip", "wb") as file:
            file.write(response.content)

        # Unzip the file
        with zipfile.ZipFile("../Songs/songs.zip", "r") as file:
            file.extractall("../Songs")

        # Remove the zip file
        os.remove("../Songs/songs.zip")

        logger.info(colored(" => Downloaded Songs to ../Songs.", "green"))

    except Exception as e:
        logger.error(colored(f"Error occurred while fetching songs: {str(e)}", "red"))

def choose_random_song():
    """
    Chooses a random song from the songs/ directory.

    Returns:
        str: The path to the chosen song.
    """
    try:
        songs = os.listdir("../Songs")
        song = random.choice(songs)
        logger.info(colored(f"Chose song: {song}", "green"))
        return f"../Songs/{song}"
    except Exception as e:
        logger.error(colored(f"Error occurred while choosing random song: {str(e)}", "red"))



def get_response(status: int, data: Any = None, message: str = ""):
    obj = {
        'status': status,
    }
    if data:
        obj['data'] = data
    if message:
        obj['message'] = message
    return obj


def to_json(obj):
    # 定义一个辅助函数来处理不同类型的对象
    def serialize(o):
        # 如果对象是可序列化类型，直接返回
        if isinstance(o, (int, float, bool, str)) or o is None:
            return o
        # 如果对象是二进制数据，转换为base64编码的字符串
        elif isinstance(o, bytes):
            return "*** binary data ***"
        # 如果对象是字典，递归处理每个键值对
        elif isinstance(o, dict):
            return {k: serialize(v) for k, v in o.items()}
        # 如果对象是列表或元组，递归处理每个元素
        elif isinstance(o, (list, tuple)):
            return [serialize(item) for item in o]
        # 如果对象是自定义类型，尝试返回其__dict__属性
        elif hasattr(o, '__dict__'):
            return serialize(o.__dict__)
        # 其他情况返回None（或者可以选择抛出异常）
        else:
            return None

    # 使用serialize函数处理输入对象
    serialized_obj = serialize(obj)

    # 序列化处理后的对象为JSON字符串
    return json.dumps(serialized_obj, ensure_ascii=False, indent=4)


def get_uuid(remove_hyphen: bool = False):
    u = str(uuid4())
    if remove_hyphen:
        u = u.replace("-", "")
    return u


def root_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def storage_dir(sub_dir: str = ""):
    d = os.path.join(root_dir(), "storage")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    return d


def resource_dir(sub_dir: str = ""):
    d = os.path.join(root_dir(), "resource")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    return d


def task_dir(sub_dir: str = ""):
    d = os.path.join(storage_dir(), "tasks")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    if not os.path.exists(d):
        os.makedirs(d)
    return d


def font_dir(sub_dir: str = ""):
    d = resource_dir(f"fonts")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    if not os.path.exists(d):
        os.makedirs(d)
    return d


def song_dir(sub_dir: str = ""):
    d = resource_dir(f"songs")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    if not os.path.exists(d):
        os.makedirs(d)
    return d


def public_dir(sub_dir: str = ""):
    d = resource_dir(f"public")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    if not os.path.exists(d):
        os.makedirs(d)
    return d


def run_in_background(func, *args, **kwargs):
    def run():
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.error(f"run_in_background error: {e}")

    thread = threading.Thread(target=run)
    thread.start()
    return thread


def time_convert_seconds_to_hmsm(seconds) -> str:
    hours = int(seconds // 3600)
    seconds = seconds % 3600
    minutes = int(seconds // 60)
    milliseconds = int(seconds * 1000) % 1000
    seconds = int(seconds % 60)
    return "{:02d}:{:02d}:{:02d},{:03d}".format(hours, minutes, seconds, milliseconds)


def text_to_srt(idx: int, msg: str, start_time: float, end_time: float) -> str:
    start_time = time_convert_seconds_to_hmsm(start_time)
    end_time = time_convert_seconds_to_hmsm(end_time)
    srt = """%d
%s --> %s
%s
        """ % (
        idx,
        start_time,
        end_time,
        msg,
    )
    return srt


