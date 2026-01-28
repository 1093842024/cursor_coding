import logging
import re
import os
from typing import List, Optional

import g4f
from openai import OpenAI

from config_loader import get_llm_config

logger = logging.getLogger(__name__)

# 开源/免费可选模型（如 g4f）
g4f_Available_model = [
    "gpt-3.5-turbo",
    "gpt-4", "gpt-4-turbo",
    "gemini", "gemini-pro",
    "claude-3-opus", "claude-3-sonnet",
]

# 商用 provider 默认 base_url / model（当环境变量未设置时使用的 fallback，仍不写密钥）
_DEFAULT_BASE_URL = {
    "moonshot": "https://api.moonshot.cn/v1",
    "glm": "https://open.bigmodel.cn/api/paas/v4/",
    "openai": "https://api.openai.com/v1",
    "coze": "https://api.coze.cn/open_api/v2/chat",
}
_DEFAULT_MODEL = {
    "moonshot": "moonshot-v1-32k",
    "glm": "GLM-4",
    "openai": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    "coze": "",
}


def _generate_response(
    prompt: str,
    llm_provider: str = "g4f",
    model_name: Optional[str] = None,
) -> str:
    """内部生成接口。商用 backend 的 api_key、base_url、model 从环境变量或 config 读取，禁止硬编码密钥。"""
    content = "no response"
    if llm_provider in ("g4f", "gpt3.5-turbo"):
        model_name = model_name or "gpt-3.5-turbo-16k-0613"
        content = g4f.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
    else:
        api_key, base_url, cfg_model = get_llm_config(llm_provider)
        base_url = base_url or _DEFAULT_BASE_URL.get(llm_provider, "")
        model_name = (model_name or "").strip() or cfg_model or _DEFAULT_MODEL.get(llm_provider, "")

        if not api_key:
            raise ValueError(
                f"{llm_provider}: api_key 未设置，请在环境变量或 .env 中配置 LLM_{llm_provider.upper().replace('-', '_')}_API_KEY"
            )
        if not model_name:
            raise ValueError(
                f"{llm_provider}: model_name 未设置，请在环境变量中配置 LLM_{llm_provider.upper().replace('-', '_')}_MODEL"
            )
        if not base_url:
            raise ValueError(
                f"{llm_provider}: base_url 未设置，请配置 LLM_{llm_provider.upper().replace('-', '_')}_BASE_URL"
            )

        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        if response:
            content = response.choices[0].message.content or content

    return content


# ---- 阶段1：主题/梗概扩写与分段提炼 ----
EXPAND_STORY_PROMPT = """你是一名畅销儿童故事作者。请根据下列主题或故事梗概，扩写为一篇完整的儿童故事。
要求：
1. 字数300～500字，分为8～10个段落，每段之间用两个换行符分隔。
2. 情节清晰、简单易懂、引人入胜，适合小朋友阅读。
3. 只输出故事正文，不要输出标题、序号或任何额外说明。
4. 每段用两个换行符（空一行）分隔。

主题或梗概：
{outline}
"""


def expand_story_from_outline(outline: str, llm_provider: str = "moonshot", model_name: str = None) -> str:
    """根据主题或梗概扩写为完整故事。"""
    if not outline or not outline.strip():
        return ""
    prompt = EXPAND_STORY_PROMPT.format(outline=outline.strip())
    return _generate_response(prompt, llm_provider, model_name or "")


SUMMARIZE_SEGMENT_PROMPT = """请用50～100字概括下列故事片段的主要内容，保留关键人物、场景和情节发展。只输出概括文字，不要序号或标题。

故事片段：
{content}
"""


def summarize_segment(segment_content: str, llm_provider: str = "moonshot", model_name: str = None) -> str:
    """对单个故事段落做提炼概括。"""
    if not segment_content or not segment_content.strip():
        return ""
    prompt = SUMMARIZE_SEGMENT_PROMPT.format(content=segment_content.strip())
    return _generate_response(prompt, llm_provider, model_name or "")


# ---- 阶段2：全局与分段角色/场景抽取（含详细描述） ----
PERSON_SCENE_DETAILED_PROMPT = """请根据以下故事内容，总结故事的主要角色和主要场景，并给出详细描述。严格按照下列格式输出（不要添加额外说明）：

[Characters]
角色i名称:xxx;外形描述:xxxx;详细描述:xxxx
（外形描述与详细描述均为英文。外形描述用于生成角色画面，侧重外貌、穿着；详细描述补充外貌、服装、标志特征等，便于后续多视角生成。）

[Scenes]
场景j名称:xxx;画面描述:xxxx;详细描述:xxxx
（画面描述与详细描述均为英文。画面描述用于生成场景图；详细描述补充地点、时间、氛围、关键物体等。）

其中 i、j 从 0 开始。角色名、场景名为中文。

Story content:
'''
{story}
'''
"""


def extract_characters_scenes_detailed(full_story: str, llm_provider: str = "moonshot", model_name: str = None) -> str:
    """全局抽取角色与场景，输出带详细描述的格式，与 Default_person_scene_prompt 兼容并增强。"""
    if not full_story or not full_story.strip():
        return ""
    prompt = PERSON_SCENE_DETAILED_PROMPT.format(story=full_story.strip())
    return _generate_response(prompt, llm_provider, model_name or "")


SEGMENT_CHARACTERS_SCENES_PROMPT = """以下是一段故事内容，以及本故事中出现的全部角色名、全部场景名。请判断这段内容中涉及到了哪些角色、哪些场景，只输出两个列表，格式如下（每行一个名称，与下面给出的名称完全一致）：

涉及角色：
角色名1
角色名2

涉及场景：
场景名1
场景名2

若某段没有涉及任何角色或场景，对应部分写「无」。

全部角色名：{character_names}
全部场景名：{scene_names}

本段故事：
'''
{content}
'''
"""


def get_segment_characters_scenes(
    segment_content: str,
    character_names: List[str],
    scene_names: List[str],
    llm_provider: str = "moonshot",
    model_name: str = None,
) -> tuple:
    """返回本段涉及的角色名列表、场景名列表（均为全局名称的子集）。"""
    if not segment_content or not segment_content.strip():
        return [], []
    cstr = "、".join(character_names) if character_names else "无"
    sstr = "、".join(scene_names) if scene_names else "无"
    prompt = SEGMENT_CHARACTERS_SCENES_PROMPT.format(
        character_names=cstr, scene_names=sstr, content=segment_content.strip()
    )
    raw = _generate_response(prompt, llm_provider, model_name or "")
    char_out, scene_out = [], []
    if not raw:
        return char_out, scene_out
    block = ""
    for line in raw.split("\n"):
        line = line.strip()
        if "涉及角色" in line or line == "涉及角色：":
            block = "char"
            continue
        if "涉及场景" in line or line == "涉及场景：":
            block = "scene"
            continue
        if line and line != "无":
            if block == "char" and line in character_names:
                char_out.append(line)
            if block == "scene" and line in scene_names:
                scene_out.append(line)
    return char_out, scene_out


# ---- 阶段3：电影化分镜剧本（按段生成，含镜头类型与景别） ----
CINEMATIC_STORYBOARD_PROMPT = """你是一位专业的分镜师，请根据下列「本段故事」「本段涉及的角色与场景」设计电影化分镜。
要求：
1. 使用电影化技巧，例如：先出现环境声和背景画面再出现人物与动作；或先聚焦场景/角色局部，再徐徐移动到人物与事件。
2. 每条分镜需包含：旁白内容、角色、场景、画面prompt、镜头类型、景别。
3. 镜头类型仅从以下选一：环境先导、局部到全景、声画同步、特写切入、其他。
4. 景别仅从以下选一：特写、中景、全景、远景、其他。
5. 角色必须是给定的本段涉及角色之一或 None；场景必须是本段涉及场景之一。
6. 严格按照下列格式输出，每条分镜一行（不要换行符在行内）：
分镜i画面j->旁白内容:xxx;角色:xxx;场景:xxx;画面prompt:xxx;镜头类型:xxx;景别:xxx.

本段涉及角色：{character_names}
本段涉及场景：{scene_names}

本段故事：
'''
{content}
'''

输出分镜（每行一条）：
"""


def generate_cinematic_storyboard_for_segment(
    segment_content: str,
    character_names: List[str],
    scene_names: List[str],
    llm_provider: str = "moonshot",
    model_name: str = None,
) -> str:
    """对单段生成电影化分镜剧本（带镜头类型、景别）。"""
    if not segment_content or not segment_content.strip():
        return ""
    cstr = "、".join(character_names) if character_names else "无"
    sstr = "、".join(scene_names) if scene_names else "无"
    prompt = CINEMATIC_STORYBOARD_PROMPT.format(
        character_names=cstr, scene_names=sstr, content=segment_content.strip()
    )
    return _generate_response(prompt, llm_provider, model_name or "")


def generate_script(
    video_subject: str,
    llm_provider: str = "g4f",
    model_name: Optional[str] = None,
    language: str = "zh-CN",
    paragraph_number: int = 1,
) -> str:
    prompt = f"""
        # Role: Video Script Generator

        ## Goals:
        Generate a script for a video, depending on the subject of the video.

        ## Constrains:
        1. the script is to be returned as a string with the specified number of paragraphs.
        2. do not under any circumstance reference this prompt in your response.
        3. get straight to the point, don't start with unnecessary things like, "welcome to this video".
        4. you must not include any type of markdown or formatting in the script, never use a title. 
        5. only return the raw content of the script. 
        6. do not include "voiceover", "narrator" or similar indicators of what should be spoken at the beginning of each paragraph or line. 
        7. you must not mention the prompt, or anything about the script itself. also, never talk about the amount of paragraphs or lines. just write the script.

        ## Output Example:
        What is the meaning of life. This question has puzzled philosophers.

        # Initialization:
        - video subject: {video_subject}
        - output language: {language}
        - number of paragraphs: {paragraph_number}
        """.strip()

    final_script = ""
    logger.info(f"subject: {video_subject}")
    logger.debug(f"prompt: \n{prompt}")
    response = _generate_response(prompt, llm_provider=llm_provider, model_name=model_name or "")

    # Return the generated script
    if response:
        # Clean the script
        # Remove asterisks, hashes
        response = response.replace("*", "")
        response = response.replace("#", "")

        # Remove markdown syntax
        response = re.sub(r"\[.*\]", "", response)
        response = re.sub(r"\(.*\)", "", response)

        # Split the script into paragraphs
        paragraphs = response.split("\n\n")

        # Select the specified number of paragraphs
        selected_paragraphs = paragraphs[:paragraph_number]

        # Join the selected paragraphs into a single string
        final_script = "\n\n".join(selected_paragraphs)

        # Print to console the number of paragraphs used
        # logger.info(f"number of paragraphs used: {len(selected_paragraphs)}")
    else:
        print("gpt returned an empty response")
        final_script='error'

    return final_script



if __name__ == "__main__":
    video_subject = "生命的意义是什么"
    script = generate_script(video_subject=video_subject, 
                             llm='moonshot',
                             language="zh-CN", paragraph_number=1)
    print(script)
