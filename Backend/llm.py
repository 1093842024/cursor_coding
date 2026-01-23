import logging
import re
import json
from typing import List
import g4f
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

g4f_Available_model=[
    'gpt-3.5-turbo', #直接免费，其他需要apikey等
    'gpt-4',
    'gpt-4-turbo',
    'gemini',
    'gemini-pro',
    'claude-3-opus',
    'claude-3-sonnet',
]

def _generate_response(prompt,llm_provider = 'g4f',model_name = "gpt-3.5-turbo"):
    content = "no response"
    if llm_provider in ["g4f",'gpt3.5-turbo']:
        if not model_name:
            model_name = "gpt-3.5-turbo-16k-0613"
        content = g4f.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
    else:
        if llm_provider == "moonshot":
            api_key = 'sk-VVBB5uQDGFxoyzwMevT72IaDkOjZe98Y6qW5hExvV9LXwE6o'
            model_name = 'moonshot-v1-32k' #长文本
            #model_name = 'moonshot-v1-8k' #短文本
            #model_name = 'moonshot-v1-128k' #超长文本
            base_url = "https://api.moonshot.cn/v1"
        elif llm_provider=='glm':
            api_key='0e1530388ca6cbf062a1c152074c047c.aemcKXjxm8sQ2KgI'
            model_name='GLM-4' #GLM-4V  #CogView-3
            base_url='https://open.bigmodel.cn/api/paas/v4/'
        elif llm_provider == "openai":
            api_key = ''
            model_name = ''
            base_url = "https://api.openai.com/v1"
        elif llm_provider == "coze":
            api_key = 'pat_4tcoVnVfFJvqR6vOgLzYoeXFrfu4VOYDXGS1C76f8voI5ZMqVQpMLMJbbfAH61X5'
            model_name = ''
            base_url = "https://api.coze.cn/open_api/v2/chat"
        else:
            raise ValueError("llm_provider is not set, please set it in the config.toml file.")

        if not api_key:
            raise ValueError(f"{llm_provider}: api_key is not set, please set it in the config.toml file.")
        if not model_name:
            raise ValueError(f"{llm_provider}: model_name is not set, please set it in the config.toml file.")
        if not base_url:
            raise ValueError(f"{llm_provider}: base_url is not set, please set it in the config.toml file.")

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        if response:
            content = response.choices[0].message.content

    return content


def generate_script(video_subject, 
                    llm='g4f',
                    model='gpt-3.5-turbo',
                    language= "zh-CN", 
                    paragraph_number= 1):
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
    response = _generate_response(prompt=prompt)

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
