import logging
import re
import json
from typing import List
import g4f
from loguru import logger
from openai import OpenAI


def _generate_response(prompt: str):
    content = ""
    llm_provider = 'g4f'
    logger.info(f"llm provider: {llm_provider}")
    if llm_provider == "g4f":
        model_name = ""
        if not model_name:
            model_name = "gpt-3.5-turbo-16k-0613"

        content = g4f.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
    else:
        if llm_provider == "moonshot":
            api_key = ''
            model_name = ''
            base_url = "https://api.moonshot.cn/v1"
        elif llm_provider == "openai":
            api_key = ''
            model_name = ''
            base_url = "https://api.openai.com/v1"
        elif llm_provider == "oneapi":
            api_key = ''
            model_name = ''
            base_url = ''
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


def generate_script(video_subject: str, language: str = "zh-CN", paragraph_number: int = 1):
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

    prompt2 = """
            Generate a script for a video, depending on the subject of the video.

            The script is to be returned as a string with the specified number of paragraphs.

            Here is an example of a string:
            "This is an example string."

            Do not under any circumstance reference this prompt in your response.

            Get straight to the point, don't start with unnecessary things like, "welcome to this video".

            Obviously, the script should be related to the subject of the video.

            YOU MUST NOT INCLUDE ANY TYPE OF MARKDOWN OR FORMATTING IN THE SCRIPT, NEVER USE A TITLE.
            YOU MUST WRITE THE SCRIPT IN THE LANGUAGE SPECIFIED IN [LANGUAGE].
            ONLY RETURN THE RAW CONTENT OF THE SCRIPT. DO NOT INCLUDE "VOICEOVER", "NARRATOR" OR SIMILAR INDICATORS OF WHAT SHOULD BE SPOKEN AT THE BEGINNING OF EACH PARAGRAPH OR LINE. YOU MUST NOT MENTION THE PROMPT, OR ANYTHING ABOUT THE SCRIPT ITSELF. ALSO, NEVER TALK ABOUT THE AMOUNT OF PARAGRAPHS OR LINES. JUST WRITE THE SCRIPT.

        """
    prompt2 += f"""
    
    Subject: {video_subject}
    Number of paragraphs: {paragraph_number}
    Language: {voice}

    """

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
        logging.error("gpt returned an empty response")

    logger.success(f"completed: \n{final_script}")
    return final_script


def generate_terms(video_subject: str, video_script: str, amount: int = 5) -> List[str]:
    prompt = f"""
# Role: Video Search Terms Generator

## Goals:
Generate {amount} search terms for stock videos, depending on the subject of a video.

## Constrains:
1. the search terms are to be returned as a json-array of strings.
2. each search term should consist of 1-3 words, always add the main subject of the video.
3. you must only return the json-array of strings. you must not return anything else. you must not return the script.
4. the search terms must be related to the subject of the video.
5. reply with english search terms only.

## Output Example:
["search term 1", "search term 2", "search term 3","search term 4","search term 5"]

## Context:
### Video Subject
{video_subject}

### Video Script
{video_script}
""".strip()

    prompt2 = f"""
    Generate {amount} search terms for stock videos,
    depending on the subject of a video.
    Subject: {video_subject}

    The search terms are to be returned as
    a JSON-Array of strings.

    Each search term should consist of 1-3 words,
    always add the main subject of the video.
    
    YOU MUST ONLY RETURN THE JSON-ARRAY OF STRINGS.
    YOU MUST NOT RETURN ANYTHING ELSE. 
    YOU MUST NOT RETURN THE SCRIPT.
    
    The search terms must be related to the subject of the video.
    Here is an example of a JSON-Array of strings:
    ["search term 1", "search term 2", "search term 3"]

    For context, here is the full text:
    {script}
    """


    logger.info(f"subject: {video_subject}")
    logger.debug(f"prompt: \n{prompt}")
    response = _generate_response(prompt)
    search_terms = []

    try:
        search_terms = json.loads(response)
        if not isinstance(search_terms, list) or not all(isinstance(term, str) for term in search_terms):
            raise ValueError("response is not a list of strings.")

    except (json.JSONDecodeError, ValueError):
        # logger.warning(f"gpt returned an unformatted response. attempting to clean...")
        # Attempt to extract list-like string and convert to list
        match = re.search(r'\["(?:[^"\\]|\\.)*"(?:,\s*"[^"\\]*")*\]', response)
        if match:
            try:
                search_terms = json.loads(match.group())
            except json.JSONDecodeError:
                logger.error(f"could not parse response: {response}")
                return []

    logger.success(f"completed: \n{search_terms}")
    return search_terms


if __name__ == "__main__":
    video_subject = "生命的意义是什么"
    script = generate_script(video_subject=video_subject, language="zh-CN", paragraph_number=1)
    # print("######################")
    # print(script)
    # search_terms = generate_terms(video_subject=video_subject, video_script=script, amount=5)
    # print("######################")
    # print(search_terms)
