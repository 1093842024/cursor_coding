import time,os,sys,datetime,random,json,cv2,shutil
import numpy as np
from PIL import Image,ImageDraw
import requests as req
from io import BytesIO
import pandas as pd
import hashlib
import torch
import re
import html
from urllib import parse
import requests
import gradio as gr

from transformers import pipeline, set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer

sys_prompt_all={
'儿童作家':'''你是一个畅销儿童故事的作者，故事往往情节精彩、引人入胜、让小朋友十分喜欢。
请以如下主题创作一个儿童故事剧本分镜，其中每一个分镜需要包含 场景名称（字数在4～6个字）和 详细内容（字数在50个字左右）。
按照以下json格式返回: 
{
    "分镜1":
        {
            "场景名称":"xxx",
            "详细内容":"xxx"
        },
    "分镜2":
        {
            "场景名称":"xxx",
            "详细内容":"xxx"
        }

}
整个剧本需要有5～10个分镜组成。
''',
}

common_prompt=[None,'请将分析以下故事剧本中的“详细内容”，每一个分镜用一句话总结主要人物的身份、外貌、行为和所处场景等关键信息。']


def call_ollama(prompt='你是谁',model='qwen:4b',systemprompt=None,stream=True):
    url='http://localhost:11434/api/generate' 
    data={"model": f"{model}","prompt": f"{prompt}","stream": stream}
    if systemprompt is not None and systemprompt in sys_prompt_all:
        data['system']=sys_prompt_all[systemprompt]
    if stream:
        allcontent=''
        response=req.post(url,json=data,stream=True)
        for i,line in enumerate(response.iter_lines(decode_unicode=True)):
            #print(i,line)
            try:
                tmpjson=json.loads(line)
                allcontent+=tmpjson['response']
                #print(i,allcontent)
            except:
                print('error line:',line)
            if allcontent[-1]=='\n':
                yield allcontent[:-1]
            else:
                yield allcontent
    else:
        response=req.post(url,json=data)
        if response.status_code==200:
            response_text=json.loads(response.text)['response']
            print(response_text)
            return response_text
        else:
            print(response)
            return 'response code not 200'

#curl http://localhost:11434/api/generate -d '{"model": "qwen:14b","prompt": "Why is the sky blue?","stream": true}'

'''
# 1. google translation API tool

'''
def google_translate(text, to_language="auto", text_language="auto"):
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

def test_google_translate():
    text = "祝愿世界和平"
    print(translate(text, "en", "zh-CN"))
    print(translate(text, "zh-CN", "en"))


class prompt_generater():
    def __init__(self,model='promptist'):
        self.model=model
        if self.model=='magic-prompt':
            self.gpt2_pipe = pipeline('text-generation', 
                         model='/Users/glennge/.cache/huggingface/hub/models--Gustavosta--MagicPrompt-Stable-Diffusion/snapshots/c2dfdbff1007791b5952aff9c02e622a0461f914/',
                         #model='Gustavosta/MagicPrompt-Stable-Diffusion', 
                         tokenizer='/Users/glennge/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/',
                         #'gpt2',
                         )
        elif self.model == 'promptist': # train for Stable Diffusion v1-4 
            self.prompter_model = AutoModelForCausalLM.from_pretrained('/Users/glennge/.cache/huggingface/hub/models--microsoft--Promptist/snapshots/cffa2f868729f40ddf7a51fed692e592349f596c/')#"microsoft/Promptist")
            self.prompter_tokenizer = AutoTokenizer.from_pretrained('/Users/glennge/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/')#"gpt2")
            self.prompter_tokenizer.pad_token = self.prompter_tokenizer.eos_token
            self.prompter_tokenizer.padding_side = "left"
            
        
    def generate(self,starting_text,max_new_tokens=75, num_beams=8, num_return_sequences=5, length_penalty=-1.0):
        seed = random.randint(100, 1000000)
        set_seed(seed)
        
        if self.model=='magic-prompt':
            starting_text = re.sub(r"[,:\-–.!;?_]", '', starting_text)
            
            #response = self.gpt2_pipe(starting_text, max_length=(len(starting_text) + random.randint(60, 90)), num_return_sequences=4)
            response = self.gpt2_pipe(starting_text, max_length=(len(starting_text) + max_new_tokens), num_return_sequences=num_return_sequences)
            response_list = []
            for x in response:
                resp = x['generated_text'].strip()
                if resp != starting_text and len(resp) > (len(starting_text) + 4) and resp.endswith((":", "-", "—")) is False:
                    response_list.append(resp+'\n')

            response_end = "\n".join(response_list)
            response_end = re.sub('[^ ]+\.[^ ]+','', response_end)
            response_end = response_end.replace("<", "").replace(">", "")
            print(response_end)
        else:
            input_ids = self.prompter_tokenizer(starting_text.strip()+" Rephrase:", return_tensors="pt").input_ids
            eos_id = self.prompter_tokenizer.eos_token_id
            outputs = self.prompter_model.generate(input_ids, 
                                                   do_sample=False,
                                                    max_new_tokens=max_new_tokens,
                                                    num_beams=num_beams,
                                                    num_return_sequences=num_return_sequences,
                                                    eos_token_id=eos_id,
                                                    pad_token_id=eos_id,
                                                    length_penalty=length_penalty)
            output_texts = self.prompter_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            result = []
            for output_text in output_texts:
                result.append(output_text.replace(starting_text + " Rephrase:", "").strip())
            response_end = "\n".join(result)           
        return response_end
    
    def generate_multiline(self,input_lines,max_new_tokens=75, num_beams=8, num_return_sequences=5, length_penalty=-1.0):
        allresponse=''
        for line in input_lines:
            response=self.generate(line,max_new_tokens, num_beams, num_return_sequences, length_penalty)
            allresponse+=response
            yield allresponse
            allresponse+='\n'





def concepts_to_sd_prompt_demo(pt_generater):
    def dropdown_select(inputstr,llm_output):
        return inputstr+'\n'+llm_output
    
    def translate_theme_cont(inputstr):
        theme_str=''
        content_str=''
        try:
            inputjson=json.loads(inputstr)
        except Exception as e:
            print(inputstr)
            print(e)
            return 'json parse failed','json parse failed'
        for f_id,value in inputjson.items():
            try:
                theme=value['场景名称']
                content=value['详细内容']
                theme_str+=theme+'\n'+google_translate(theme,'en','zh-CN')+'\n'
                content_str+=content+'\n'+google_translate(content,'en','zh-CN')+'\n'
            except:
                return '场景名称 parse failed','详细内容 parse failed'
        return theme_str[:-1],content_str[:-1]
            
        
    
    with gr.Blocks() as block:    
        with gr.Accordion('LLM参数', open=False):
            with gr.Row():
                modeltype=gr.Dropdown(choices=['qwen:4b'],value='qwen:4b',label='model',scale=0)
        with gr.Row():
            with gr.Column():
                with gr.Tab('大纲生成'):
                #with gr.Column():
                    sys_prompt=gr.Dropdown(choices=[None,'儿童作家'],value='儿童作家',label='system prompt模版')
                    with gr.Row():
                        llm_input1=gr.Textbox(label='主题大纲-LLM输入',lines=2,max_lines=2)
                        submit1=gr.Button('大纲生成', variant="primary",scale=0,visible=True)
                    llm_output1=gr.Textbox(label='大纲-LLM输出',show_label=True,show_copy_button=True,lines=8,max_lines=8)
                    translate_them_content = gr.Button('分镜场景与内容翻译与生成', variant="primary")
                llm_input1.submit(call_ollama,inputs=[llm_input1,modeltype,sys_prompt],outputs=[llm_output1]) 
                submit1.click(call_ollama,inputs=[llm_input1,modeltype,sys_prompt],outputs=[llm_output1])  
                with gr.Tab('画面描述分析'):
                #with gr.Column():
                    analyze_prompt=gr.Dropdown(choices=common_prompt,value=common_prompt[1],label='analyze prompt模版')
                    with gr.Row():
                        llm_input2=gr.Textbox(label='画面分析-LLM输入',lines=3,max_lines=3)
                        submit2=gr.Button('画面分析', variant="primary",scale=0,visible=True) 
                    llm_output2=gr.Textbox(label='画面分析-LLM输出',show_label=True,show_copy_button=True,lines=8,max_lines=8)
                analyze_prompt.select(dropdown_select,inputs=[analyze_prompt,llm_output1],outputs=[llm_input2])
                llm_input2.submit(call_ollama,inputs=[llm_input2,modeltype],outputs=[llm_output2]) 
                submit2.click(call_ollama,inputs=[llm_input2,modeltype],outputs=[llm_output2])
        
            with gr.Column():
                with gr.Tab('综合prompt翻译与优化'):
                    with gr.Row():
                        submit3=gr.Button('画面描述copy', variant="primary",scale=0,visible=True) 
                        llm_input3=gr.Textbox(label='二次处理-LLM输入',lines=3,max_lines=3)
                    llm_output3=gr.Textbox(label='二次分析-LLM输出',show_label=True,show_copy_button=True,lines=8,max_lines=8)
                    submit3.click(dropdown_select,inputs=[llm_input3,llm_output2],outputs=[llm_input3]) 
                    llm_input3.submit(call_ollama,inputs=[llm_input3,modeltype],outputs=[llm_output3])
                    translate_generate_btn = gr.Button('综合prompt翻译与生成', variant="primary") 
                with gr.Tab('单条prompt翻译与优化'):
                    with gr.Row():
                        input_text = gr.Textbox(lines=1, label='内容描述')
                        translate_output = gr.Textbox(lines=1, label='内容描述翻译结果')
                    with gr.Row():
                        from_lang = gr.Dropdown(choices=['zh-CN','en',], value='zh-CN',label="输入语言")
                        to_lang =gr.Dropdown(choices=['zh-CN','en',], value='en',label="目标语言")
                        translate_btn = gr.Button('翻译',scale=0) 
                        generate_prompter_btn = gr.Button('prompt优化生成', variant="primary")                    
                    generate_prompter_output = gr.Textbox(lines=5, label='生成的Prompt')
                    translate_btn.click(fn=google_translate,inputs=[input_text,to_lang,from_lang],outputs=[translate_output])                
                with gr.Row():
                    with gr.Accordion('prompt 生成参数设置', open=False):
                        max_new_tokens = gr.Slider(1, 255, 75, label='max_new_tokens', step=1)
                        nub_beams = gr.Slider(1, 30, 8, label='num_beams', step=1)
                        num_return_sequences = gr.Slider(1, 30, 5, label='num_return_sequences', step=1)
                        length_penalty = gr.Slider(-1.0, 1.0, -1.0, label='length_penalty')
                       
                generate_prompter_btn.click(
                        fn=pt_generater.generate,
                        inputs=[translate_output, max_new_tokens, nub_beams, num_return_sequences, length_penalty],
                        outputs=[generate_prompter_output])         
        with gr.Row():
            theme = gr.Textbox(lines=5,max_lines=8, label='剧本分镜场景',show_label=True,show_copy_button=True,scale=1.5)
            content_description = gr.Textbox(lines=5,max_lines=8, label='剧本分镜内容描述',show_label=True,show_copy_button=True,scale=4)
            prompt_eng = gr.Textbox(lines=5,max_lines=8, label='剧本分镜SD prompt',show_label=True,show_copy_button=True,scale=4)
        translate_them_content.click(translate_theme_cont,inputs=[llm_output1],outputs=[theme,content_description])
        translate_generate_btn.click(pt_generater.generate_multiline,inputs=[llm_output3, max_new_tokens, nub_beams, num_return_sequences, length_penalty],outputs=[prompt_eng])
                    
    return block


if __name__ == '__main__':
    #call_ollama(stream=True)
    #sys.exit()

    
    #pt_generator=prompt_generater('fake')
    pt_generator=prompt_generater('promptist')
    concepts_to_sd_prompt_demo(pt_generator).queue(concurrency_count=5, max_size=5).launch(show_api=True,
                                                           enable_queue=True, 
                                                           debug=True, 
                                                           share=False, 
                                                           server_name='0.0.0.0')
    
    pass