import torch,sys,os,datetime
from pathlib import Path
from PIL import Image
import numpy as np
import gradio as gr
from generate_image import sdxl_lightning_model,sd15_model
from iterate_generate_video import svd_model,animatedifflcm_model,video_to_gif,load_video_to_frames

os.environ["no_proxy"] = "localhost,0.0.0.0,:8082"

style_list = [
    {
        "name": "No Style",
        "prompt": "{prompt}",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
    },
    {
        "name": "Pixel art",
        "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
    },
    {
        "name": "Neonpunk",
        "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
    },
    {
        "name": "Manga",
        "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
    },
]
styles = {k["name"]: k["prompt"] for k in style_list}
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "No Style"
MAX_SEED = np.iinfo(np.int32).max
    

def generate_image_gr_demo(t2imodel,i2vmodel,t2vmodel):
    def get_select_image(gallary_image,select_gallary_image,evt: gr.SelectData):
        print(gallary_image)
        print(select_gallary_image)
        print(f'selected {evt.value} at {evt.index} from {evt.target}')
        select_imagelabel=evt.value['caption']
        select_imagepath=evt.value['image']['path']
        
        all_select_images=[]
        if select_gallary_image is not None:
            for gallary_data in select_gallary_image:
                imagepath,label=gallary_data
                if label!=select_imagelabel:
                    all_select_images.append((imagepath,label))
        all_select_images.append((select_imagepath,select_imagelabel))
        return all_select_images,select_imagepath
    
    def get_select_to_candi_image(select_gallary_image,evt: gr.SelectData):
        select_imagelabel=evt.value['caption']
        select_imagepath=evt.value['image']['path']
        return select_imagepath
    
    def image_generate_video(image,select_gallary_video,seedvideo,fps,num_iterations,videosavepath,i2vmodeltype):
        if image is None:gr.Error('请先上传/生成图片')
        if i2vmodeltype=='svd':
            all_generate_video=[]
            if select_gallary_video is not None:
                for gallary_data in select_gallary_video:
                    gifpath,label=gallary_data
                    all_generate_video.append((gifpath,label))
                yield all_generate_video,None
            videopath,gif_video = i2vmodel.generate_and_concatenate_videos(image,seedvideo,fps,num_iterations,videosavepath)
            all_generate_video.append((gif_video,'i2v'))
            print(all_generate_video,videopath)
            #torch.cuda.empty_cache()
            yield all_generate_video,videopath
        else:
            yield select_gallary_video,None

    
    def generate_text_video(t2v_prompt,neg_prompt,select_gallary_video,seed_t2v,fps_t2v,framenum_t2v,guidance_t2v,num_inference_steps,videosavepath,t2vmodeltype):
        if len(t2v_prompt)<1:gr.Error('请先输入视频输入prompt')
        if t2vmodeltype=='animtedifflcm':
            all_generate_video=[]
            if select_gallary_video is not None:
                for gallary_data in select_gallary_video:
                    gifpath,label=gallary_data
                    all_generate_video.append((gifpath,label))
            videopath,gif_video  = t2vmodel.generate_and_concatenate_videos(t2v_prompt,neg_prompt,seed_t2v,fps_t2v,framenum_t2v,guidance_t2v,num_inference_steps,videosavepath)
            all_generate_video.append((gif_video,t2v_prompt))
            print(all_generate_video,videopath)
            #torch.cuda.empty_cache()
            return all_generate_video,videopath
        else:        
            return select_gallary_video,None
    
    with gr.Blocks(title='安全管理部-视觉生成技术') as block:   
        neg_prompt=gr.Textbox(label='neg_prompt',value='(nipples:1.5), (worst quality:1.3), unfinished sketch, blurry, normal, mundane, boring, everyday, ordinary, monochrome, greyscale, NSFW, text, watermark, low resolution,nude')
        with gr.Row():
            with gr.Accordion('文生图生成参数设置', open=False):
                    with gr.Row():
                        modeltype=gr.Dropdown(choices=['sdxl_lightning'],value='sdxl_lightning',label='文生图模型')
                        imagenum_per_prompt = gr.Slider(1, 5, 2, label='imagenum_per_prompt', step=1)
                    with gr.Accordion('生成参数设置', open=False):
                        guidance = gr.Slider(0, 10, 0, label='seed', step=0.5)
                        seed = gr.Slider(1, 10000000000000, 100, label='seed', step=1)
                        width = gr.Slider(480, 1920, 1024, label='width', step=24)
                        height = gr.Slider(480, 1920, 1024, label='width', step=24)
            with gr.Accordion('视频生成参数设置', open=False):
                    with gr.Row():
                        t2vmodeltype=gr.Dropdown(choices=['animtedifflcm'],value='animtedifflcm',label='文生超短视频模型')  
                        i2vmodeltype=gr.Dropdown(choices=['svd'],value='svd',label='图生超短视频模型')
                    with gr.Row():
                        with gr.Accordion('文生视频生成参数设置', open=False):
                                    seed_t2v = gr.Slider(1, 10000000000000, 999889999, label='seed', step=1)
                                    fps_t2v = gr.Slider(1, 25, 8, label='fps', step=1)
                                    framenum_t2v = gr.Slider(12, 32, 16, label='framenum', step=1)
                                    guidance_t2v = gr.Slider(0, 10, 1.8, label='guidance', step=0.1)
                                    num_inference_steps = gr.Slider(4, 25, 6, label='inference step', step=1)
                                    videosavepath_t2v=gr.Textbox(label='文生图视频savepath',value='tmpt2v/')
                    
                        with gr.Accordion('图生视频生成参数设置', open=False):
                            seedvideo = gr.Slider(1, 10000000000000, 100, label='seedvideo', step=1)
                            fps = gr.Slider(1, 25, 8, label='fps', step=1)
                            num_iterations = gr.Slider(1, 5, 1, label='num_iteration', step=1)
                            videosavepath=gr.Textbox(label='图生视频savepath',value='tmpi2v/')
        
        with gr.Row():
            with gr.Column(scale=2):
                pos_prompt=gr.Textbox(label='图片生成prompt',show_label=True)
                generate_btn=gr.Button('文生图', variant="primary")
                with gr.Row():
                    generate_image = gr.Image(type='pil',label='generated_image',height=400)
                with gr.Row():
                    with gr.Tab('生成的图片'):
                        gallary_image=gr.Gallery(label="生成的图片", show_download_button=True, object_fit="contain",columns=2,height=700)#.style(grid=[3], height=600)
                    with gr.Tab('生成的候选图片'):
                        select_gallary_image=gr.Gallery(label="生成的候选图片", show_download_button=True, object_fit="contain",columns=2,height=700)#.style(grid=[3], height=300)
            with gr.Column(scale=2):    
                t2v_prompt = gr.Textbox(label='视频生成prompt')#value='art photo by michiko kon, intricate, ultra-detailed, sharp details, cg 8k wallpaper, woman, swimsuit, medium hair,beautiful underwater scene')  
                
                with gr.Row():
                    text_generate_video_btn = gr.Button('文生超短视频', variant="primary")
                    img_generate_video_btn = gr.Button('图生超短短视频', variant="primary")
                generate_video=gr.Video(label='short_generated_video',show_download_button=True,height=400)
                     
        with gr.Row():
            with gr.Column(scale=2):
                select_gallary_video=gr.Gallery(label="生成的候选超短视频", show_download_button=True,object_fit="contain",columns=2,height=600)#.style(grid=[2], height=600)
            
        pos_prompt.submit(t2imodel.multi_generate,
                               inputs=[pos_prompt,imagenum_per_prompt,neg_prompt,guidance,seed,width,height],
                               outputs=[gallary_image,generate_image])
        generate_btn.click(t2imodel.multi_generate,
                               inputs=[pos_prompt,imagenum_per_prompt,neg_prompt,guidance,seed,width,height],
                               outputs=[gallary_image,generate_image])
        gallary_image.select(get_select_image,inputs=[gallary_image,select_gallary_image],outputs=[select_gallary_image,generate_image])

        select_gallary_image.select(get_select_to_candi_image,inputs=[select_gallary_image],outputs=[generate_image])

        i2v_input=[generate_image,select_gallary_video,seedvideo,fps,num_iterations,videosavepath,i2vmodeltype]
        img_generate_video_btn.click(image_generate_video,inputs=i2v_input,outputs=[select_gallary_video,generate_video])
        
        t2v_input=[t2v_prompt,neg_prompt,select_gallary_video,seed_t2v,fps_t2v,framenum_t2v,guidance_t2v,num_inference_steps,videosavepath_t2v,t2vmodeltype]
        text_generate_video_btn.click(generate_text_video,inputs=t2v_input,outputs=[select_gallary_video,generate_video])
        
                 
    return block       
    
        
    
if __name__ =="__main__":
    try:
        t2imodel=sdxl_lightning_model()
    except Exception as e:
        print('t2i error',e)
        t2imodel=sd15_model()
    try:
        i2vmodel=svd_model()
    except Exception as e:
        print('i2v error',e)
        i2vmodel=svd_model('fake')
    #torch.cuda.empty_cache()
    try:
        t2vmodel=animatedifflcm_model()
    except Exception as e:
        print('t2v error',e)
        t2vmodel=animatedifflcm_model('fake')
    

    generate_image_gr_demo(t2imodel,i2vmodel,t2vmodel).queue().launch(max_threads=15,
                                                            show_api=True,
                                                            share=False, 
                                                            server_name='0.0.0.0',server_port=8082)
    
    pass