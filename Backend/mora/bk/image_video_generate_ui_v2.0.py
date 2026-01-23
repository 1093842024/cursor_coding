import torch,sys,os,datetime
from pathlib import Path
from PIL import Image
import numpy as np
import gradio as gr
from generate_image import sdxl_lightning_model,sd15_model
from iterate_generate_video import svd_model,animatedifflcm_model,video_to_gif,load_video_to_frames

activate_long_video=False

if activate_long_video:
    from StreamingT2V.streamingT2V_interface import init_streamingt2v_model,init_v2v_model,stream_long_gen,video2video_randomized
else:
    init_streamingt2v_model,init_v2v_model,stream_long_gen,video2video_randomized=None,None,None,None

os.environ["no_proxy"] = "localhost,0.0.0.0,:8082"

ckpt_file_streaming_t2v = Path("StreamingT2V/t2v_enhanced/checkpoints/streaming_t2v.ckpt").absolute()
Path('stream_output').mkdir(parents=True, exist_ok=True)
result_fol = Path('stream_output').absolute()

cfg_v2v = {'downscale': 1, 'upscale_size': (1280, 720), 'model_id': '/mnt/glennge/diffusers/hub/damo/Video-to-Video/', 'pad': True}

examples_0 = [
        ["Experience the dance of jellyfish: float through mesmerizing swarms of jellyfish, pulsating with otherworldly grace and beauty.",
            "(nipples:1.5), (worst quality:1.3), unfinished sketch, blurry, normal, mundane, boring, everyday, ordinary, monochrome, greyscale, NSFW, text, watermark, low resolution,nude",
            None,100,8,16,2,6,'tmpt2v/','animtedifflcm'],
        ["A hummingbird flutters among colorful flowers, its wings beating rapidly.",
            "(nipples:1.5), (worst quality:1.3), unfinished sketch, blurry, normal, mundane, boring, everyday, ordinary, monochrome, greyscale, NSFW, text, watermark, low resolution,nude",
            None,100,8,16,2,6,'tmpt2v/','animtedifflcm']
        ]

examples_1 = [
        [None,None,"Experience the dance of jellyfish: float through mesmerizing swarms of jellyfish, pulsating with otherworldly grace and beauty.",
            "(nipples:1.5), (worst quality:1.3), unfinished sketch, blurry, normal, mundane, boring, everyday, ordinary, monochrome, greyscale, NSFW, text, watermark, low resolution,nude",
            100,8,16,2,6,'tmpt2v/',56,50,9.0],
        [None,None,"A hummingbird flutters among colorful flowers, its wings beating rapidly.",
            "(nipples:1.5), (worst quality:1.3), unfinished sketch, blurry, normal, mundane, boring, everyday, ordinary, monochrome, greyscale, NSFW, text, watermark, low resolution,nude",
            100,8,16,2,6,'tmpt2v/',56,50,9.0]
        ]

examples_2 = [
        [None,None,None,"Fishes swimming in ocean camera moving, cinematic.",
            "(nipples:1.5), (worst quality:1.3), unfinished sketch, blurry, normal, mundane, boring, everyday, ordinary, monochrome, greyscale, NSFW, text, watermark, low resolution,nude",
            100,8,16,2,6,'tmpt2v/',56,50,9.0],
        [None,None,None,"A squirrel on a table full of big nuts.",
            "(nipples:1.5), (worst quality:1.3), unfinished sketch, blurry, normal, mundane, boring, everyday, ordinary, monochrome, greyscale, NSFW, text, watermark, low resolution,nude",
            100,8,16,2,6,'tmpt2v/',56,50,9.0]
        ]


def generate_image_gr_demo(t2imodel,i2vmodel,t2vmodel,stream_cli, stream_model,msxl_model):
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
     
    def v2long_video(select_gallary_video_long,short_video,
        t2v_prompt,neg_prompt,seed_t2v,
        long_framenum,long_t,image_guidance,device='cuda'
        ):
        if not os.path.exists(short_video):gr.Error('请先生成超短视频')
        all_generate_video_long=[]
        if select_gallary_video_long is not None:
                for gallary_data in select_gallary_video:
                    gifpath,label=gallary_data
                    all_generate_video_long.append((gifpath,label))

        n_autoreg_gen = (long_framenum-8)//8
        now = datetime.datetime.now()
        result_file_stem = t2v_prompt[:100].replace(" ", "_") + "_" + str(now.time()).replace(":", "_").replace(".", "_")

        frames = load_video_to_frames(short_video)

        video_path,gif_path=stream_long_gen(t2v_prompt, frames, n_autoreg_gen, neg_prompt, seed_t2v, long_t, image_guidance, result_file_stem, stream_cli, stream_model,result_fol,device)
        all_generate_video_long.append((gif_path,t2v_prompt))
        #torch.cuda.empty_cache()
        return video_path,all_generate_video_long
        

    def long2_enhance_video(select_gallary_video_enhance,long_video,
        t2v_prompt,
        chunk_size=24, overlap_size=8,device='cuda'):
        if not os.path.exists(long_video):gr.Error('请先生成长视频')

        all_generate_video_enhance=[]
        if select_gallary_video_enhance is not None:
                for gallary_data in select_gallary_video_enhance:
                    gifpath,label=gallary_data
                    all_generate_video_enhance.append((gifpath,label))

        encoded_video = video2video_randomized(t2v_prompt, long_video, result_fol, cfg_v2v, msxl_model,chunk_size=chunk_size, overlap_size=overlap_size)
        encoded_gif = video_to_gif(encoded_video)
        all_generate_video_enhance.append((encoded_gif,t2v_prompt))
        #torch.cuda.empty_cache()
        return encoded_video,all_generate_video_enhance
    
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
                        with gr.Accordion('长视频生成参数设置', open=False):
                            v2vmodeltype=gr.Dropdown(choices=['st_t2v'],value='st_t2v',label='长视频生成模型')    
                            long_t = gr.Slider(label="Timesteps", minimum=20, maximum=100, value=50, step=1,)
                            image_guidance = gr.Slider(label='Image guidance scale', minimum=1, maximum=10, value=9.0, step=1.0)  

                        with gr.Accordion('enhance视频生成参数设置', open=False):
                            enhance_v2vmodeltype=gr.Dropdown(choices=['Vid2Vid-XL'],value='Vid2Vid-XL',label='视频enhance模型')
                            chunk_size=gr.Slider(2, 64, 24, label='chunk size', step=1)
                            overlap_size=gr.Slider(2, 32, 8, label='overlap size', step=1)
        
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
                        generate_video_long=gr.Video(label='long_generated_video',show_download_button=True,height=300)
                    with gr.Column(scale=1):
                        gr.Markdown(" 根据生成的视频长度，生成时间在几min到几十min不等 ") 
                        long_framenum = gr.Slider(24, 480, 48, label='long frame num', step=8)
                        #t2v_long_video_btn = gr.Button('文生长视频', variant="primary")
                        #t2v_video_enhance_btn = gr.Button('文生高清长视频', variant="primary")
                        t2v_long_video_btn = gr.Button('超短视频生成长视频', variant="primary")
                        t2v_video_enhance_btn = gr.Button('长视频生成高清视频', variant="primary")
                generate_video_enhance=gr.Video(label='enhance_generated_video',show_download_button=True,height=400)
                
        with gr.Row():
            with gr.Column(scale=2):
                select_gallary_video=gr.Gallery(label="生成的候选超短视频", show_download_button=True,object_fit="contain",columns=2,height=600)#.style(grid=[2], height=600)
            with gr.Column(scale=2):
                select_gallary_video_long=gr.Gallery(label="生成的候选长视频", show_download_button=True,object_fit="contain",columns=2,height=600)#.style(grid=[2], height=600)
            with gr.Column(scale=2):
                select_gallary_video_enhance=gr.Gallery(label="生成的高清候选长视频", show_download_button=True,object_fit="contain",columns=2,height=600)#.style(grid=[2], height=600)
                        

            
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
        
        '''
        t2v_long_input=[select_gallary_video_long,t2v_prompt,neg_prompt,seed_t2v,fps_t2v,framenum_t2v,guidance_t2v,num_inference_steps,videosavepath_t2v,long_framenum,long_t,image_guidance]
        t2v_long_video_btn.click(t2v_long_video,
            inputs=t2v_long_input,
            outputs=[generate_video_long,select_gallary_video_long])

        t2v_video_enhance_input=[select_gallary_video_enhance,t2v_prompt,neg_prompt,seed_t2v,fps_t2v,framenum_t2v,guidance_t2v,num_inference_steps,videosavepath_t2v,long_framenum,long_t,image_guidance,chunk_size,overlap_size]
        t2v_video_enhance_btn.click(t2v_enhance_video,
            inputs=t2v_video_enhance_input,
            outputs=[generate_video_enhance,select_gallary_video_enhance])
        '''
        
        v2long_input=[select_gallary_video_long,generate_video,t2v_prompt,neg_prompt,seed_t2v,long_framenum,long_t,image_guidance]
        t2v_long_video_btn.click(v2long_video,
            inputs=v2long_input,
            outputs=[generate_video_long,select_gallary_video_long])

        long2enhance_input=[select_gallary_video_enhance,generate_video_long,t2v_prompt,chunk_size,overlap_size]
        t2v_video_enhance_btn.click(long2_enhance_video,
            inputs=long2enhance_input,
            outputs=[generate_video_enhance,select_gallary_video_enhance])        

        


        if True:
            gr.Examples(examples=examples_0,
                        inputs=t2v_input,
                        outputs=[select_gallary_video,generate_video],
                        fn=generate_text_video,
                        run_on_click=False,
                        cache_examples=True,
                        preprocess=False,
                        postprocess=True,
                        )
                 
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

    if activate_long_video:
        try:
            stream_cli, stream_model = init_streamingt2v_model(ckpt_file_streaming_t2v, result_fol)
        except Exception as e:
            print('stream model error',e)
            stream_cli, stream_model=None,None
        #torch.cuda.empty_cache()
        try:
            msxl_model= init_v2v_model(cfg_v2v)
        except Exception as e:
            print('v2v model error',e)
            msxl_model=None
    else:
        stream_cli, stream_model=None,None
        msxl_model=None
    

    generate_image_gr_demo(t2imodel,i2vmodel,t2vmodel,stream_cli, stream_model,msxl_model).queue().launch(max_threads=15,
                                                            show_api=True,
                                                            share=False, 
                                                            server_name='0.0.0.0',server_port=8082)
    
    pass