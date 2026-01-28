import torch
import os,sys,shutil,random
from io import BytesIO
import imageio
import requests
import numpy as np
from PIL import Image
from glob import glob
from pathlib import Path
import uuid
import gradio as gr
import time
from moviepy.editor import VideoFileClip, concatenate_videoclips

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video#,export_to_gif
from huggingface_hub import hf_hub_download

from safetensors.torch import load_file as load_safetensors


device = 'cuda' if torch.cuda.is_available() else 'cpu'
BASEDIR='/mnt/glennge/MoneyPrinter/Backend/mora/'

# 阶段5 扩展点：文生视频/图生视频可插拔。可在此接入 HunyuanVideo、CogVideoX、Luma、Kling、Minimax 等：
# def get_t2v_model(name): return {"animatedifflcm": animatedifflcm_model(), "kling": ...}[name]
# def get_i2v_model(name): return {"svd": svd_model(), "cogvideox": ...}[name]

def I2VGenXL_demo():
    from diffusers import I2VGenXLPipeline

    pipeline = I2VGenXLPipeline.from_pretrained(
        #"ali-vilab/i2vgen-xl",
        "/mnt/glennge/diffusers/hub/models--ali-vilab--i2vgen-xl/snapshots/39e1979ea27be737b0278c06755e321f2b4360d5/",
         torch_dtype=torch.float16, variant="fp16")
    pipeline.enable_model_cpu_offload()

    image_url = ("https://huggingface.co/datasets/diffusers/docs-images/resolve/main/i2vgen_xl_images/img_0009.png")
    image = load_image(image_url).convert("RGB")

    prompt = "Papers were floating in the air on a table in the library"
    negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
    generator = torch.manual_seed(8888)

    frames = pipeline(
        prompt=prompt,
        image=image,
        height=720,
        width=1280,
        num_frames=16,
        num_inference_steps=50,
        negative_prompt=negative_prompt,
        guidance_scale=9.0,
        generator=generator,
    ).frames[0]
    video_path = export_to_gif(frames, "i2v.gif")
    

def animatediff_ligtning_t2v_demo(): #最大只支持32帧
    from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
    from diffusers.utils import export_to_gif
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    device = "cuda"
    dtype = torch.float16

    step = 4  # Options: [1,2,4,8]
    repo = "ByteDance/AnimateDiff-Lightning"
    ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
    base = "emilianJR/epiCRealism"  # Choose to your favorite base model.

    adapter = MotionAdapter().to(device, dtype)
    adapter.load_state_dict(load_file(hf_hub_download(repo ,ckpt), device=device))
    pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

    output = pipe(prompt="A girl smiling", guidance_scale=1.0, num_inference_steps=step)
    export_to_gif(output.frames[0], "animation.gif")

def animatediff_lcm_t2v_demo():
    from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
    from diffusers.utils import export_to_gif

    #adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=torch.float16)
    #pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16)

    adapter = MotionAdapter.from_pretrained("/mnt/glennge/diffusers/hub/models--wangfuyun--AnimateLCM/snapshots/6cdc714205bbc04c3b2031ee63725cd6e54dbe56/", torch_dtype=torch.float16)
    pipe = AnimateDiffPipeline.from_pretrained("/mnt/glennge/diffusers/hub/models--emilianJR--epiCRealism/snapshots/6522cf856b8c8e14638a0aaa7bd89b1b098aed17/", motion_adapter=adapter, torch_dtype=torch.float16)

    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, 
        beta_schedule="linear",
        )

    pipe.load_lora_weights("/mnt/glennge/diffusers/hub/models--wangfuyun--AnimateLCM/snapshots/6cdc714205bbc04c3b2031ee63725cd6e54dbe56/", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
    pipe.set_adapters(["lcm-lora"], [0.8])

    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()

    output = pipe(
        prompt="A space rocket with trails of smoke behind it launching into space from the desert, 4k, high resolution",
        negative_prompt="bad quality, worse quality, low resolution",
        num_frames=16,
        guidance_scale=1.8,
        num_inference_steps=6,
        generator=torch.Generator("cpu").manual_seed(999889999),
    )
    frames = output.frames[0]
    export_to_gif(frames, "animatelcm.gif")

def iterate_generate_video(imagepath,iter_num=5,pipe=None,pipetype='svd'):
    if pipetype=='svd':
        pipe = StableVideoDiffusionPipeline.from_pretrained(
        "/mnt/glennge/diffusers/hub/models--stabilityai--stable-video-diffusion-img2vid-xt-1-1/snapshots/bfaf882a74971cae6fe4c22935110030db21fae9/",
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        #"stabilityai/stable-video-diffusion-img2vid-xt-1-1", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        pipe.enable_model_cpu_offload()
    
    def resize_image(image, output_size=(1024, 576)):
        # Calculate aspect ratios
        target_aspect = output_size[0] / output_size[1]  # Aspect ratio of the desired size
        image_aspect = image.width / image.height  # Aspect ratio of the original image

        # Resize then crop if the original image is larger
        if image_aspect > target_aspect:
            # Resize the image to match the target height, maintaining aspect ratio
            new_height = output_size[1]
            new_width = int(new_height * image_aspect)
            resized_image = image.resize((new_width, new_height), Image.LANCZOS)
            # Calculate coordinates for cropping
            left = (new_width - output_size[0]) / 2
            top = 0
            right = (new_width + output_size[0]) / 2
            bottom = output_size[1]
        else:
            # Resize the image to match the target width, maintaining aspect ratio
            new_width = output_size[0]
            new_height = int(new_width / image_aspect)
            resized_image = image.resize((new_width, new_height), Image.LANCZOS)
            # Calculate coordinates for cropping
            left = 0
            top = (new_height - output_size[1]) / 2
            right = output_size[0]
            bottom = (new_height + output_size[1]) / 2

        torch.cuda.empty_cache()
        cropped_image = resized_image.crop((left, top, right, bottom))
        return cropped_image
    
    def Get_Last_Frame(video_path, output_image_path):
        # Load the video file using VideoFileClip
        with VideoFileClip(video_path) as video:
            # Get the last frame by going to the last second of the video
            last_frame = video.get_frame(video.duration - 0.01)  # a fraction before the end

        # Now, we save the last frame as an image using PIL
        last_frame_image = Image.fromarray(last_frame)
        last_frame_image.save(output_image_path)
        return output_image_path

    def generate_and_concatenate_videos(initial_image_path, num_iterations=60):
        video_paths = []  # To keep track of all generated video paths
        current_image_path = initial_image_path

        for iteration in range(num_iterations):
            if pipetype=='svd':
                # Generate frames based on the current image
                #image = resize_image(Image.open(current_image_path).convert('RGB'))
                image = Image.open(current_image_path).convert('RGB').resize((1024, 576))
                seed = int(time.time())
                torch.manual_seed(seed)
                frames = pipe(image, decode_chunk_size=12, generator=torch.Generator(), motion_bucket_id=127).frames[0]

            # Export frames to video and save the path
            video_path = f"data/video_segment_{iteration}.mp4"
            export_to_video(frames, video_path, fps=8)
            video_paths.append(video_path)

            # Get the last frame of the current video for the next iterationRGB
            current_image_path = Get_Last_Frame(video_path, "1.png")

        # Load and concatenate all video segments
        clips = [VideoFileClip(path) for path in video_paths]
        final_clip = concatenate_videoclips(clips)

        # Save the final video
        final_clip.write_videofile("data/final_output_video.mp4")
    
    generate_and_concatenate_videos(imagepath,iter_num)
    
def resize_image(image, output_size=(1024, 576)):
    # Calculate aspect ratios
    target_aspect = output_size[0] / output_size[1]  # Aspect ratio of the desired size
    image_aspect = image.width / image.height  # Aspect ratio of the original image

    target_image=Image.new('RGB',output_size,(0,0,0))
    # Resize then crop if the original image is larger
    if image_aspect < target_aspect:
        # Resize the image to match the target height, maintaining aspect ratio
        new_height = output_size[1]
        new_width = int(new_height * image_aspect)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        x,y = int((output_size[0]-new_width)/2) ,0
    else:
        # Resize the image to match the target width, maintaining aspect ratio
        new_width = output_size[0]
        new_height = int(new_width / image_aspect)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        x,y = 0,int((output_size[1]-new_height)/2)

    target_image.paste(resized_image,(x,y))
    return target_image.convert('RGB')

def Get_Last_Frame(video_path):
    # Load the video file using VideoFileClip
    with VideoFileClip(video_path) as video:
        # Get the last frame by going to the last second of the video
        last_frame = video.get_frame(video.duration - 0.01)  # a fraction before the end

    # Now, we save the last frame as an image using PIL
    last_frame_image = Image.fromarray(last_frame)
    return last_frame_image

import tempfile
def export_to_gif(images, output_gif_path= None,duration=100):
    if output_gif_path is None:
        output_gif_path = tempfile.NamedTemporaryFile(suffix=".gif").name

    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=duration,
        loop=0,
    )
    return output_gif_path

def video_to_gif(videopath):
    os.system(f'ffmpeg -y -i {videopath} -b 2048k {videopath}.gif')
    return f'{videopath}.gif'


def load_video_to_frames(file_path: str):
    images = []

    if file_path.startswith(('http://', 'https://')):
        # If the file_path is a URL
        response = requests.get(file_path)
        response.raise_for_status()
        content = BytesIO(response.content)
        vid = imageio.get_reader(content)
    else:
        # Assuming it's a local file path
        vid = imageio.get_reader(file_path)

    for frame in vid:
        pil_image = Image.fromarray(frame)
        images.append(pil_image)

    return images


class animatedifflcm_model():
    def __init__(self,model='animatedifflcm'):
        if model=='animatedifflcm':
            from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
            #adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=torch.float16)
            #pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16)

            adapter = MotionAdapter.from_pretrained("/mnt/glennge/diffusers/hub/models--wangfuyun--AnimateLCM/snapshots/6cdc714205bbc04c3b2031ee63725cd6e54dbe56/", torch_dtype=torch.float16)
            self.pipe = AnimateDiffPipeline.from_pretrained("/mnt/glennge/diffusers/hub/models--emilianJR--epiCRealism/snapshots/6522cf856b8c8e14638a0aaa7bd89b1b098aed17/", motion_adapter=adapter, torch_dtype=torch.float16)
            self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config, beta_schedule="linear",)
            self.pipe.load_lora_weights("/mnt/glennge/diffusers/hub/models--wangfuyun--AnimateLCM/snapshots/6cdc714205bbc04c3b2031ee63725cd6e54dbe56/", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
            self.pipe.set_adapters(["lcm-lora"], [0.8])
            self.pipe.enable_vae_slicing()
            self.pipe.enable_model_cpu_offload()
            #self.pipe.to(device)
        else:
            self.pipe=None
        
    def generate(self,pos_prompt="A space rocket with trails of smoke behind it launching into space from the desert, 4k, high resolution",
        neg_prompt="bad quality, worse quality, low resolution",
        seed=999889999,
        fps=8,
        framenums=16,
        guidance=1.8,
        num_inference_steps=6,
        savepath='tmpt2v/',savename=None):

        savepath=BASEDIR+savepath
        if not os.path.exists(savepath):os.makedirs(savepath)

        output = self.pipe(
            prompt=pos_prompt,
            negative_prompt=neg_prompt,
            num_frames=framenums,
            guidance_scale=guidance,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(seed),
            )
        frames = output.frames[0]
        if savename is not None:
            video_path = f"{savepath}animatelcm_{savename}.mp4"
            gif_path = f"{savepath}animatelcm_{savename}.gif"
            export_to_video(frames, video_path, fps=fps)
            export_to_gif(frames, gif_path,int(1000/fps))
        else:
            video_path,gif_path=None,None
        return frames,video_path,gif_path

    def generate_and_concatenate_videos(self,
        pos_prompt="A space rocket with trails of smoke behind it launching into space from the desert, 4k, high resolution",
        neg_prompt="bad quality, worse quality, low resolution",
        seed=999889999,
        fps=8,
        framenums=16,
        guidance=1.8,
        num_inference_steps=6,
        savepath='tmpt2v/'):
        
        frames,video_path,gif_path=self.generate(pos_prompt,neg_prompt,seed,fps,framenums,guidance,num_inference_steps,savepath,None)
        
        # Export frames to video and save the path
        video_path = f"{savepath}animatelcm_{pos_prompt[:10]}.mp4"
        gif_path = f"{savepath}animatelcm_{pos_prompt[:10]}.gif"
        export_to_video(frames, video_path, fps=fps)
        export_to_gif(frames, gif_path,int(1000/fps))
        return video_path,gif_path

class svd_model():
    def __init__(self,model='svd'):
        if model=='svd':
            self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            "/mnt/glennge/diffusers/hub/models--stabilityai--stable-video-diffusion-img2vid-xt-1-1/snapshots/bfaf882a74971cae6fe4c22935110030db21fae9/",
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True
            )
            self.pipe.enable_model_cpu_offload()
            #self.pipe.to(device)
        else:
            self.pipe=None
    
    def generate(self,image,seed=100,fps=8,savepath='tmpi2v/',savename=None):
        savepath=BASEDIR+savepath
        if not os.path.exists(savepath):os.makedirs(savepath)
        image = resize_image(image.convert('RGB'),(1024, 576))
        #seed = int(time.time())
        torch.manual_seed(seed)
        frames = self.pipe(image, decode_chunk_size=8, generator=torch.Generator(), motion_bucket_id=127).frames[0]
        if savename is not None:
            # Export frames to video and save the path
            video_path = f"{savepath}svd_{savename}.mp4"
            export_to_video(frames, video_path, fps=fps)
            gif_path = f"{savepath}svd_{savename}.gif"
            export_to_gif(frames, gif_path,int(1000/fps))
        else:
            video_path,gif_path=None,None
        return frames,video_path,gif_path


    def generate_and_concatenate_videos(self,image,seed=100,fps=8,num_iterations=1,savepath='tmpi2v/'):
        video_paths = []  # To keep track of all generated video paths

        for iteration in range(num_iterations):
            # Generate frames based on the current image
            #image = image.convert('RGB').resize((1024, 576))
            frames,video_path,gif_path = self.generate(image,seed,fps,savepath,None)
            # Export frames to video and save the path
            video_path = f"{savepath}svd_segment_{iteration}.mp4"
            export_to_video(frames, video_path, fps=fps)
            video_paths.append(video_path)
            # Get the last frame of the current video for the next iterationRGB
            image = frames[-1]#Get_Last_Frame(video_path)

        # Load and concatenate all video segments
        clips = [VideoFileClip(path) for path in video_paths]
        final_clip = concatenate_videoclips(clips)

        # Save the final video
        finalvideopath=f"{savepath}final_svd_video.mp4"
        final_clip.write_videofile(finalvideopath)
        return finalvideopath,video_to_gif(finalvideopath)




if __name__ =="__main__":
    #animatediff_lcm_t2v_demo()
    
    #iterate_generate_video('data/merlion.png',2)
    #i2vmodel=svd_model()
    #i2vmodel.generate_and_concatenate_videos(Image.open('data/merlion.png'))

    #t2vmodel=animatedifflcm_model()
    #t2vmodel.generate_and_concatenate_videos()
    
    
    pass