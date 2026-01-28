import logging
import torch
import sys
import os
import cv2
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler, DDIMScheduler, TCDScheduler
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from PIL import Image
import numpy as np
import gradio as gr

device = "cuda" if torch.cuda.is_available() else "cpu"
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)

# 便于从 Backend 根目录解析 config_loader（story_generate 等从 Backend 启动）
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

logger = logging.getLogger(__name__)

# ---- T2I/I2I 多后端常量（AGENT.md 3.2）----
T2I_BACKEND_OPEN_SOURCE = "open_source"
T2I_BACKEND_COMMERCIAL = "commercial"
T2I_OPEN_SOURCE_MODEL_IDS = ["flux_dev", "flux_schnell", "z_image_turbo", "sdxl_lightning", "sd15", "ipadapter"]
T2I_COMMERCIAL_MODEL_IDS = ["nano_banana_pro", "tongyi", "bytedance", "kling"]

def sdxl_demo(prompt):
    # load both base & refiner
    base = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    base.to("cuda")
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",)
    refiner.to("cuda")
    # Define how many steps and what % of steps to be run on each experts (80/20) here
    n_steps = 40
    high_noise_frac = 0.8

    def Get_image(prompt,base,refiner):
        # run both experts

        image = base(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            height=576,
            width=1024,
            output_type="latent",
        ).images
        image = refiner(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            height=576,
            width=1024,
            image=image,
        ).images[0]
        image.save('image.png')
        return image
    
    return Get_image(prompt,base,refiner)

def sdxl_lightning_demo():
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    base ='/mnt/glennge/diffusers/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b/'
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_4step_lora.safetensors" # Use the correct ckpt for your step setting!

    # Load model.
    pipe = StableDiffusionXLPipeline.from_pretrained(base, torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.load_lora_weights('/mnt/glennge/diffusers/hub/models--ByteDance--SDXL-Lightning/snapshots/c9a24f48e1c025556787b0c58dd67a091ece2e44/'+ckpt)#hf_hub_download(repo, ckpt))
    pipe.fuse_lora()

    # Ensure sampler uses "trailing" timesteps.
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.to(device)
    
    # Ensure using the same inference steps as the loaded model and CFG set to 0.
    pipe("A girl smiling", num_inference_steps=4, guidance_scale=1).images[0].save("sdxl-lightning.png")#.show()

def playground_demo():
    pipe = DiffusionPipeline.from_pretrained(
        "playgroundai/playground-v2.5-1024px-aesthetic",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
        #).to("cuda")
        ).to("cuda")

    # # Optional: Use DPM++ 2M Karras scheduler for crisper fine details
    # from diffusers import EDMDPMSolverMultistepScheduler
    # pipe.scheduler = EDMDPMSolverMultistepScheduler()

    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    image = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=3).images[0].save("data/playground.png")

def hypersd_demo(scheduler_type='default'):
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    repo_name = "ByteDance/Hyper-SD"
    # Take 2-steps lora as an example
    ckpt_name = "Hyper-SDXL-2steps-lora.safetensors"
    ckpt_name = "Hyper-SDXL-1step-lora.safetensors"
    # Load model.
    #hf_hub_download(repo_name, ckpt_name)
    
    pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
    pipe.fuse_lora()
    if scheduler_type=='default': # for 2、4、8 steps
        # Ensure ddim scheduler timestep spacing set as trailing !!!
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        # lower eta results in more detail
        prompt="a photo of a cat"
        image=pipe(prompt=prompt, num_inference_steps=2, guidance_scale=0).images[0]
    elif 'high quality': # for 1 step
        # Use TCD scheduler to achieve better image quality
        pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
        # Lower eta results in more detail for multi-steps inference
        eta=1.0
        prompt="a photo of a cat"
        image=pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0, eta=eta).images[0]

def resize_img(input_image,max_side=1280,min_side=1024,size=None,pad_to_max_side=False,mode=Image.BILINEAR,base_pixel_number=64,):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y : offset_y + h_resize_new, offset_x : offset_x + w_resize_new] = (
            np.array(input_image)
        )
        input_image = Image.fromarray(res)
    return input_image

def pil_to_cv2(image_pil):
    image_np = np.array(image_pil)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_cv2

def get_canny_map(input_image,control_scale,size=(1024,1024)):
    if input_image is not None:
        input_image = resize_img(input_image, size=size)
        cv_input_image = pil_to_cv2(input_image)
        detected_map = cv2.Canny(cv_input_image, 50, 200)
        canny_map = Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB))
    else:
        canny_map = Image.new("RGB",size, color=(255, 255, 255))
        control_scale = 0
    return canny_map,control_scale


from ip_adapter import IPAdapterXL
class hyper_sd_model():
    def __init__(self,device='cuda',instantstyle=True,controlnet=False,savememory=True):
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        base ='/mnt/glennge/diffusers/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b/'
        repo = "ByteDance/Hyper-SD"
        lora_ckpt = "/mnt/glennge/diffusers/hub/models--ByteDance--Hyper-SD/snapshots/1a091a0ba26e5639d083c6377ba4f2bed5115ccc/Hyper-SDXL-1step-lora.safetensors" 
        self.device=device
        self.controlnet=controlnet
        self.instantstyle=instantstyle
        # Load model.
        if self.controlnet:
            controlnet_path = "/mnt/glennge/diffusers/hub/models--diffusers--controlnet-canny-sdxl-1.0-small/snapshots/edd85f64c5f87dfb6d73762949d9daca16389518/"
            controlnet = ControlNetModel.from_pretrained(controlnet_path, use_safetensors=True, torch_dtype=torch.float16).to(device)
            # load SDXL pipeline
            self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(base,
                controlnet=controlnet,torch_dtype=torch.float16,variant="fp16",add_watermarker=False)
        else:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(base, torch_dtype=torch.float16, variant="fp16",add_watermarker=False)#.to(device)
        self.pipe.load_lora_weights(lora_ckpt)
        self.pipe.fuse_lora()
        if '1step' in lora_ckpt:
            # Use TCD scheduler to achieve better image quality
            self.pipe.scheduler = TCDScheduler.from_config(self.pipe.scheduler.config)
            # Lower eta results in more detail for multi-steps inference
            eta=1.0
        else:
            # Ensure sampler uses "trailing" timesteps.
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")
        self.pipe.enable_vae_tiling()

        if savememory:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)

        if self.instantstyle:self.init_instantstyle()
            
    def init_instantstyle(self,target='only style block'):
        self.image_encoder_path = "/mnt/glennge/diffusers/hub/models--InstantX--InstantStyle/sdxl_models/image_encoder"
        self.ip_ckpt = "/mnt/glennge/diffusers/hub/models--InstantX--InstantStyle/sdxl_models/ip-adapter_sdxl.bin"
        self.target=target
        if target=='origin IP-Adapter':
            target=["blocks"]
        elif target=='only style block':
            target=["up_blocks.0.attentions.1"]
        elif target=='style+layout block':    
            target=["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"]
        self.ip_model = IPAdapterXL(self.pipe, self.image_encoder_path, self.ip_ckpt, self.device, target_blocks=target)
        
    def generate(self,pos_prompt="A girl smiling",neg_prompt='NSFW, nude',guidance=0.0,seed=100,width=1024,height=1024,
                 num_inference_steps=1,image_canny=None,control_scale=0.5):
        # Ensure using the same inference steps as the loaded model and CFG set to 0.
        generator = torch.manual_seed(seed)
        if self.controlnet:
            canny_map,control_scale=get_canny_map(image_canny,control_scale,size=(width,height)) 
            image=self.pipe(pos_prompt, 
                        negative_prompt=neg_prompt,
                        width=width,   # Make sure height and width are both multiples of 8
                        height=height,
                        num_inference_steps=num_inference_steps, 
                        generator=generator,
                        guidance_scale=guidance,
                        image=canny_map,
                        controlnet_conditioning_scale=float(control_scale),# 0~1
                        ).images[0]
        else:
            image=self.pipe(pos_prompt, 
                        negative_prompt=neg_prompt,
                        width=width,   # Make sure height and width are both multiples of 8
                        height=height,
                        num_inference_steps=num_inference_steps, 
                        generator=generator,
                        guidance_scale=guidance).images[0]
        return image
    
    def multi_generate(self,multi_pos_prompt,imagenum_per_prompt,neg_prompt,guidance,seed,width,height,
                       num_inference_steps=1,image_canny=None,control_scale=0.5):
        image_prompt=[]
        for line in multi_pos_prompt.split('\n'):
            if len(line)<4:continue
            for i in range(imagenum_per_prompt):
                image=self.generate(line,neg_prompt,guidance,seed+i*10,width,height,num_inference_steps,image_canny,control_scale)
                image_prompt.append((image,f'{line}_{i}'))
                yield image_prompt,image

    def generate_style(self,image_style,
        prompt="a cat, masterpiece, best quality, high quality",
        negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
        guidance=0.0,seed=42,width=1024,height=1024,
        scale=1.0,num_inference_steps=1,target='only style block',image_canny=None,control_scale=0.5):
        if not self.instantstyle: gr.Error('exist controlnet or no instantstyle model')
        if target!=self.target:self.init_instantstyle(target)   
        if self.controlnet:
            canny_map,control_scale=get_canny_map(image_canny,control_scale,size=(width,height)) 
            images = self.ip_model.generate(
                            pil_image=image_style,
                            prompt=prompt,
                            negative_prompt= negative_prompt,
                            width=width,   # Make sure height and width are both multiples of 8
                            height=height,
                            scale=scale,  # 0~2
                            guidance_scale=guidance,#5,
                            num_samples=1,
                            num_inference_steps=num_inference_steps, # 1~8
                            image=canny_map,
                            controlnet_conditioning_scale=float(control_scale),# 0~1
                            #device_map="balanced",
                            #max_memory={0:"10GB", 1:"10GB"},
                            seed=seed,eta=1.0)[0]   
        else:
            images = self.ip_model.generate(
                            pil_image=image_style,
                            prompt=prompt,
                            negative_prompt= negative_prompt,
                            width=width,   # Make sure height and width are both multiples of 8
                            height=height,
                            scale=scale,
                            guidance_scale=guidance,#5,
                            num_samples=1,
                            num_inference_steps=num_inference_steps, #30,
                            #device_map="balanced",
                            #max_memory={0:"10GB", 1:"10GB"},
                            seed=seed,eta=1.0)[0]
        return images

    def multi_generate_style(self,ref_img,multi_pos_prompt,imagenum_per_prompt,neg_prompt,guidance,seed,width,height,
                             scale=1.0,num_inference_steps=1,target='only style block',image_canny=None,control_scale=0.5):
        image_prompt=[]
        for line in multi_pos_prompt.split('\n'):
            if len(line)<4:continue
            for i in range(imagenum_per_prompt):
                image=self.generate_style(ref_img,line,neg_prompt,guidance,seed+i*10,width,height,scale,num_inference_steps)
                image_prompt.append((image,f'{line}_{i}'))
                yield image_prompt,image
                
class sdxl_lightning_model():
    def __init__(self,device='cuda',instantstyle=True,controlnet=False,savememory=True):
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        base ='/mnt/glennge/diffusers/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b/'
        repo = "ByteDance/SDXL-Lightning"
        lora_ckpt = "/mnt/glennge/diffusers/hub/models--ByteDance--SDXL-Lightning/snapshots/c9a24f48e1c025556787b0c58dd67a091ece2e44/sdxl_lightning_4step_lora.safetensors" # Use the correct ckpt for your step setting!
        self.device=device
        self.controlnet=controlnet
        self.instantstyle=instantstyle
        # Load model.
        if self.controlnet:
            controlnet_path = "/mnt/glennge/diffusers/hub/models--diffusers--controlnet-canny-sdxl-1.0-small/snapshots/edd85f64c5f87dfb6d73762949d9daca16389518/"
            controlnet = ControlNetModel.from_pretrained(controlnet_path, use_safetensors=True, torch_dtype=torch.float16).to(device)
            # load SDXL pipeline
            self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(base,
                controlnet=controlnet,torch_dtype=torch.float16,variant="fp16",add_watermarker=False)
        else:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(base, torch_dtype=torch.float16, variant="fp16",add_watermarker=False)#.to(device)
        self.pipe.load_lora_weights(lora_ckpt)
        self.pipe.fuse_lora()
        # Ensure sampler uses "trailing" timesteps.
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")
        self.pipe.enable_vae_tiling()

        if savememory:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)

        if self.instantstyle:self.init_instantstyle()

    def init_instantstyle(self,target='only style block'):
        self.image_encoder_path = "/mnt/glennge/diffusers/hub/models--InstantX--InstantStyle/sdxl_models/image_encoder"
        self.ip_ckpt = "/mnt/glennge/diffusers/hub/models--InstantX--InstantStyle/sdxl_models/ip-adapter_sdxl.bin"
        self.target=target
        if target=='origin IP-Adapter':
            target=["blocks"]
        elif target=='only style block':
            target=["up_blocks.0.attentions.1"]
        elif target=='style+layout block':    
            target=["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"]
        self.ip_model = IPAdapterXL(self.pipe, self.image_encoder_path, self.ip_ckpt, self.device, target_blocks=target)
        
    def generate(self,pos_prompt="A girl smiling",neg_prompt='NSFW, nude',guidance=0.0,seed=100,width=1024,height=1024,
                 num_inference_steps=1,image_canny=None,control_scale=0.5):
        # Ensure using the same inference steps as the loaded model and CFG set to 0.
        generator = torch.manual_seed(seed)
        if self.controlnet:
            canny_map,control_scale=get_canny_map(image_canny,control_scale,size=(width,height)) 
            image=self.pipe(pos_prompt, 
                        negative_prompt=neg_prompt,
                        width=width,   # Make sure height and width are both multiples of 8
                        height=height,
                        num_inference_steps=4, 
                        generator=generator,
                        guidance_scale=guidance,
                        image=canny_map,
                        controlnet_conditioning_scale=float(control_scale),# 0~1
                        ).images[0]
        else:
            image=self.pipe(pos_prompt, 
                        negative_prompt=neg_prompt,
                        width=width,   # Make sure height and width are both multiples of 8
                        height=height,
                        num_inference_steps=4, 
                        generator=generator,
                        guidance_scale=guidance).images[0]
        return image
    
    def multi_generate(self,multi_pos_prompt,imagenum_per_prompt,neg_prompt,guidance,seed,width,height,
                       num_inference_steps=1,image_canny=None,control_scale=0.5):
        image_prompt=[]
        for line in multi_pos_prompt.split('\n'):
            if len(line)<4:continue
            for i in range(imagenum_per_prompt):
                image=self.generate(line,neg_prompt,guidance,seed+i*10,width,height,num_inference_steps,image_canny,control_scale)
                image_prompt.append((image,f'{line}_{i}'))
                yield image_prompt,image

    def generate_style(self,image_style,
        prompt="a cat, masterpiece, best quality, high quality",
        negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
        guidance=0.0,seed=42,width=1024,height=1024,
        scale=1.0,num_inference_steps=1,target='only style block',image_canny=None,control_scale=0.5):
        if not self.instantstyle: gr.Error('exist controlnet or no instantstyle model')
        if target!=self.target:self.init_instantstyle(target)   
        if self.controlnet:
            canny_map,control_scale=get_canny_map(image_canny,control_scale,size=(width,height)) 
            images = self.ip_model.generate(
                            pil_image=image_style,
                            prompt=prompt,
                            negative_prompt= negative_prompt,
                            width=width,   # Make sure height and width are both multiples of 8
                            height=height,
                            scale=scale,  # 0~2
                            guidance_scale=guidance,#5,
                            num_samples=1,
                            num_inference_steps=4, # 1~8
                            image=canny_map,
                            controlnet_conditioning_scale=float(control_scale),# 0~1
                            #device_map="balanced",
                            #max_memory={0:"10GB", 1:"10GB"},
                            seed=seed,eta=1.0)[0]   
        else:
            images = self.ip_model.generate(
                            pil_image=image_style,
                            prompt=prompt,
                            negative_prompt= negative_prompt,
                            width=width,   # Make sure height and width are both multiples of 8
                            height=height,
                            scale=scale,
                            guidance_scale=guidance,#5,
                            num_samples=1,
                            num_inference_steps=4, #30,
                            #device_map="balanced",
                            #max_memory={0:"10GB", 1:"10GB"},
                            seed=seed,eta=1.0)[0]
        return images

    def multi_generate_style(self,ref_img,multi_pos_prompt,imagenum_per_prompt,neg_prompt,guidance,seed,width,height,
                             scale=1.0,num_inference_steps=1,target='only style block',image_canny=None,control_scale=0.5):
        image_prompt=[]
        for line in multi_pos_prompt.split('\n'):
            if len(line)<4:continue
            for i in range(imagenum_per_prompt):
                image=self.generate_style(ref_img,line,neg_prompt,guidance,seed+i*10,width,height,scale,num_inference_steps)
                image_prompt.append((image,f'{line}_{i}'))
                yield image_prompt,image
                
class sd15_model():
    def __init__(self,device='cuda'):
        self.pipe=None
    
    def generate(self,pos_prompt="A girl smiling",neg_prompt='NSFW, nude',guidance=0.0,seed=100,width=1024,height=1024):
        image=Image.open('data/merlion.png')
        return image
    
    def multi_generate(self,multi_pos_prompt,imagenum_per_prompt,neg_prompt,guidance,seed,width,height):
        image_prompt=[]
        for line in multi_pos_prompt.split('\n'):
            if len(line)<4:continue
            for i in range(imagenum_per_prompt):
                image=Image.open('data/merlion.png')
                image_prompt.append((image,f'{line}_{i}'))
                yield image_prompt,image 
        pass

class upscale_model():
    def __init__(self,device='cuda',scale=4):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #pip install git+https://github.com/sberbank-ai/Real-ESRGAN.git
        from RealESRGAN import RealESRGAN
        self.model = RealESRGAN(device, scale=scale)
        if scale==4:
            self.model.load_weights(
            curdir+'/weights/RealESRGAN_x4.pth'
            #'/Users/glennge/work/cv/diffusion/ComfyUI/models/upscale_models/RealESRGAN_x4plus.pth'
            )
        else:
            self.model.load_weights('/Users/glennge/work/cv/diffusion/ComfyUI/models/upscale_models/RealESRGAN_x4plus.pth')
        
    def generate(self,image):
        # Ensure using the same inference steps as the loaded model and CFG set to 0.
        sr_image = self.model.predict(image.convert('RGB'))
        return sr_image
    
import torch
from diffusers import AutoPipelineForText2Image, DDIMScheduler
from transformers import CLIPVisionModelWithProjection
from diffusers.utils import load_image
from diffusers.image_processor import IPAdapterMaskProcessor
class ipadapter_model_multi_adapter():
    def __init__(self,lightning=False,hypersd=False,claylora=False):
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "/mnt/glennge/diffusers/hub/models--h94--IP-Adapter/snapshots/018e402774aeeddd60609b4ecdb7e298259dc729/",#"h94/IP-Adapter",
            subfolder="models/image_encoder",
            #subfolder="sdxl_models/image_encoder",
            torch_dtype=torch.float16,)
        
        #self.pipe = AutoPipelineForText2Image.from_pretrained(
        #    #"stabilityai/stable-diffusion-xl-base-1.0",
        #    '/mnt/glennge/diffusers/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b/',
        #    torch_dtype=torch.float16,
        #    image_encoder=image_encoder,)

        base='/mnt/glennge/diffusers/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b/'
        self.pipe = StableDiffusionXLPipeline.from_pretrained(base, torch_dtype=torch.float16,
            image_encoder=image_encoder,
            variant="fp16",add_watermarker=False)#.to(device)

        self.claylora=claylora
        if self.claylora:
            self.pipe.load_lora_weights("/mnt/glennge/diffusers/hub/ClayAnimationRedmond/ClayAnimationRedm.safetensors",
                adapter_name="clay")
            self.pipe.fuse_lora()

        self.lightning=lightning
        self.hypersd=hypersd
        if self.lightning:
            repo = "ByteDance/SDXL-Lightning"
            lora_ckpt = "/mnt/glennge/diffusers/hub/models--ByteDance--SDXL-Lightning/snapshots/c9a24f48e1c025556787b0c58dd67a091ece2e44/sdxl_lightning_4step_lora.safetensors" # Use the correct ckpt for your step setting!
            self.pipe.load_lora_weights(lora_ckpt)
            self.pipe.fuse_lora()
            # Ensure sampler uses "trailing" timesteps.
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")
            self.pipe.enable_vae_tiling()
            #self.pipe.enable_model_cpu_offload()
            self.pipe.to(device)
        elif self.hypersd:
            repo = "ByteDance/Hyper-SD"
            lora_ckpt = "/mnt/glennge/diffusers/hub/models--ByteDance--Hyper-SD/snapshots/1a091a0ba26e5639d083c6377ba4f2bed5115ccc/Hyper-SDXL-1step-lora.safetensors" 
            self.pipe.load_lora_weights(lora_ckpt)
            self.pipe.fuse_lora()
            # Use TCD scheduler to achieve better image quality
            self.pipe.scheduler = TCDScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.enable_vae_tiling()
            #self.pipe.enable_model_cpu_offload()
            self.pipe.to(device)
        else:
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        
        self.processor = IPAdapterMaskProcessor()
        self.init_style_face_adapter()
        self.fake_image =Image.new("RGB",(1024,1024), color=(255, 255, 255))

    def init_style_face_adapter(self):
        self.pipe.load_ip_adapter(
            "/mnt/glennge/diffusers/hub/models--h94--IP-Adapter/snapshots/018e402774aeeddd60609b4ecdb7e298259dc729/",#"h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name=["ip_plus_composition_sdxl.safetensors","ip-adapter-plus_sdxl_vit-h.safetensors", "ip-adapter-plus-face_sdxl_vit-h.safetensors"]
            )
        self.ip_scale=[0,0.5, 0.5] # 使用original Ip-adapter  三个adpter,第一个构图权重为0,其他两个权重为0.5
        self.pipe.set_ip_adapter_scale(self.ip_scale)

        self.style_scale = {"up": {"block_0": [0.0, 1.0, 0.0]},}# 一个adpter只参考 style 且权重为1
        self.style_layout_scale = {                       # 一个adpter参考 layout+style 且权重为1
            "down": {"block_2": [0.0, 1.0]},
            "up": {"block_0": [0.0, 1.0, 0.0]},
        }
        #self.pipe.enable_model_cpu_offload()
        #self.pipe.to(device)
    
    def generate_face_style(self,face_image,style_images,prompt,
        negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
        steps=20,seed=0,guidance=1.0,width=1024,height=1024,
        face_scale=1.0,bgstyle_scale=1.0,onlystyle=False):

        fake_image =Image.new("RGB",(224,224), color=(255, 255, 255))

        if onlystyle:
            face_scale=self.style_scale
            bgstyle_scale=self.style_scale

        if face_image is None and style_images is None:
            adapterimage=[fake_image,fake_image,fake_image]
            self.pipe.set_ip_adapter_scale([0,0,0])
        elif face_image is None:
            adapterimage=[fake_image,style_images,fake_image]
            self.pipe.set_ip_adapter_scale([0,bgstyle_scale,0])
        elif style_images is None:
            adapterimage=[fake_image,fake_image,face_image]
            self.pipe.set_ip_adapter_scale([0,0,face_scale])
        else:
            adapterimage=[fake_image,style_images,face_image]
            self.pipe.set_ip_adapter_scale([0,bgstyle_scale,face_scale])

        if self.lightning:
            steps=4
        elif self.hypersd:
            steps=1
        generator = torch.Generator().manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            ip_adapter_image=adapterimage,
            negative_prompt=negative_prompt,
            width=width,   # Make sure height and width are both multiples of 8
            height=height,
            num_inference_steps=steps, 
            num_images_per_prompt=1,
            guidance_scale=guidance,
            generator=generator,eta=1.0).images[0]
        return image
   
    def generate_face_style_mask(self,style_images,style_masks,
        face_images,face_masks,
        prompt,negative_prompt,steps=20,seed=0,guidance=1.0,width=1024,height=1024,
        face_scale=1.0,bgstyle_scale=1.0):

        #生成人物行为图，然后检测人形框，最后利用SAM提取人形mask和背景mask
        if style_images is not None:assert(len(style_images)==len(style_masks))
        if face_images is not None: assert(len(face_images)==len(face_masks))
        
        composition_image=self.fake_image
        composition_mask= self.processor.preprocess([self.fake_image], height=1024, width=1024)
        
        if style_images is None:
            style_images=[]
            style_masks=[]
        if face_images is None:
            face_images=[]
            face_masks=[]

        stylenum=len(style_images)
        facenum=len(face_images)

        if len(style_masks)+len(face_images)==0:
            face_style_image=[self.fake_image]
            style_mask = self.processor.preprocess([self.fake_image], height=1024, width=1024)
            scale2=0
        else:
            face_style_image=face_images+style_images
            style_mask = self.processor.preprocess(face_masks+style_masks, height=1024, width=1024)
            style_mask = style_mask.reshape(1, style_mask.shape[0], style_mask.shape[2], style_mask.shape[3])   # output -> (1, N, 1024, 1024)
            '''detailed_scale={
            "down": [[1.0]*facenum+[bgstyle_scale]*stylenum],         # 0
            "mid": [[face_scale]*facenum+[bgstyle_scale]*stylenum],   # 1
            "up": {"block_0": [[0.0]*facenum+[bgstyle_scale]*stylenum,  # 2
                        [1.0]*facenum+[bgstyle_scale]*stylenum,    # 3
                        [0.0]*facenum+[bgstyle_scale]*stylenum],    # 4
                    "block_1": [[0.0]*facenum+[bgstyle_scale]*stylenum]},  # 5
            }'''
            detailed_scale=[face_scale]*facenum+[bgstyle_scale]*stylenum
            scale2=detailed_scale
        
        self.pipe.set_ip_adapter_scale([0,scale2,0])

        print(composition_mask.shape,style_mask.shape)

        if self.lightning:
            steps=4
        elif self.hypersd:
            steps=1
        generator = torch.Generator().manual_seed(seed)
        
        image = self.pipe(
            prompt=prompt,
            ip_adapter_image=[composition_image, face_style_image,self.fake_image],
            cross_attention_kwargs={"ip_adapter_masks": [composition_mask, style_mask,composition_mask]},
            negative_prompt=negative_prompt,
            width=width,   # Make sure height and width are both multiples of 8
            height=height,
            num_inference_steps=steps, 
            num_images_per_prompt=1,
            guidance_scale=guidance,
            generator=generator,eta=1.0).images[0]
        print(len(face_style_image),composition_mask.shape,style_mask.shape,scale2)
        return image

    def generate_composition_face_style_mask(self,composition_image,composition_mask,
        style_images,style_masks,
        face_images,face_masks,
        prompt,negative_prompt,steps=20,seed=0,guidance=1.0,width=1024,height=1024,
        composition_scale=1.0,face_scale1=1.0,face_scale2=1.0,bgstyle_scale=1.0):
        #生成人物行为图，然后检测人形框，最后利用SAM提取人形mask和背景mask
        if style_images is not None:assert(len(style_images)==len(style_masks))
        if face_images is not None: assert(len(face_images)==len(face_masks))
        '''
        scale1=composition_scale
        detailed_scale={
            "down": [[1.0]*facenum+[bgstyle_scale]*stylenum],         # 0
            "mid": [[face_scale1]*facenum+[bgstyle_scale]*stylenum],   # 1
            "up": {"block_0": [[face_scale2]*facenum+[bgstyle_scale]*stylenum,  # 2
                        [1.0]*facenum+[bgstyle_scale]*stylenum,    # 3
                        [0.0]*facenum+[bgstyle_scale]*stylenum],    # 4
                    "block_1": [[0.0]*facenum+[bgstyle_scale]*stylenum]},  # 5
            }

        scale2=detailed_scale
        face_style_image=face_images+style_images
        composition_mask = self.processor.preprocess([composition_mask], height=1024, width=1024)
        style_mask = self.processor.preprocess(face_style_image, height=1024, width=1024)
        style_mask = style_mask.reshape(1, style_mask.shape[0], style_mask.shape[2], style_mask.shape[3])   # output -> (1, N, 1024, 1024)
        '''
        if composition_image is None:
            composition_image=self.fake_image
            composition_mask= self.processor.preprocess([self.fake_image], height=1024, width=1024)
            scale1=0
            if style_images is None:
                style_images=[]
                style_masks=[]
            if face_images is None:
                face_images=[]
                face_masks=[]

            stylenum=len(style_images)
            facenum=len(face_images)

            if len(style_masks)+len(face_images)==0:
                face_style_image=[self.fake_image]
                style_mask = self.processor.preprocess([self.fake_image], height=1024, width=1024)
                scale2=0
            else:
                face_style_image=face_images+style_images
                style_mask = self.processor.preprocess(face_masks+style_masks, height=1024, width=1024)
                style_mask = style_mask.reshape(1, style_mask.shape[0], style_mask.shape[2], style_mask.shape[3])   # output -> (1, N, 1024, 1024)
                detailed_scale={
                "down": [[1.0]*facenum+[bgstyle_scale]*stylenum],         # 0
                "mid": [[face_scale1]*facenum+[bgstyle_scale]*stylenum],   # 1
                "up": {"block_0": [[face_scale2]*facenum+[bgstyle_scale]*stylenum,  # 2
                            [1.0]*facenum+[bgstyle_scale]*stylenum,    # 3
                            [0.0]*facenum+[bgstyle_scale]*stylenum],    # 4
                        "block_1": [[0.0]*facenum+[bgstyle_scale]*stylenum]},  # 5
                }
                scale2=detailed_scale
        else:
            composition_mask = self.processor.preprocess([composition_mask], height=1024, width=1024)
            scale1=composition_scale

            if style_images is None:
                style_images=[]
                style_masks=[]
            if face_images is None:
                face_images=[]
                face_masks=[]
                
            stylenum=len(style_images)
            facenum=len(face_images)

            if len(style_masks)+len(face_images)==0:
                face_style_image=[self.fake_image]
                style_mask = self.processor.preprocess([self.fake_image], height=1024, width=1024)
                scale2=0
            else:
                face_style_image=face_images+style_images
                style_mask = self.processor.preprocess(face_masks+style_masks, height=1024, width=1024)
                style_mask = style_mask.reshape(1, style_mask.shape[0], style_mask.shape[2], style_mask.shape[3])   # output -> (1, N, 1024, 1024)
                detailed_scale={
                "down": [[1.0]*facenum+[bgstyle_scale]*stylenum],         # 0
                "mid": [[face_scale1]*facenum+[bgstyle_scale]*stylenum],   # 1
                "up": {"block_0": [[face_scale2]*facenum+[bgstyle_scale]*stylenum,  # 2
                            [1.0]*facenum+[bgstyle_scale]*stylenum,    # 3
                            [0.0]*facenum+[bgstyle_scale]*stylenum],    # 4
                        "block_1": [[0.0]*facenum+[bgstyle_scale]*stylenum]},  # 5
                }
                scale2=detailed_scale

        self.pipe.set_ip_adapter_scale([scale1,scale2,0])

        print(composition_mask.shape,style_mask.shape)

        if self.lightning:
            steps=4
        elif self.hypersd:
            steps=1
        generator = torch.Generator().manual_seed(seed)
        
        image = self.pipe(
            prompt=prompt,
            ip_adapter_image=[composition_image, face_style_image,self.fake_image],
            cross_attention_kwargs={"ip_adapter_masks": [composition_mask, style_mask,composition_mask]},
            negative_prompt=negative_prompt,
            width=width,   # Make sure height and width are both multiples of 8
            height=height,
            num_inference_steps=steps, 
            num_images_per_prompt=1,
            guidance_scale=guidance,
            generator=generator,eta=1.0).images[0]
        return image
    
    def demo(self):
        #face_image = Image.open('data/ip/women_input.png')
        #style_images = [Image.open(f"data/ip/img{i}.png") for i in range(8)]
        face_image = Image.open('../data/image/boy.png')
        style_images = [Image.open(f"../data/image/room.png") for i in range(1)]
        image=self.generate_face_style(face_image,style_images,
            "a boy playing in the room",
            #"wonderwoman",
            "monochrome, lowres, bad anatomy, worst quality, low quality")
        image.save('data/ipadapter.png')

    def demo2(self):
        import random
        '''female_mask = Image.open("data/multiip/female_mask.png")
        male_mask = Image.open("data/multiip/male_mask.png")
        background_mask = Image.open("data/multiip/background_mask.png")
        
        ip_female_style = Image.open("data/multiip/ip_female_style.png")
        ip_male_style = Image.open("data/multiip/ip_male_style.png")
        ip_background = Image.open("data/multiip/ip_background.png")
        

        female_mask = Image.open("data/test/ip1_mask.png")
        male_mask = Image.open("data/test/ip2_mask.png")
        background_mask = Image.open("data/test/bg_mask.png")
        
        ip_female_style = Image.open("data/test/ip1.png")
        ip_male_style = Image.open("data/test/ip2.png")
        ip_background = Image.open("data/test/bg.png")
        '''

        '''
        female_mask = Image.open("data/test2/ip1_mask.png")
        male_mask = Image.open("data/test2/ip2_mask.png")
        background_mask = Image.open("data/test2/bg_mask.png")
        
        ip_female_style = Image.open("data/test2/ip1.png")
        ip_male_style = Image.open("data/test2/ip2.png")
        ip_background = Image.open("data/test2/bg.png")
        '''

        female_mask = Image.open("../data/tmp/ip0_mask.png")
        male_mask = Image.open("../data/tmp/ip1_mask.png")
        background_mask = Image.open("../data/tmp/bg_mask.png")
        
        ip_female_style = Image.open("../data/tmp/ip0.png")
        ip_male_style = Image.open("../data/tmp/ip1.png")
        ip_background = Image.open("../data/tmp/bg.png")
        
        for i  in range(1):
            image=self.generate_face_style_mask(
                #None,None,
                [ip_background],[background_mask],
                #None,None,
                [ip_female_style,ip_male_style],[female_mask, male_mask],
                'two frogs. high quality, cinematic photo, cinemascope, 35mm, film grain, highly detailed','',seed=random.randint(1,100000000))
            image.save(f'data/multiip_mask2_{i}.png')

    def demo3(self):
        female_mask = Image.open("data/multiip/female_mask.png")
        male_mask = Image.open("data/multiip/male_mask.png")
        background_mask = Image.open("data/multiip/background_mask.png")
        composition_mask = Image.open("data/multiip/composition_mask.png")
        scale_1 = {
            "down": [[0.0, 0.0, 1.0]],  # 权重1（重要） 三个mask只参考第三个
            "mid": [[0.0, 0.0, 1.0]],   # 权重2（较重要），三个mask只参考第三个
            "up": {"block_0": [[0.0, 0.0, 1.0], #权重3（一般） 三个mask只参考第三个
                        [1.0, 1.0, 1.0],   # 权重4（重要） 三个mask都参考
                        [0.0, 0.0, 1.0]], #权重5 三个mask只参考第三个
                    "block_1": [[0.0, 0.0, 1.0]]}, #无效权重 三个mask只参考第三个
                }
        ip_female_style = Image.open("data/multiip/ip_female_style.png")
        ip_male_style = Image.open("data/multiip/ip_male_style.png")
        ip_background = Image.open("data/multiip/ip_background.png")
        ip_composition_image = Image.open("data/multiip/ip_composition_image.png")
        
        image=self.generate_face_style_mask(
            #None,None,
            ip_composition_image,composition_mask,
            #None,None,
            [ip_background],[background_mask],
            #None,None,
            [ip_female_style,ip_male_style],[female_mask, male_mask],
            'high quality, cinematic photo, cinemascope, 35mm, film grain, highly detailed','')
        image.save('data/multiip_mask.png')


        
# ---- 统一 T2I 接口：generate / generate_from_image / generate_face_style（AGENT.md 3.2）----
# 供仅支持 T2I 的模型（如 sdxl_lightning/sd15）包装成与 ipadapter 同义的 generate_face_style 调用
class _UnifiedT2IWrapper:
    """包装仅有 generate/generate_style 的本地模型，使其支持 generate_face_style(face_image, style_images, prompt, ...)。"""

    def __init__(self, inner):
        self._inner = inner

    def generate_face_style(
        self,
        face_image,
        style_images,
        prompt,
        negative_prompt="",
        steps=4,
        seed=0,
        guidance=1.0,
        width=1024,
        height=1024,
        face_scale=1.0,
        bgstyle_scale=1.0,
        onlystyle=False,
        **kwargs,
    ):
        if face_image is None and (style_images is None or len(style_images) == 0):
            return self._inner.generate(
                prompt, negative_prompt, guidance=guidance, seed=seed, width=width, height=height,
                num_inference_steps=steps, **kwargs
            )
        if hasattr(self._inner, "generate_style") and style_images and len(style_images) > 0:
            ref = style_images[0] if isinstance(style_images, (list, tuple)) else style_images
            return self._inner.generate_style(
                ref, prompt, negative_prompt=negative_prompt, guidance=guidance, seed=seed,
                width=width, height=height, num_inference_steps=steps, **kwargs
            )
        return self._inner.generate(
            prompt, negative_prompt, guidance=guidance, seed=seed, width=width, height=height,
            num_inference_steps=steps, **kwargs
        )

    def generate_face_style_mask(
        self, style_images, style_masks, face_images, face_masks, prompt, negative_prompt,
        steps=20, seed=0, guidance=1.0, width=1024, height=1024, face_scale=1.0, bgstyle_scale=1.0,
    ):
        raise NotImplementedError("generate_face_style_mask 仅支持 ipadapter 后端，请选用 ipadapter 或 commercial 后端。")


# ---- 开源后端：Flux（flux_dev / flux_schnell）----
class _FluxT2IAdapter:
    """Flux.1 Schnell/Dev 文生图，统一成 generate_face_style。无参考时走 T2I，有参考时可用首图做 I2I（若管线支持）。"""

    def __init__(self, model_id="flux_schnell", device=None, savememory=True, **kwargs):
        self.model_id = model_id
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._savememory = savememory
        self._pipe = None
        self._load_pipe(**kwargs)

    def _load_pipe(self, **kwargs):
        try:
            from diffusers import FluxPipeline
        except ImportError:
            raise RuntimeError("使用 Flux 后端需要 diffusers 支持 FluxPipeline，请升级: pip install -U diffusers")
        repo = "black-forest-labs/FLUX.1-schnell" if self.model_id == "flux_schnell" else "black-forest-labs/FLUX.1-dev"
        dtype = torch.bfloat16 if getattr(torch, "bfloat16", None) else torch.float16
        self._pipe = FluxPipeline.from_pretrained(repo, torch_dtype=dtype, **kwargs)
        if self._savememory:
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe.to(self._device)

    def generate_face_style(
        self,
        face_image,
        style_images,
        prompt,
        negative_prompt="",
        steps=4,
        seed=0,
        guidance=0.0,
        width=1024,
        height=1024,
        face_scale=1.0,
        bgstyle_scale=1.0,
        onlystyle=False,
        **kwargs,
    ):
        # Flux Schnell 建议 guidance_scale=0, steps=4
        if self.model_id == "flux_schnell":
            guidance = 0.0
            steps = min(max(steps, 1), 4)
        gen = torch.Generator(device=self._device).manual_seed(seed) if seed else None
        out = self._pipe(
            prompt,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=gen,
            height=height,
            width=width,
            **kwargs,
        )
        return out.images[0]


# ---- 开源后端：Z-Image-Turbo（阿里）----
class _ZImageTurboAdapter:
    """阿里 Z-Image-Turbo 文生图占位实现。接入时在此加载对应 diffusers 管线并实现 generate_face_style。"""

    def __init__(self, device=None, **kwargs):
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe = None
        try:
            # 若有官方/社区 diffusers 管线，在此 from_pretrained
            from diffusers import AutoPipelineForText2Image
            self._pipe = AutoPipelineForText2Image.from_pretrained(
                "Alibaba-Nano/Z-Image-Turbo", torch_dtype=torch.float16, **kwargs
            )
            self._pipe.to(self._device)
        except Exception as e:
            logger.warning("Z-Image-Turbo 加载失败，将使用占位返回: %s", e)
            self._pipe = None

    def generate_face_style(
        self,
        face_image,
        style_images,
        prompt,
        negative_prompt="",
        steps=4,
        seed=0,
        guidance=1.0,
        width=1024,
        height=1024,
        face_scale=1.0,
        bgstyle_scale=1.0,
        onlystyle=False,
        **kwargs,
    ):
        if self._pipe is None:
            # 占位：返回纯色图，便于联调
            from PIL import Image
            return Image.new("RGB", (width, height), color=(128, 128, 128))
        gen = torch.Generator(device=self._device).manual_seed(seed) if seed else None
        out = self._pipe(
            prompt,
            negative_prompt=negative_prompt or None,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=gen,
            height=height,
            width=width,
            **kwargs,
        )
        return out.images[0]

    def generate_face_style_mask(self, *args, **kwargs):
        raise NotImplementedError("Z-Image-Turbo 暂不支持 generate_face_style_mask。")


# ---- 商用后端：Nano Banana Pro / 通义 / 字节 / Kling 统一 HTTP 适配器 ----
class _CommercialT2IAdapter:
    """商用 T2I/I2I 统一适配：按 model_id 读配置并调用对应 API。"""

    def __init__(self, model_id, **kwargs):
        from config_loader import get_t2i_config
        self.model_id = (model_id or "").strip().lower()
        self._api_key, self._base_url, self._model_name = get_t2i_config(self.model_id)
        self._client = None
        if self._api_key:
            self._prepare_client()

    def _prepare_client(self):
        if self.model_id == "tongyi":
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._api_key, base_url=self._base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1")
            except Exception as e:
                logger.warning("通义 T2I client 初始化失败: %s", e)
        elif self.model_id in ("nano_banana_pro", "bytedance", "kling"):
            # 占位：按各厂商 API 文档实现 HTTP 调用
            self._client = "placeholder"

    def generate_face_style(
        self,
        face_image,
        style_images,
        prompt,
        negative_prompt="",
        steps=4,
        seed=0,
        guidance=1.0,
        width=1024,
        height=1024,
        face_scale=1.0,
        bgstyle_scale=1.0,
        onlystyle=False,
        **kwargs,
    ):
        from PIL import Image
        if not self._api_key:
            logger.warning("商用 T2I 未配置 api_key，返回占位图。请设置 T2I_%s_API_KEY", self.model_id.upper().replace("-", "_"))
            return Image.new("RGB", (width, height), color=(128, 128, 128))
        # 商用 API 调用占位：T2I 走各厂商 imagen/wanx 等接口
        if self.model_id == "tongyi" and self._client and self._client != "placeholder":
            return self._tongyi_t2i(prompt, negative_prompt, width, height, seed)
        return Image.new("RGB", (width, height), color=(128, 128, 128))

    def _tongyi_t2i(self, prompt, negative_prompt, width, height, seed):
        """通义万相 T2I：通过 dashscope ImageSynthesis 调用 wanx。"""
        from PIL import Image
        import io
        import requests
        try:
            import dashscope
            from dashscope import ImageSynthesis
            dashscope.api_key = self._api_key
            size_map = ["1024*1024", "720*1280", "1280*720"]
            size = "%d*%d" % (width, height) if (width, height) in [(1024, 1024), (720, 1280), (1280, 720)] else "1024*1024"
            if size not in size_map:
                size = "1024*1024"
            kw = {"model": self._model_name or "wanx-v1", "prompt": prompt, "n": 1, "size": size}
            if negative_prompt:
                kw["negative_prompt"] = negative_prompt
            resp = ImageSynthesis.call(**kw)
            if resp and getattr(resp, "status_code", None) == 200 and getattr(resp, "output", None):
                out = resp.output
                results = getattr(out, "results", None) or []
                if results and getattr(results[0], "url", None):
                    r = requests.get(results[0].url, timeout=30)
                    r.raise_for_status()
                    img = Image.open(io.BytesIO(r.content)).convert("RGB")
                    if (img.width, img.height) != (width, height):
                        img = img.resize((width, height), Image.Resampling.LANCZOS)
                    return img
        except Exception as e:
            logger.warning("通义 T2I 调用失败: %s", e)
        return Image.new("RGB", (width, height), color=(128, 128, 128))

    def generate_face_style_mask(self, *args, **kwargs):
        raise NotImplementedError("商用后端暂不支持 generate_face_style_mask，请使用 ipadapter。")


# ---- 阶段5：文生图/图生图模型可插拔入口；backend + model_id 多后端（AGENT.md 3.2）----
def get_t2i_model(backend=None, model_id=None, name=None, **kwargs):
    """
    按 backend + model_id 或兼容旧参 name 返回 T2I/I2I 模型实例。
    - backend: "open_source" | "commercial"
    - model_id: 开源 flux_dev, flux_schnell, z_image_turbo, sdxl_lightning, sd15, ipadapter；
                商用 nano_banana_pro, tongyi, bytedance, kling
    - name: 兼容旧用法，等价于 model_id；当 backend/model_id 未传时从环境变量 T2I_BACKEND、T2I_MODEL 读取。
    返回对象均支持 generate_face_style(face_image, style_images, prompt, negative_prompt, ...)。
    """
    # 从环境变量补全
    backend = backend or os.environ.get("T2I_BACKEND", "").strip().lower()
    model_id = (model_id or name or os.environ.get("T2I_MODEL", "sdxl_lightning")).strip().lower()
    if not backend:
        backend = T2I_BACKEND_COMMERCIAL if model_id in T2I_COMMERCIAL_MODEL_IDS else T2I_BACKEND_OPEN_SOURCE
    if name and not model_id:
        model_id = name.strip().lower()

    if model_id in ("ip_adapter", "ipadapter"):
        model_id = "ipadapter"

    if backend == T2I_BACKEND_COMMERCIAL:
        if model_id not in T2I_COMMERCIAL_MODEL_IDS:
            model_id = T2I_COMMERCIAL_MODEL_IDS[0] if T2I_COMMERCIAL_MODEL_IDS else "nano_banana_pro"
        return _CommercialT2IAdapter(model_id=model_id, **kwargs)

    # open_source
    if model_id == "flux_dev":
        return _FluxT2IAdapter(model_id="flux_dev", **kwargs)
    if model_id == "flux_schnell":
        return _FluxT2IAdapter(model_id="flux_schnell", **kwargs)
    if model_id == "z_image_turbo":
        return _ZImageTurboAdapter(**kwargs)
    if model_id == "sdxl_lightning":
        inner = sdxl_lightning_model(**kwargs)
        return _UnifiedT2IWrapper(inner)
    if model_id == "sd15":
        inner = sd15_model(**kwargs)
        return _UnifiedT2IWrapper(inner)
    if model_id == "ipadapter":
        return ipadapter_model_multi_adapter(lightning=True, **kwargs)

    # 未识别时回退
    if model_id in T2I_OPEN_SOURCE_MODEL_IDS:
        return get_t2i_model(backend=T2I_BACKEND_OPEN_SOURCE, model_id=model_id, **kwargs)
    return get_t2i_model(backend=T2I_BACKEND_OPEN_SOURCE, model_id="sdxl_lightning", **kwargs)


def regenerate_template():
    from sd_prompt.sdxl_styles import sdxl_styles
    sl_model=sdxl_lightning_model(instantstyle=False,controlnet=False,savememory=False)
    #newtemplate=sdxl_styles(defaultdir='samples',model=sl_model,template_prompt='a sitting cat')
    newtemplate=sdxl_styles(defaultdir='man',model=sl_model,template_prompt='full body of man')
    newtemplate=sdxl_styles(defaultdir='woman',model=sl_model,template_prompt='full body of woman')
    #newtemplate=sdxl_styles(defaultdir='natural_scene',model=sl_model,template_prompt='mountain,river and forest')

    
if __name__ =="__main__":
    #regenerate_template()

    #sys.exit()

    ip_model=ipadapter_model_multi_adapter(lightning=True,hypersd=False)
    #ip_model.demo()
    ip_model.demo2()
    #ip_model.demo3()

    sys.exit()
    
    
    #sdxl_demo("A boy")
    #playground_demo()

    image = Image.open("/mnt/glennge/diffusers/visual_generation/InstantStyle/assets/0.jpg").convert('RGB')

    #sl_model=sdxl_lightning_model(instantstyle=True,controlnet=True)
    #cl_image = Image.open("/mnt/glennge/diffusers/visual_generation/InstantStyle/assets/yann-lecun.jpg").convert('RGB')
    #sl_model.generate_style(image).save('data/res.png')

    hp_model=hyper_sd_model(instantstyle=True,controlnet=True)
    hp_model.generate_style(image).save('data/res.png')

    
    #upmd=upscale_model()
    #image=Image.open('1.jpg')
    #upmd.generate(image).show()
    
    
    pass
    
    
        