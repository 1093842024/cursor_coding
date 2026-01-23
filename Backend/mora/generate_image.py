import torch,sys,os,cv2
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler,DDIMScheduler,TCDScheduler
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from PIL import Image
import numpy as np
import gradio as gr
import sys,os
device = 'cuda' if torch.cuda.is_available() else 'cpu'
curdir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(curdir)

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
    
    
        