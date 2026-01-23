import torch,sys,os,datetime,random,time,shutil,hashlib
from pathlib import Path
from PIL import Image
import numpy as np
import gradio as gr
from mora.generate_image  import sdxl_lightning_model,sd15_model,ipadapter_model_multi_adapter
from mora.sd_prompt.sdxl_styles import sdxl_style_template,sdxl_styles

from tts_utils import CH_LANGUAGE_ID,EN_LANGUAGE_ID,translate_ch_to_en,translate_en_to_ch,generate_text_audio
from video_utils import comb_video
from llm import _generate_response

os.environ["no_proxy"] = "localhost,0.0.0.0,:8082"
style_type=['origin IP-Adapter','only style block','style+layout block']
MAX_SEED = np.iinfo(np.int32).max
DEFAULT_NEG_PROMPT="bad hands,bad face,distort limbs,text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry"


DeFault_BGM_path=[file for file in os.listdir('/mnt/glennge/MoneyPrinter/source/Songs/')]
Max_fenjin_num=20

def generate_fgmaskimg(position,W,H):
    maskimg=np.array(Image.new('RGB',(W,H),color=(0,0,0)))
    p1,p2,pose,size=position.split('-')
    p1,p2,pose,size=int(p1),int(p2),float(pose),int(size)
    basesize1=(int(W/3),int(H/3))
    basesize2=(int(W/9),int(H/9))
    x_index1 = int(p1/3)
    y_index1 = p1%3
    x_index2 = int(p2/3)
    y_index2 = p2%3
    center_x= int(x_index1*basesize1[0]+(0.5+x_index2)*basesize2[0])
    center_y= int(y_index1*basesize1[1]+(0.5+y_index2)*basesize2[1])

    scale=max(min(3,random.uniform(-0.2,0.2)+size),0.5)

    w_2 = int(scale*basesize1[0]/pose/2)
    h_2 = int(scale*basesize1[1]*pose/2)
    x1,x2=max(0,center_x-w_2),min(W,center_x+w_2)
    y1,y2=max(0,center_y-h_2),min(W,center_y+h_2)
    #print(position,x1,y1,x2,y2)
    maskimg[y1:y2,x1:x2,:]=255
    maskimg=Image.fromarray(maskimg)
    return maskimg

def get_default_maskimg():
    maskimgs=[]
    ip_position_infos=[]
    for p1 in range(0,9):
        for p2 in range(0,9):
            if p1<3 and p2<5:continue
            if p1>5 and p2>2:continue
            if p1 in [0,3,6] and p2 in [0,3,6]:continue  # 81宫格最外边的一圈格子，不选取
            if p1 in  [2,5,8] and p2 in [2,5,8]:continue
            for pose in [0.7,1,1.4]:  # 竖、正方、横
                for size in [3,2,1]: # 小，中、大
                    if p1!=4 and size==3:continue   #只有最中心的9宫格，可以生成大尺寸图
                    if p1 in [0,1,2,6,7,8] and p2 in [1,4,7] and size==2:continue #只有次中心，可以生成中尺寸图
                    if p1 in [3,5] and p2==4 and size==2:continue #只有次中心，可以生成中尺寸图
                    
                    ip_position_infos.append(f'{p1}-{p2}-{pose}-{size}')
                    maskimg=generate_fgmaskimg(f'{p1}-{p2}-{pose}-{size}',1024,1024) 
                    maskimgs.append((maskimg,f'{p1}-{p2}-{pose}-{size}')) 
    return maskimgs,ip_position_infos


Default_mask_img,Default_ip_position_infos=get_default_maskimg()
 
Clay_lora_hint_word='Clay Animation, Clay'


Default_story_prompt="""作为一个儿童故事畅销作者，按如下主题创作一篇300～500字，8～10个段落的儿童故事。要求情节清晰、简单易懂、引人入胜、让小朋友十分喜欢。
故事主题: 坐井观天的故事，讲述两个小青蛙，一个坐井观天，一个勇敢探索的故事
"""

Default_person_scene_prompt="""请根据以下故事内容总结故事的主要角色和主要场景，按照以下格式总结：
'''
[Characters]
角色i名称:xxx;外形描述:xxxx
[Scenes]
场景j名称:xxx;画面描述:xxxx
'''
其中角色名称和场景名称都为中文输出，外形描述和画面描述都为英文输出。
其中，外形描述主要用于文字描述生成角色画面，需要重点描述角色的外貌、外形特点，可以从皮肤颜色、长相、五官、四肢、穿着等角度描述，而非性格和行为特点。
其中，场景画面描述主要用于文字描述生成场景画面，场景画面不包含角色和角色行为，可以通过合理的联想创作，重点描述和故事匹配的场景地点、景观、景色、天气、植被、建筑、色彩等内容。
其中i代表角色序号，从0开始。j代表场景序号，从0开始。
Story content:
'''
{story}
'''"""

DeFault_fenjin_prompt="""请根据以下主要故事角色、主要故事场景和故事内容，将故事内容的每个段落设计成一个分镜剧本,需包含7～9个分镜，每个分镜都包括多个画面，每个画面都包括四个内容：旁白内容，角色，场景，内容描述，并按照以下格式总结：
'''
分镜i画面j->旁白内容:xxx;角色:xxx;场景:xxx;画面prompt:xxx.
'''
其中，每个段落一个分镜，然后将一个段落依次拆分为一到多个画面，连接起来可以完整的表达整个段落的故事。
其中，i代表段落序号，从1开始。j代表画面序号，从1开始。
其中，旁白内容、角色、场景内容为中文输出。
其中，画面prompt用英文输出。
其中，旁白内容用于生成解说朗读的音频，可根据该段落内容做适当的拓展，增加解说的有趣性和口语化，不少于两句话、12个词。
其中，角色可以为None和有角色两种，其中有角色情况下为故事主要角色中的其中一个角色或多个角色（多个角色用,分隔）。
其中，场景必须为主要故事场景中的其中一个场景，不能为None或其他。
其中，画面prompt为生成图像的英文提示语，为纯英文描述，只包含画面内容的主要词汇，不直接与故事内容相同。1）有角色情况：主要描述角色的年纪、性别、身份、穿着、外貌和行为，2）无角色None情况：只描绘场景的具体景观、时代、地点、景色、天气、植被、建筑、色彩等。

主要故事角色和故事场景为：
'''
{person_scene}
'''
故事内容：
'''
{story}
'''"""

DeFault_person_scene_text='''[Characters]
角色0名称:小蓝;外形描述:Small and blue frog, with big round eyes and a smooth skin texture, dressed in a simple green outfit that matches the color of the well.
角色1名称:小绿;外形描述:Small and green frog, with bright eyes and a smooth skin texture, dressed in a simple green outfit that matches the color of the well.
[Scenes]
场景0名称:古老井口外;画面描述:An deep well with moss-covered stones.
场景3名称:古老井口内;画面描述:An sight from inside of deep well with moss-covered stones.
场景1名称:外面的世界;画面描述:A vast expanse of lush green meadows, dotted with colorful flowers and a clear, sparkling stream under a bright, sunny sky.
场景2名称:森林通向外面的小路;画面描述:A winding path from forest  to the outside world.'''

DeFault_fenjin_text='''分镜1画面1->旁白内容:在一个遥远的森林深处，有一口古老的井，井里住着两只小青蛙。;角色:小蓝,小绿;场景:古老井口内;画面prompt:Frogs in an ancient well.

分镜1画面2->旁白内容:小蓝总是喜欢坐在井底，抬头看着天空，而小绿却总是梦想着外面的世界。;角色:小蓝;场景:古老井口内;画面prompt:Blue frog looking up at the sky.

分镜1画面3->旁白内容:小绿总是好奇地问外面的世界是什么样的，但小蓝却认为井底的生活已经很好了。;角色:小绿;场景:古老井口内;画面prompt:Green frog curious about the outside world.

分镜2画面1->旁白内容:终于有一天，小绿鼓起勇气，决定跳出井口，去探索外面的世界。;角色:小绿;场景:古老井口外;画面prompt:Green frog jumping out of the well.

分镜2画面2->旁白内容:小蓝害怕地摇摇头，认为外面的世界太危险，但小绿没有被吓倒。;角色:小蓝;场景:古老井口内;画面prompt:Blue frog shaking head in fear.

分镜3画面1->旁白内容:小绿跳出井口后，发现外面的世界真的很美，有绿油油的草地和五彩斑斓的花朵。;角色:None;场景:外面的世界;画面prompt:Lush meadows and colorful flowers.

分镜3画面2->旁白内容:还有清澈的小溪，小绿兴奋地跳来跳去，感受着新鲜的空气和温暖的阳光。;角色:小绿;场景:外面的世界;画面prompt:Green frog jumping in the meadow.

分镜4画面1->旁白内容:小绿在外面的世界遇到了很多新朋友，他们告诉小绿，外面的世界还有很多奇妙的地方。;角色:小绿,小鸟,小兔子,小松鼠;场景:外面的世界;画面prompt:Green frog meeting new friends.

分镜5画面1->旁白内容:小绿在外面的世界玩得很开心，但他没有忘记井底的小蓝。;角色:小绿;场景:外面的世界;画面prompt:Green frog remembering blue frog.

分镜5画面2->旁白内容:他决定回到井底，告诉小蓝外面的世界有多么美好。;角色:小绿;场景:古老井口外;画面prompt:Green frog returning to the well.

分镜6画面1->旁白内容:小绿回到井底，兴奋地向小蓝描述外面的世界。;角色:小绿,小蓝;场景:古老井口内;画面prompt:Green frog describing the outside world to blue frog.

分镜6画面2->旁白内容:小蓝听了小绿的描述，也开始对外面的世界产生了好奇。;角色:小蓝;场景:古老井口内;画面prompt:Blue frog becoming curious.

分镜7画面1->旁白内容:最后，小蓝鼓起勇气，和小绿一起跳出了井口。;角色:小蓝,小绿;场景:古老井口外;画面prompt:Blue and green frogs jumping out together.

分镜7画面2->旁白内容:他们一起探索外面的世界，发现了很多新奇的事物，也学到了很多知识。;角色:小蓝,小绿;场景:外面的世界;画面prompt:Frogs exploring and learning.

分镜8画面1->旁白内容:这个故事告诉我们，不要害怕未知的事物，要勇敢地去探索。;角色:None;场景:外面的世界;画面prompt:Exploring the unknown.

分镜8画面2->旁白内容:只有通过探索，我们才能发现更多的美好和学到更多的知识。同时，我们也要关心身边的朋友，鼓励他们一起成长。;角色:None;场景:外面的世界;画面prompt:Encouraging growth and exploration.'''



def generate_image_gr_demo(t2imodel):
    def get_bgm_song(bgmlist):return '/mnt/glennge/MoneyPrinter/source/Songs/'+bgmlist

    def get_select_template_stylename(evt: gr.SelectData):
        print(f'selected {evt.value} at {evt.index} from {evt.target}')
        stylename=evt.value['caption']
        return stylename

    def change_template_type(template_type):
        if template_type=='cat':
            return gr.update(choices=sdxl_style_template.get_style_name()),gr.update(value=sdxl_style_template.get_exampleimg_path())
        else:
            sdxl_style_template_new=sdxl_styles(defaultdir=template_type)
            return gr.update(choices=sdxl_style_template_new.get_style_name()),gr.update(value=sdxl_style_template_new.get_exampleimg_path())
        #stylename,style_example_img
    
    def apply_select_stylename(stylename,person_text,scene_text):
        pos1,neg1,imgpath1=sdxl_style_template.get_name_style_prompt(stylename,person_text.split(':')[-1])
        pos2,neg2,imgpath2=sdxl_style_template.get_name_style_prompt(stylename,scene_text.split(':')[-1])
        return neg1,pos1,neg2,pos2,person_text.split(':')[-1],scene_text.split(':')[-1]

    def get_select_template_prompt(person_text,scene_text,evt: gr.SelectData):
        print(f'selected {evt.value} at {evt.index} from {evt.target}')
        stylename=evt.value['caption']
        pos1,neg1,imgpath1=sdxl_style_template.get_name_style_prompt(stylename,person_text.split(':')[-1])
        pos2,neg2,imgpath2=sdxl_style_template.get_name_style_prompt(stylename,scene_text.split(':')[-1])
        return neg1,pos1,neg2,pos2,stylename,person_text.split(':')[-1],scene_text.split(':')[-1]

    def get_select_name_template_prompt(text,stylename):
        pos1,neg1,imgpath1=sdxl_style_template.get_name_style_prompt(stylename,text.split(':')[-1])
        return neg1,pos1,text.split(':')[-1]
    
    def register_img_template(image,stylename):
        if image is not None:
            sdxl_style_template.update_example_img(stylename,image)
            gr.Info('register img template success,try refresh!')
        else:
            gr.Warning('no image to register!')
    
    def register_img_to_candi(images,name,gallery):
        if name is None or len(name)<1:raise gr.Error('注册名称不能为空')
        new_gallery=[]
        if gallery is not None:
            for gallary_data in gallery:
                tmpimage,tmpname = gallary_data
                new_gallery.append((tmpimage,tmpname))
        if images is not None:
            for image_data in images:
                tmpimage,tmpname=image_data
                new_gallery.append((tmpimage,name))
        return new_gallery

    def generate_no_ref(prompt_neg,prompt_pos,seed,num_inference_steps,guidance,width,height,num_img_per_prompt=1):
        all_images=[]
        for _ in range(num_img_per_prompt):
            if seed==0:seed=random.randint(1,MAX_SEED)
            image = t2imodel.generate_face_style(None,None,prompt_pos,prompt_neg,steps=num_inference_steps,seed=seed,guidance=guidance,width=width,height=height)
            all_images.append((image,prompt_pos))
            seed=random.randint(1,MAX_SEED)
        return all_images

    def generate_adapter_person(prompt_neg,prompt_pos,person_ref,seed,num_inference_steps,guidance,width,height,num_img_per_prompt,scale):
        all_images=[]

        if person_ref is not None and isinstance(person_ref,list): #兼容 gallery 输入
            new_person_ref=[]
            for pr in person_ref:
                imagepath,label=pr
                new_person_ref.append(Image.open(imagepath).convert('RGB'))
            person_ref=new_person_ref

        for _ in range(num_img_per_prompt):
            if seed==0:seed=random.randint(1,MAX_SEED)
            image = t2imodel.generate_face_style(person_ref,None,prompt_pos,prompt_neg,steps=num_inference_steps,seed=seed,guidance=guidance,width=width,height=height,face_scale=scale,bgstyle_scale=scale)
            all_images.append((image,prompt_neg))
            seed=random.randint(1,MAX_SEED)
        return all_images
    
    def generate_adapter_scene(scene_prompt_neg,scene_prompt_pos,style_ref,seed,num_inference_steps,guidance,width,height,num_img_per_prompt,scale):
        all_images=[]
        if style_ref is not None and isinstance(style_ref,list):
            new_style_ref=[]
            for pr in style_ref:
                imagepath,label=pr
                new_style_ref.append(Image.open(imagepath).convert('RGB'))
            style_ref=new_style_ref

        for _ in range(num_img_per_prompt):
            if seed==0:seed=random.randint(1,MAX_SEED)
            image = t2imodel.generate_face_style(None,style_ref,scene_prompt_pos,scene_prompt_neg,steps=num_inference_steps,seed=seed,guidance=guidance,width=width,height=height,face_scale=scale,bgstyle_scale=scale)
            all_images.append((image,scene_prompt_pos))
            seed=random.randint(1,MAX_SEED)
        return all_images
    
    def get_ref_imgs(text,gallery):
        select_type_imgs={}
        select_type_prompts={}
        if gallery is not None and len(text)>0:
            multi_text=text.split(',')
            for sg_text in multi_text:
                for img_label in gallery:
                    image,label=img_label
                    if sg_text.lower() in label.lower():
                        if sg_text not in select_type_imgs:
                            select_type_imgs[sg_text]=[]
                        select_type_imgs[sg_text].append(Image.open(image).convert('RGB'))
                        select_type_prompts[sg_text]=label.replace(sg_text+':','')
        return select_type_imgs,select_type_prompts


    def generate_adapter_person_scene(prompt_pos,person_text,scene_text,gallery_person,gallery_scene,seed,num_inference_steps,guidance,width,height,num_img_per_prompt,
        person_scale,bg_scale):
        all_images=[]
        for _ in range(num_img_per_prompt):
            if seed==0:seed=random.randint(1,MAX_SEED)
            person_ref_imgs,person_type_prompts = get_ref_imgs(person_text,gallery_person)

            tmp_person_ref=[]
            for person in person_text.split(','):
                if person in person_ref_imgs:
                    tmp_person_ref.extend(person_ref_imgs[person])
            person_ref_imgs=tmp_person_ref if len(tmp_person_ref) else None
            
            scene_ref_imgs,scene_type_prompts = get_ref_imgs(scene_text,gallery_scene) 
            scene_ref_imgs=scene_ref_imgs[scene_text] if scene_text in scene_ref_imgs else None

            print(person_ref_imgs,scene_ref_imgs)
            image = t2imodel.generate_face_style(person_ref_imgs,scene_ref_imgs,prompt_pos,DEFAULT_NEG_PROMPT,steps=num_inference_steps,seed=seed,guidance=guidance,width=width,height=height,
                face_scale=person_scale,bgstyle_scale=bg_scale)
            all_images.append((image,prompt_pos))
            seed=random.randint(1,MAX_SEED)
        return all_images
    
    def generate_bgmaskimg(fgmaskimgs):
        w,h=fgmaskimgs[0].size
        new_bgimg=Image.new('RGB',(w,h),color=(255,255,255))
        fg_maskimg=np.array(Image.new('RGB',(w,h),color=(0,0,0)))
        for maskimg in fgmaskimgs:fg_maskimg+=np.array(maskimg)
        new_bgimg=Image.fromarray(np.array(new_bgimg)-fg_maskimg)
        return new_bgimg

    def get_person_position_mask(mask_img_select,person_text,person_ref_imgs):
        face_masks=[]
        face_images=[]
        for ref_mask,person in zip(mask_img_select,person_text.split(',')):
            ref_mask_path,label=ref_mask
            ref_img=random.choice(person_ref_imgs[person])
            face_images.append(ref_img)
            face_masks.append(Image.open(ref_mask_path).convert('RGB'))
        bg_mask=generate_bgmaskimg(face_masks).convert('RGB')
        #print(len(face_masks),bg_mask.size)
        return face_masks,face_images,bg_mask

    def generate_adapter_person_scene_with_mask_pro(prompt_pos,person_text,scene_text,gallery_person,gallery_scene,seed,num_inference_steps,guidance,width,height,num_img_per_prompt,
        mask_img_select,person_scale,bg_scale):
        all_images=[]

        person_ref_imgs,person_type_prompts = get_ref_imgs(person_text,gallery_person)
        scene_ref_imgs,scene_type_prompts = get_ref_imgs(scene_text,gallery_scene) 
        if len(list(person_ref_imgs.keys()))==len(person_text.split(',')):
            for _ in range(num_img_per_prompt):
                #if seed==0:seed=random.randint(1,MAX_SEED)
                bg_style=random.choice(scene_ref_imgs[scene_text]) if scene_text in scene_ref_imgs else None
                face_masks,face_images,bg_mask=get_person_position_mask(mask_img_select,person_text,person_ref_imgs)

                if not os.path.exists('data/tmp/'):os.makedirs('data/tmp/')
                for i,(faceimg,facemask) in enumerate(zip(face_images,face_masks)):
                    faceimg.save(f'data/tmp/ip{i}.png')
                    facemask.save(f'data/tmp/ip{i}_mask.png')
                bg_style.save(f'data/tmp/bg.png')
                bg_mask.save(f'data/tmp/bg_mask.png')

                ipref_imgs,ipref_masks=[],[]
                for i in range(len(face_images)):
                    ipref_imgs.append(Image.open(f'data/tmp/ip{i}.png'))
                    ipref_masks.append(Image.open(f'data/tmp/ip{i}_mask.png'))
                bg_style=Image.open(f'data/tmp/bg.png')
                bg_mask=Image.open(f'data/tmp/bg_mask.png')

                image = t2imodel.generate_face_style_mask(
                    [bg_style],[bg_mask],
                    ipref_imgs,ipref_masks,
                    prompt_pos,'',seed=seed,
                    steps=num_inference_steps,guidance=guidance,width=width,height=height,face_scale=person_scale,bgstyle_scale=bg_scale)
                all_images.append((image,prompt_pos))
                seed=random.randint(1,MAX_SEED)
        else:
            gr.Warning('某个角色名不在角色库中')
        return all_images
    
    def generate_single_fenjin_img(gallery_person,gallery_scene,seed,num_inference_steps,guidance,width,height,num_img_per_prompt,
        mask_img_select,person_scale,bg_scale,
        ip,scene,prompt):
        if prompt is not None and len(prompt)>0 and ip is not None and scene is not None:
            if mask_img_select is not None and len(mask_img_select)==len(ip.split(',')):
                images=generate_adapter_person_scene_with_mask_pro(prompt,ip,scene,gallery_person,gallery_scene,seed,num_inference_steps,guidance,width,height,num_img_per_prompt,
                    mask_img_select,person_scale,bg_scale)
                return gr.update(value=images,visible=True)
            else:
                images=generate_adapter_person_scene(prompt,ip,scene,gallery_person,gallery_scene,seed,num_inference_steps,guidance,width,height,num_img_per_prompt,
                    person_scale,bg_scale)
                return gr.update(value=images,visible=True)
        else:
            return None

    def generate_all_fenjin_imgs(gallery_person,gallery_scene,seed,num_inference_steps,guidance,width,height,num_img_per_prompt=1,
                                 *ip_scene_prompt):
        if seed==0:seed=random.randint(1,MAX_SEED)
        fenjin_images=[gr.update(value=None,visible=False)]*Max_fenjin_num
        ips,scenes,prompts=[],[],[]
        step=len(ip_scene_prompt)/3
        for i,data in enumerate(ip_scene_prompt):
            if int(i/step)==0:
                ips.append(data)
            elif int(i/step)==1:
                scenes.append(data)
            else:
                prompts.append(data)
        print(ips,scenes,prompts)
        for i,(ip,scene,prompt) in enumerate(zip(ips,scenes,prompts)):
            if prompt is not None and len(prompt)>0 and ip is not None and scene is not None:
                images=generate_adapter_person_scene(prompt,ip,scene,gallery_person,gallery_scene,seed,num_inference_steps,guidance,width,height,num_img_per_prompt,
                    0.5,0.5)
                fenjin_images[i]=gr.update(value=images,visible=True)
                yield fenjin_images
    
    def generate_single_fenjin_video(fadeinfadeout,audio,subtitle,img):
        outputdir='data/video/'
        if audio is not None and subtitle is not None and img is not None and len(audio)>0 and len(subtitle)>0:
            imagefiles=[]
            for data in img:
                imgpath,label=data
                imagefiles.append(imgpath)
            allimagefiles=';'.join(imagefiles)
            video=comb_video.image_to_video_with_audio_subtitle(allimagefiles,audio,subtitle,outputdir,f'video_{i}.mp4',fadeinfadeout)
            print(video,subtitle)
            return gr.update(value=video,visible=True)
        else:
            return None


    def generate_all_fenjin_video(fadeinfadeout,*audio_subtitle_img):
        outputdir='data/video/'
        if not os.path.exists(outputdir):os.makedirs(outputdir)
        fenjin_video=[None]*Max_fenjin_num
        audios,subtitles,imgs=[],[],[]
        step=len(audio_subtitle_img)/3
        for i,data in enumerate(audio_subtitle_img):
            if int(i/step)==0:
                audios.append(data)
            elif int(i/step)==1:
                subtitles.append(data)
            else:
                imgs.append(data)
        print(audios,subtitles,imgs)
        for i,(audio,subtitle,img) in enumerate(zip(audios,subtitles,imgs)):
            if audio is not None and subtitle is not None and img is not None and len(audio)>0 and len(subtitle)>0:
                imagefiles=[]
                for data in img:
                    imgpath,label=data
                    imagefiles.append(imgpath)
                allimagefiles=';'.join(imagefiles)
                video=comb_video.image_to_video_with_audio_subtitle(allimagefiles,audio,subtitle,outputdir,f'video_{i}.mp4',fadeinfadeout)
                fenjin_video[i]=gr.update(value=video,visible=True)
                print(i,video,subtitle)
                #yield fenjin_video
                #fenjin_video.append(gr.update(value=(video,subtitle),visible=True))
                #fenjin_video.append(gr.update(value=video,visible=True))
            else:
                fenjin_video[i]=None
                #fenjin_video.append(gr.update(value=None,visible=False))
        return fenjin_video
    
    def generate_single_tts(pb_tts_id,ps_tts_id,language,rate,audiotext):
        if audiotext is not None and len(audiotext)>1:
            if language!='en':
                tmpat=translate_en_to_ch(audiotext)
                if len(tmpat)>1:audiotext=tmpat
            audiofile,subtitlefile=generate_text_audio(audiotext,pb_tts_id,f'audio',rate)
            addata=gr.update(value=audiofile,visible=True)
            adst=gr.update(value=subtitlefile,visible=True)
            return addata,adst
        else:
            return None,None

    def generate_tts(pb_tts_id,ps_tts_id,language,rate,*audiotext):
        addata=[gr.update(value=None,visible=False)]*Max_fenjin_num
        adst=[gr.update(value=None,visible=False)]*Max_fenjin_num
        print(pb_tts_id,ps_tts_id,audiotext)
        for i,at in enumerate(audiotext):
            if at is not None and len(at)>1:
                if language!='en':
                    tmpat=translate_en_to_ch(at)
                    if len(tmpat)>1:at=tmpat
                audiofile,subtitlefile=generate_text_audio(at,pb_tts_id,f'audio_{i}',rate)
                addata[i]=gr.update(value=audiofile,visible=True)
                adst[i]=gr.update(value=subtitlefile,visible=True)
                yield addata+adst

    def generate_final_video(start_img,end_img,BGM,volumn,starttext,storyname,endtext,*videoclip):
        outputvideopath='data/video/'
        valideclip=[]
        for vdclip in videoclip:
            if vdclip is not None:valideclip.append(vdclip)
        if len(valideclip)==0:return None
        comb_video.generate_final_video(valideclip,start_img,end_img,outputvideopath,'finalvideo.mp4',bgm=BGM,bgmvolume=volumn,starttext=starttext,storyname=storyname,endtext=endtext)
        return outputvideopath+'finalvideo.mp4'
    
    def get_select_to_candi_image(img_ref,evt: gr.SelectData):
        select_imagelabel=evt.value['caption']
        select_imagepath=evt.value['image']['path']
        new_gallery=[]
        if img_ref is not None:
            for data in img_ref:
                new_gallery.append(data)
        new_gallery.append((select_imagepath,select_imagelabel))
        return gr.update(value=new_gallery,visible=True)
    
    def get_select_to_candi_image2(evt: gr.SelectData):
        select_imagelabel=evt.value['caption']
        select_imagepath=evt.value['image']['path']
        image=Image.open(select_imagepath)
        return image

    
    def get_select_to_remove(img_gallery,evt: gr.SelectData):
        select_imagelabel=evt.value['caption']
        select_imagepath=evt.value['image']['path']
        new_gallery=[]
        for data in img_gallery:
            imgpath,label=data
            if imgpath!=select_imagepath:
                new_gallery.append(data)
        return new_gallery

    def llm_story(prompt,modeltype):
        story=_generate_response(prompt,modeltype)
        return story

    def llm_person_scene(prompt,story,modeltype):
        newprompt=prompt.replace('{story}',story)
        person_scene=_generate_response(newprompt,modeltype)
        return person_scene

    def person_scene_text_parse(person_scene):
        persons=[]
        scenes=[]
        for line in person_scene.split('\n'):
            if '角色' in line  and '外形' in line:
                name,desc=line.split(';')
                name=name.split(':')[-1]
                desc=desc.split(':')[-1]
                persons.append(name+':'+desc)
            if '场景' in line  and '画面' in line:
                name,desc=line.split(';')
                name=name.split(':')[-1]
                desc=desc.split(':')[-1]
                scenes.append(name+':'+desc)
        return gr.update(choices=persons,value=persons[0]),gr.update(choices=scenes,value=scenes[0])

    def llm_fenjing(prompt,story,person_scene,modeltype):
        newprompt=prompt.replace('{person_scene}',person_scene).replace('{story}',story)
        fenjin_response=_generate_response(newprompt,modeltype)
        return fenjin_response

    def fenjin_parse(content_text):
        fenjin_person=[]
        fenjing_scene=[]
        fenjing_prompt=[]
        fenjing_pangbai=[]

        fenjin_contents={}
        for line in content_text.split('\n'):
            if '->' in line:
                startcontent,infocontent=line.split('->')
                if '分镜' in startcontent and '画面' in startcontent:
                    i=int(startcontent.split('分镜')[-1].split('画面')[0])
                    if i in fenjin_contents:
                        fenjin_contents[i].append(infocontent)
                    else:
                        fenjin_contents[i]=[infocontent]

        num=len(fenjin_contents)
        for i in range(Max_fenjin_num):
            if i <num:
                contents=fenjin_contents[i+1]
                pangbai_all,person_all,scene_all,prompt_all='',[],[],[]
                for cnt in contents:
                    pangbai,person,scene,prompt=cnt.split(';')
                    pangbai_all+=pangbai.split(':')[-1]+'.'
                    person_all.append(person.split(':')[-1])
                    scene_all.append(scene.split(':')[-1])
                    prompt_all.append(prompt.split(':')[-1])
                fenjing_pangbai.append(gr.update(value=pangbai_all,visible=True))
                fenjin_person.append(gr.update(choices=person_all,value=person_all[0],visible=True))
                fenjing_scene.append(gr.update(choices=scene_all,value=scene_all[0],visible=True))
                fenjing_prompt.append(gr.update(choices=prompt_all,value=prompt_all[0],visible=True))
            else:
                fenjing_pangbai.append(gr.update(value=None,visible=False))
                fenjin_person.append(gr.update(choices=[],value=None,visible=False))
                fenjing_scene.append(gr.update(choices=[],value=None,visible=False))
                fenjing_prompt.append(gr.update(choices=[],value=None,visible=False))
        return fenjing_pangbai+fenjin_person+fenjing_scene+fenjing_prompt
    
    def get_filemd5(filepath):
        with open(filepath, 'rb') as fp:
            data = fp.read()
        file_md5= hashlib.md5(data).hexdigest()
        return file_md5

    def savelib(img_select,dirpath):
        if img_select is not None:
            for data in img_select:
                imgpath,caption = data
                caption=caption.split(':')[0]

                savedir=f'{dirpath}{caption}/'
                if not os.path.exists(savedir):os.makedirs(savedir)
                files_md5=[]
                for filepath in os.listdir(savedir):
                    tmpmd5=get_filemd5(savedir+filepath)
                    if tmpmd5 in files_md5:
                        os.remove(savedir+filepath)
                    else:
                        files_md5.append(tmpmd5)

                imgmd5=get_filemd5(imgpath)
                if imgmd5 not in files_md5:
                    savename=f'{savedir}{time.time()}.png'
                    shutil.copy(imgpath,savename)
        alllibname=os.listdir(dirpath)+['all']
        return gr.update(choices=alllibname)

    def saveperson_lib(img_select,dirpath='set_lib/character/'):
        return savelib(img_select,dirpath)

    def savescene_lib(img_select,dirpath='set_lib/scene/'):
        return savelib(img_select,dirpath)

    def saveaction_lib(img_select,dirpath='set_lib/action/'):
        return savelib(img_select,dirpath)

    def selectlib(libnames,dirpath):
        img_caption=[]
        if 'all' in libnames:
            for libname in os.listdir(dirpath):
                imgs=os.listdir(dirpath+libname)
                for img in imgs:
                    img_caption.append((dirpath+libname+'/'+img,libname))
        else:
            for libname in libnames:
                if not os.path.exists(dirpath+libname):continue
                imgs=os.listdir(dirpath+libname)
                for img in imgs:
                    img_caption.append((dirpath+libname+'/'+img,libname))
        return img_caption

    def selectperson_lib(libnames,dirpath='set_lib/character/'):
        return selectlib(libnames,dirpath)

    def selectscene_lib(libnames,dirpath='set_lib/scene/'):
        return selectlib(libnames,dirpath)
    
    def selectaction_lib(libnames,dirpath='set_lib/action/'):
            return selectlib(libnames,dirpath)


    with gr.Blocks(title='安全管理部-视觉生成技术') as demo:   
        with gr.Tab('故事动画生成'):
            gr.Markdown('1、故事和角色内容生成')
            with gr.Tab('故事创作'):
                with gr.Row():
                    story_prompt=gr.Textbox(value=Default_story_prompt,label='故事prompt',scale=3,max_lines=3)
                    generate_story_text=gr.Button('生成故事设定与描述',variant="primary")
                story_text=gr.Textbox(label='总体故事text',max_lines=8)
            with gr.Tab('角色场景创作/分镜创作'):
                with gr. Row():
                    with gr.Column():
                        with gr.Row():
                            person_scene_prompt=gr.Textbox(value=Default_person_scene_prompt,label='角色场景prompt',scale=3,max_lines=3)
                            generate_ps_text=gr.Button('生成角色场景text',variant="primary")
                        person_scene_text=gr.Textbox(value=DeFault_person_scene_text,label='角色场景text',max_lines=8) 
                    with gr.Column():
                        with gr.Row():
                            content_prompt=gr.Textbox(value=DeFault_fenjin_prompt,label='分镜prompt',scale=3,max_lines=3)
                            generate_content_text=gr.Button('生成分镜描述',variant="primary",scale=1)
                        content_text=gr.Textbox(value=DeFault_fenjin_text,label='分镜text',max_lines=8) 

            
            with gr.Row():
                with gr.Column(scale=1):
                    llm_modeltype=gr.Dropdown(choices=['moonshot','gpt3.5-turbo','glm'],value='moonshot',label='语言模型')
                with gr.Column(scale=9):
                    with gr.Accordion('图像生成参数设置', open=False):
                        with gr.Row(): 
                            modeltype=gr.Dropdown(choices=['sdxl_lightning'],value='sdxl_lightning',label='文生图模型')
                            seed = gr.Slider(0, 10000000000000, 0, label='seed', step=1)
                            guidance = gr.Slider(0, 10, 1, label='guidance', step=0.5)
                            width = gr.Slider(480, 1920, 1024, label='width', step=24)
                            height = gr.Slider(480, 1920, 1024, label='width', step=24)
                            num_inference_steps=gr.Slider(1, 8, 4, label='infer steps', step=1)
                            num_img_per_prompt=gr.Slider(1, 4, 3, label='imgnum/prompt', step=1)
            
            generate_story_text.click(llm_story,inputs=[story_prompt,llm_modeltype],outputs=[story_text])   
            generate_ps_text.click(llm_person_scene,inputs=[person_scene_prompt,story_text,llm_modeltype],outputs=[person_scene_text])   
            generate_content_text.click(llm_fenjing,inputs=[content_prompt,story_text,person_scene_text,llm_modeltype],outputs=[content_text])  
                    
            with gr.Tab('2、角色设定图/场景设定图生成'):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            stylename=gr.Dropdown(choices=sdxl_style_template.get_style_name(),value='No Style',label='prompt风格类型',scale=1)
                            template_type=gr.Dropdown(choices=['cat','man','woman','natural_scene'],value='cat',label='模版主体',scale=1)
                        person_name = gr.Dropdown(choices=[],label='角色名称',allow_custom_value=True)
                        scene_name = gr.Dropdown(choices=[],label='场景名称',allow_custom_value=True)

                    with gr.Column(scale=3):
                        style_example_img = gr.Gallery(sdxl_style_template.get_exampleimg_path(),label='风格模板图',show_download_button=True,columns=8,height=360)
                template_type.select(change_template_type,inputs=[template_type],outputs=[stylename,style_example_img])
                person_scene_text.change(person_scene_text_parse,inputs=[person_scene_text],outputs=[person_name,scene_name])

                with gr.Tab('角色设定图'):
                    with gr.Row():
                        with gr.Column(scale=3):
                            person_prompt_neg=gr.Textbox(label='角色neg prompt(模板词)')
                            person_prompt_pos=gr.Textbox(label='角色pos prompt(模板词)')
                            with gr.Row():
                                generate_person = gr.Button('模板生成角色图',variant="primary")
                            gr.Markdown('点击生成图片可自动移动到参考区')
                        with gr.Column(scale=2):
                            person_img=gr.Gallery(label='角色图',columns=1,show_download_button=True,height=300)
                            register_person = gr.Button('注册角色图',variant="primary")
                        with gr.Column(scale=3):
                            person_simple_neg=gr.Textbox(value=DEFAULT_NEG_PROMPT,label='角色neg prompt(参考图)')
                            person_simple_pos=gr.Textbox(label='角色pos prompt(参考图)')
                            person_scale_simple=gr.Slider(0, 1, 0.5, label='参考权重', step=0.1)
                            generate_person_simple = gr.Button('参考生成角色图',variant="primary")
                        with gr.Column(scale=2):
                            gr.Markdown('点击图片自动删除')
                            person_ref=gr.Gallery(label='角色 ref',columns=1,show_download_button=True,height=300)
                        person_img.select(get_select_to_candi_image,inputs=[person_ref],outputs=[person_ref])
                        person_ref.select(get_select_to_remove,inputs=[person_ref],outputs=[person_ref])

                with gr.Tab('场景设定图'):
                    with gr.Row():
                        with gr.Column(scale=3):
                            scene_prompt_neg=gr.Textbox(label='场景neg prompt(模板词)')
                            scene_prompt_pos=gr.Textbox(label='场景pos prompt(模板词)')
                            with gr.Row():
                                generate_scene = gr.Button('模版生成场景图',variant="primary")
                            gr.Markdown('点击生成图片可自动移动到参考区')
                        with gr.Column(scale=2):
                            scene_img=gr.Gallery(label='场景图',columns=1,show_download_button=True,height=300) 
                            register_scene = gr.Button('注册场景图',variant="primary")
                        with gr.Column(scale=3):
                            scene_simple_neg=gr.Textbox(value=DEFAULT_NEG_PROMPT,label='场景neg prompt(参考图)')
                            scene_simple_pos=gr.Textbox(label='场景pos prompt(参考图)')
                            scene_scale_simple=gr.Slider(0, 1, 0.5, label='参考权重', step=0.1)
                            generate_scene_simple = gr.Button('参考生成场景图',variant="primary")
                        with gr.Column(scale=2):
                            gr.Markdown('点击图片自动删除')
                            style_ref=gr.Gallery(label='场景 ref',columns=1,show_download_button=True,height=300) 
                            #style_ref=gr.Image(type='pil',label='场景 ref',show_download_button=True,height=300)
                        scene_img.select(get_select_to_candi_image,inputs=[style_ref],outputs=[style_ref])
                        style_ref.select(get_select_to_remove,inputs=[style_ref],outputs=[style_ref])
                
                with gr.Tab('动作设定图'):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown('点击图片可自动移动到选定区')
                            action_img=gr.Gallery(label='动作图',columns=1,show_download_button=True,height=300) 
                            select_action_img=gr.Image(type='pil',label='动作选定图',show_download_button=True,height=300)
                        with gr.Column(scale=2):
                            action_prompt_neg=gr.Textbox(value=DEFAULT_NEG_PROMPT,label='动作neg prompt(模板词)')
                            action_prompt_pos=gr.Textbox(label='动作pos prompt(模板词)')
                            generate_action = gr.Button('生成动作图',variant="primary")
                            generate_action_simple = gr.Button('prompt生成前景和mask',variant="primary")
                            with gr.Row():
                                register_action_name = gr.Textbox(label='动作注册名')
                                register_action = gr.Button('注册动作图',variant="primary")
                        with gr.Column(scale=2):
                            gr.Markdown('点击图片自动删除')
                            action_fg_mask=gr.Gallery(label='动作前景与mask',columns=1,show_download_button=True,height=600) 
                        action_img.select(get_select_to_candi_image2,inputs=None,outputs=[select_action_img])
                        action_fg_mask.select(get_select_to_remove,inputs=[action_fg_mask],outputs=[action_fg_mask])
                        #generate_action_simple.click(,inputs=[select_action_img,action_fg_mask],outputs=[action_fg_mask])

                stylename.select(apply_select_stylename,inputs=[stylename,person_name,scene_name],outputs=[person_prompt_neg,person_prompt_pos,scene_prompt_neg,scene_prompt_pos,person_simple_pos,scene_simple_pos])
                style_example_img.select(get_select_template_prompt,inputs=[person_name,scene_name],outputs=[person_prompt_neg,person_prompt_pos,scene_prompt_neg,scene_prompt_pos,stylename,person_simple_pos,scene_simple_pos])
                person_name.change(get_select_name_template_prompt,inputs=[person_name,stylename],outputs=[person_prompt_neg,person_prompt_pos,person_simple_pos])
                scene_name.change(get_select_name_template_prompt,inputs=[scene_name,stylename],outputs=[scene_prompt_neg,scene_prompt_pos,scene_simple_pos])

                input_person=[person_prompt_neg,person_prompt_pos,seed,num_inference_steps,guidance,width,height,num_img_per_prompt]
                generate_person.click(generate_no_ref,inputs=input_person,outputs=[person_img])
                input_person_simple=[person_simple_neg,person_simple_pos,person_ref,seed,num_inference_steps,guidance,width,height,num_img_per_prompt,person_scale_simple]
                generate_person_simple.click(generate_adapter_person,inputs=input_person_simple,outputs=[person_img])
                
                input_scene=[scene_prompt_neg,scene_prompt_pos,seed,num_inference_steps,guidance,width,height,num_img_per_prompt]
                generate_scene.click(generate_no_ref,inputs=input_scene,outputs=[scene_img])
                input_scene_simple=[scene_simple_neg,scene_simple_pos,style_ref,seed,num_inference_steps,guidance,width,height,num_img_per_prompt,scene_scale_simple]
                generate_scene_simple.click(generate_adapter_scene,inputs=input_scene_simple,outputs=[scene_img])
                
            with gr.Tab('3、分镜音视频生成'):
                with gr.Row():
                    language=gr.Radio(choices=['en','zh'],value='en',label='旁白字幕语言',visible=False)
                    with gr.Column():
                        with gr.Row():
                            pangbai_tts_id = gr.Dropdown(choices=CH_LANGUAGE_ID+EN_LANGUAGE_ID,value=CH_LANGUAGE_ID[0],label='旁白声personid')
                            person_tts_id = gr.Dropdown(choices=CH_LANGUAGE_ID+EN_LANGUAGE_ID,value=CH_LANGUAGE_ID[2],label='角色声音id')
                    with gr.Column():
                        generate_fenjin_tts = gr.Button('生成所有分镜旁白',variant="primary")
                    with gr.Column():
                        rate= gr.Slider(-100, 100, 0, label='语速', step=1)
                        fadeinfadeout = gr.Slider(0, 5, 1, label='渐入渐出时长', step=0.5)
                    with gr.Column():
                        person_scale=gr.Slider(0, 1, 1, label='人物mask权重', step=0.1)
                        bg_scale=gr.Slider(0, 1, 1, label='背景mask权重', step=0.1)
                    with gr.Column():
                        generate_fenjin = gr.Button('生成所有分镜图',variant="secondary",visible=False)
                        generate_fenjin_video = gr.Button('生成所有分镜视频',variant="secondary",visible=False)
                    
                fenjin_imgs=[None]*Max_fenjin_num
                fenjin_imgs_select=[None]*Max_fenjin_num
                IP_type=[None]*Max_fenjin_num
                #IP_position=[None]*Max_fenjin_num
                scene_type=[None]*Max_fenjin_num
                fenjin_img_prompt=[None]*Max_fenjin_num
                audiotext=[None]*Max_fenjin_num
                audiodata=[None]*Max_fenjin_num
                audiosubtitle=[None]*Max_fenjin_num
                videoclip=[None]*Max_fenjin_num
                fenjin_audio_generate=[None]*Max_fenjin_num
                fenjin_img_generate=[None]*Max_fenjin_num
                fenjin_video_generate=[None]*Max_fenjin_num
                for i in range(Max_fenjin_num):
                    visual=True if i<1 else False
                    with gr.Tab(f'分镜{i}'):
                        with gr.Row():
                            audiotext[i] = gr.Textbox(label='分镜剧情旁白+角色说话',visible=visual,scale=2)
                            IP_type[i] =gr.Dropdown(label='角色',visible=visual,allow_custom_value=True,scale=1)
                            #IP_position[i] =gr.Textbox(label='位置大小',visible=visual,scale=1)
                            scene_type[i] =gr.Dropdown(label='场景',visible=visual,allow_custom_value=True,scale=0)
                            fenjin_img_prompt[i] =gr.Dropdown(label='分镜图prompt',visible=visual,allow_custom_value=True,scale=2)
                            with gr.Column():
                                fenjin_audio_generate[i] = gr.Button('优化分镜音频',variant="primary")
                                fenjin_img_generate[i] = gr.Button('优化分镜图',variant="primary")
                                fenjin_video_generate[i] = gr.Button('优化分镜视频',variant="primary")
                        with gr.Tab('分镜音视频'):
                            with gr.Row():
                                audiodata[i] = gr.Audio(type='filepath',label=f'分镜音频{i}',visible=visual,show_download_button=True)
                                audiosubtitle[i]=gr.File(label=f'分镜音频字幕{i}',visible=visual)
                                videoclip[i] = gr.Video(label=f'分镜视频{i}',show_download_button=True,visible=visual)
                        with gr.Tab('分镜图片'):
                            with gr.Row():
                                fenjin_imgs[i] = gr.Gallery(columns=2,height=300,label=f'分镜图{i}',show_download_button=True,visible=visual,scale=2)
                                fenjin_imgs_select[i] = gr.Gallery(columns=2,height=300,label=f'选定分镜图{i}',show_download_button=True,visible=visual,scale=2)

                                
                with gr.Row():
                    mask_img_lib=gr.Gallery(value=Default_mask_img,label='mask池',columns=5,show_download_button=True,height=190,scale=5) 
                    mask_img_select=gr.Gallery(label='mask选择池',columns=3,show_download_button=True,height=190,scale=3) 
                    mask_img_lib.select(get_select_to_candi_image,inputs=[mask_img_select],outputs=[mask_img_select])
                    mask_img_select.select(get_select_to_remove,inputs=[mask_img_select],outputs=[mask_img_select])

            gr.Markdown('注意：点击设定池中图片自动删除(假操作)')
            with gr.Row():
                with gr.Column():
                    person_img_select=gr.Gallery(label='角色图注册池',columns=3,show_download_button=True,height=320) 
                    with gr.Row():
                        select_person_lib = gr.Dropdown(choices=['all'],label='选择角色加载',multiselect=True)
                        save_person_lib=gr.Button('更新/存储角色库',variant="primary")
                with gr.Column():
                    scene_img_select=gr.Gallery(label='场景图注册池',columns=3,show_download_button=True,height=320) 
                    with gr.Row():
                        select_scene_lib = gr.Dropdown(choices=['all'],label='选择场景加载',multiselect=True)
                        save_scene_lib=gr.Button('更新/存储场景库',variant="primary")
                with gr.Column():
                    action_img_select=gr.Gallery(label='动作图注册池',columns=3,show_download_button=True,height=320) 
                    with gr.Row():
                        select_action_lib = gr.Dropdown(choices=['all'],label='选择动作加载',multiselect=True)
                        save_action_lib=gr.Button('更新/存储动作库',variant="primary")

                person_img_select.select(get_select_to_remove,inputs=[person_img_select],outputs=[person_img_select])
                scene_img_select.select(get_select_to_remove,inputs=[scene_img_select],outputs=[scene_img_select])
                action_img_select.select(get_select_to_remove,inputs=[action_img_select],outputs=[action_img_select])
                
                save_person_lib.click(saveperson_lib,inputs=[person_img_select],outputs=[select_person_lib])
                save_scene_lib.click(savescene_lib,inputs=[scene_img_select],outputs=[select_scene_lib])
                save_action_lib.click(saveaction_lib,inputs=[action_img_select],outputs=[select_action_lib])

                select_person_lib.change(selectperson_lib,inputs=[select_person_lib],outputs=[person_img_select])
                select_scene_lib.change(selectscene_lib,inputs=[select_scene_lib],outputs=[scene_img_select])
                select_action_lib.change(selectaction_lib,inputs=[select_action_lib],outputs=[action_img_select])

            
            content_text.change(fenjin_parse,inputs=[content_text],outputs=audiotext+IP_type+scene_type+fenjin_img_prompt)

            register_person.click(register_img_to_candi,inputs=[person_img,person_name,person_img_select],outputs=[person_img_select])
            register_scene.click(register_img_to_candi,inputs=[scene_img,scene_name,scene_img_select],outputs=[scene_img_select])
            register_action.click(register_img_to_candi,inputs=[action_fg_mask,register_action_name,action_img_select],outputs=[action_img_select])
                

            generate_fenjin_tts.click(generate_tts,inputs=[pangbai_tts_id,person_tts_id,language,rate]+audiotext,outputs=audiodata+audiosubtitle)
            
            for i in range(Max_fenjin_num):
                fenjin_audio_generate[i].click(generate_single_tts,inputs=[pangbai_tts_id,person_tts_id,language,rate,audiotext[i]],outputs=[audiodata[i],audiosubtitle[i]])
                fenjin_img_generate[i].click(generate_single_fenjin_img,inputs=[person_img_select,scene_img_select,seed,num_inference_steps,guidance,width,height,num_img_per_prompt,
                            mask_img_select,person_scale,bg_scale,
                            IP_type[i],scene_type[i],fenjin_img_prompt[i]],outputs=[fenjin_imgs[i]])
                fenjin_imgs[i].select(get_select_to_candi_image,inputs=[fenjin_imgs_select[i]],outputs=[fenjin_imgs_select[i]])
                fenjin_imgs_select[i].select(get_select_to_remove,inputs=[fenjin_imgs_select[i]],outputs=[fenjin_imgs_select[i]])
                fenjin_video_generate[i].click(generate_single_fenjin_video,inputs=[fadeinfadeout,audiodata[i],audiosubtitle[i],fenjin_imgs_select[i]],outputs=videoclip[i])
            

            generate_fenjin.click(generate_all_fenjin_imgs,
                                      inputs=[person_img_select,scene_img_select,seed,num_inference_steps,guidance,width,height,num_img_per_prompt]+IP_type+scene_type+fenjin_img_prompt,
                                      outputs=fenjin_imgs)
            generate_fenjin_video.click(generate_all_fenjin_video,inputs=[fadeinfadeout]+audiodata+audiosubtitle+fenjin_imgs_select,outputs=videoclip)
                
            gr.Markdown('4、完整视频生成')
            with gr.Row():
                with gr.Column(scale=1):
                    start_img = gr.Image('data/image/dragon_baby1.png',type='filepath',label='片头',show_download_button=True)
                    end_img = gr.Image('data/image/goodnight1.png',type='filepath',label='片尾',show_download_button=True)
                with gr.Column(scale=2):
                    BGM = gr.Audio(value='/mnt/glennge/MoneyPrinter/source/Songs/'+DeFault_BGM_path[0],type='filepath',label='背景音乐',show_download_button=True)
                    with gr.Row():
                        bgmlist=gr.Dropdown(choices=DeFault_BGM_path,value=DeFault_BGM_path[0],label='可选bgm')
                        volumn = gr.Slider(0, 2, 1, label='背景音量', step=0.1)
                    with gr.Row():
                        starttext=gr.Textbox(value='龙宝的睡前故事',label='片头字')
                        storyname=gr.Textbox(value='',label='故事名')
                        endtext=gr.Textbox(value='～晚安好梦～',label='片尾字')
                with gr.Column(scale=2):
                    generate_allvideo = gr.Button('生成完整视频',variant="primary")
                    allvideo = gr.Video(label='完整视频',show_download_button=True)  
            bgmlist.select(get_bgm_song,inputs=bgmlist,outputs=[BGM])
            generate_allvideo.click(generate_final_video,inputs=[start_img,end_img,BGM,volumn,starttext,storyname,endtext]+videoclip,outputs=[allvideo])                         
    return demo       
    
        
    
if __name__ =="__main__":
    try:
        #t2imodel=sdxl_lightning_model()
        t2imodel=ipadapter_model_multi_adapter(lightning=True)
        #t2imodel=sd15_model()
    except Exception as e:
        print('t2i error',e)
        t2imodel=sd15_model()
    

    generate_image_gr_demo(t2imodel).queue().launch(max_threads=15,
                                                            show_api=True,
                                                            share=False, 
                                                            server_name='0.0.0.0',server_port=8082)
