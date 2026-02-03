# -*- coding: utf-8 -*-
"""
五大环节共用：常量、默认 prompt、跨 Tab 复用的小工具与库读写。
"""
import os
import sys
import hashlib
import shutil
import time
import random
from PIL import Image
import numpy as np
import gradio as gr

# 保证以 Backend 为工作目录时 tool/llm 等可导入
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

MAX_SEED = np.iinfo(np.int32).max
DEFAULT_NEG_PROMPT = (
    "bad hands,bad face,distort limbs,text, watermark, lowres, low quality, worst quality, "
    "deformed, glitch, low contrast, noisy, saturation, blurry"
)
Max_fenjin_num = 20

try:
    _bgm_root = "/mnt/glennge/MoneyPrinter/source/Songs/"
    DeFault_BGM_path = [f for f in os.listdir(_bgm_root)] if os.path.isdir(_bgm_root) else []
except Exception:
    DeFault_BGM_path = []


# ---------- 默认 Prompt 文案 ----------
Default_story_prompt = """作为一个儿童故事畅销作者，按如下主题创作一篇300～500字，8～10个段落的儿童故事。要求情节清晰、简单易懂、引人入胜、让小朋友十分喜欢。
故事主题: 坐井观天的故事，讲述两个小青蛙，一个坐井观天，一个勇敢探索的故事
"""

Default_person_scene_prompt = """请根据以下故事内容总结故事的主要角色和主要场景，按照以下格式总结：
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

DeFault_fenjin_prompt = """请根据以下主要故事角色、主要故事场景和故事内容，将故事内容的每个段落设计成一个分镜剧本,需包含7～9个分镜，每个分镜都包括多个画面，每个画面都包括四个内容：旁白内容，角色，场景，内容描述，并按照以下格式总结：
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

DeFault_person_scene_text = """[Characters]
角色0名称:小蓝;外形描述:Small and blue frog, with big round eyes and a smooth skin texture, dressed in a simple green outfit that matches the color of the well.
角色1名称:小绿;外形描述:Small and green frog, with bright eyes and a smooth skin texture, dressed in a simple green outfit that matches the color of the well.
[Scenes]
场景0名称:古老井口外;画面描述:An deep well with moss-covered stones.
场景3名称:古老井口内;画面描述:An sight from inside of deep well with moss-covered stones.
场景1名称:外面的世界;画面描述:A vast expanse of lush green meadows, dotted with colorful flowers and a clear, sparkling stream under a bright, sunny sky.
场景2名称:森林通向外面的小路;画面描述:A winding path from forest  to the outside world."""

DeFault_fenjin_text = """分镜1画面1->旁白内容:在一个遥远的森林深处，有一口古老的井，井里住着两只小青蛙。;角色:小蓝,小绿;场景:古老井口内;画面prompt:Frogs in an ancient well.
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
分镜8画面2->旁白内容:只有通过探索，我们才能发现更多的美好和学到更多的知识。同时，我们也要关心身边的朋友，鼓励他们一起成长。;角色:None;场景:外面的世界;画面prompt:Encouraging growth and exploration."""


# ---------- 跨 Tab 复用的 UI 小工具 ----------
def get_select_to_candi_image(img_ref, evt: gr.SelectData):
    select_imagelabel = evt.value["caption"]
    select_imagepath = evt.value["image"]["path"]
    new_gallery = []
    if img_ref is not None:
        for data in img_ref:
            new_gallery.append(data)
    new_gallery.append((select_imagepath, select_imagelabel))
    return gr.update(value=new_gallery, visible=True)


def get_select_to_candi_image2(evt: gr.SelectData):
    select_imagepath = evt.value["image"]["path"]
    return Image.open(select_imagepath)


def get_select_to_remove(img_gallery, evt: gr.SelectData):
    select_imagepath = evt.value["image"]["path"]
    new_gallery = []
    for data in (img_gallery or []):
        imgpath, label = data
        if imgpath != select_imagepath:
            new_gallery.append(data)
    return new_gallery


def register_img_to_candi(images, name, gallery):
    if name is None or len(name) < 1:
        raise gr.Error("注册名称不能为空")
    new_gallery = []
    if gallery is not None:
        for tmpimage, tmpname in gallery:
            new_gallery.append((tmpimage, tmpname))
    if images is not None:
        for tmpimage, tmpname in images:
            new_gallery.append((tmpimage, name))
    return new_gallery


def get_bgm_song(bgmlist):
    return "/mnt/glennge/MoneyPrinter/source/Songs/" + bgmlist


# ---------- Mask 相关（Tab4 使用）----------
def generate_fgmaskimg(position, W, H):
    maskimg = np.array(Image.new("RGB", (W, H), color=(0, 0, 0)))
    p1, p2, pose, size = position.split("-")
    p1, p2, pose, size = int(p1), int(p2), float(pose), int(size)
    basesize1 = (int(W / 3), int(H / 3))
    basesize2 = (int(W / 9), int(H / 9))
    x_index1 = int(p1 / 3)
    y_index1 = p1 % 3
    x_index2 = int(p2 / 3)
    y_index2 = p2 % 3
    center_x = int(x_index1 * basesize1[0] + (0.5 + x_index2) * basesize2[0])
    center_y = int(y_index1 * basesize1[1] + (0.5 + y_index2) * basesize2[1])
    scale = max(min(3, random.uniform(-0.2, 0.2) + size), 0.5)
    w_2 = int(scale * basesize1[0] / pose / 2)
    h_2 = int(scale * basesize1[1] * pose / 2)
    x1, x2 = max(0, center_x - w_2), min(W, center_x + w_2)
    y1, y2 = max(0, center_y - h_2), min(W, center_y + h_2)
    maskimg[y1:y2, x1:x2, :] = 255
    return Image.fromarray(maskimg)


def get_default_maskimg():
    maskimgs = []
    ip_position_infos = []
    for p1 in range(0, 9):
        for p2 in range(0, 9):
            if p1 < 3 and p2 < 5:
                continue
            if p1 > 5 and p2 > 2:
                continue
            if p1 in [0, 3, 6] and p2 in [0, 3, 6]:
                continue
            if p1 in [2, 5, 8] and p2 in [2, 5, 8]:
                continue
            for pose in [0.7, 1, 1.4]:
                for size in [3, 2, 1]:
                    if p1 != 4 and size == 3:
                        continue
                    if p1 in [0, 1, 2, 6, 7, 8] and p2 in [1, 4, 7] and size == 2:
                        continue
                    if p1 in [3, 5] and p2 == 4 and size == 2:
                        continue
                    ip_position_infos.append(f"{p1}-{p2}-{pose}-{size}")
                    maskimg = generate_fgmaskimg(f"{p1}-{p2}-{pose}-{size}", 1024, 1024)
                    maskimgs.append((maskimg, f"{p1}-{p2}-{pose}-{size}"))
    return maskimgs, ip_position_infos


Default_mask_img, Default_ip_position_infos = get_default_maskimg()


# ---------- 库读写（Tab3/Tab4 使用）----------
def get_filemd5(filepath):
    with open(filepath, "rb") as fp:
        data = fp.read()
    return hashlib.md5(data).hexdigest()


def savelib(img_select, dirpath):
    if img_select is not None:
        for data in img_select:
            imgpath, caption = data
            caption = caption.split(":")[0]
            savedir = f"{dirpath}{caption}/"
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            files_md5 = []
            for fname in os.listdir(savedir):
                fp = os.path.join(savedir, fname)
                if not os.path.isfile(fp):
                    continue
                tmpmd5 = get_filemd5(fp)
                if tmpmd5 in files_md5:
                    os.remove(fp)
                else:
                    files_md5.append(tmpmd5)
            imgmd5 = get_filemd5(imgpath)
            if imgmd5 not in files_md5:
                savename = f"{savedir}{time.time()}.png"
                shutil.copy(imgpath, savename)
    alllibname = list(os.listdir(dirpath)) + ["all"]
    return gr.update(choices=alllibname)


def saveperson_lib(img_select, dirpath="set_lib/character/"):
    return savelib(img_select, dirpath)


def savescene_lib(img_select, dirpath="set_lib/scene/"):
    return savelib(img_select, dirpath)


def saveaction_lib(img_select, dirpath="set_lib/action/"):
    return savelib(img_select, dirpath)


def selectlib(libnames, dirpath):
    img_caption = []
    if not libnames or not os.path.isdir(dirpath):
        return img_caption
    if "all" in libnames:
        for libname in os.listdir(dirpath):
            p = os.path.join(dirpath, libname)
            if not os.path.isdir(p):
                continue
            for img in os.listdir(p):
                img_caption.append((os.path.join(p, img), libname))
    else:
        for libname in libnames:
            p = os.path.join(dirpath, libname)
            if not os.path.exists(p):
                continue
            for img in os.listdir(p):
                img_caption.append((os.path.join(p, img), libname))
    return img_caption


def selectperson_lib(libnames, dirpath="set_lib/character/"):
    return selectlib(libnames, dirpath)


def selectscene_lib(libnames, dirpath="set_lib/scene/"):
    return selectlib(libnames, dirpath)


def selectaction_lib(libnames, dirpath="set_lib/action/"):
    return selectlib(libnames, dirpath)
