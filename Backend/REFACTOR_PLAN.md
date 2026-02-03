# 五大生成环节拆分优化方案

## 一、目标

1. **story_generate.py 瘦身**：入口文件只负责组装五 Tab、全局 State 与启动，业务逻辑与 UI 下沉到各环节模块。
2. **环节可单独运行与调试**：每个 Tab 对应一个模块，支持 `python -m Backend.story_ui.tabN_xxx` 或 `python Backend/story_ui/tabN_xxx.py` 仅启动该 Tab 的 Gradio 页面，便于单环节开发与排查。

## 二、目录与模块划分

```
Backend/
├── story_generate.py          # 精简为：建 Blocks、建 State、依次挂载五 Tab、启动
└── story_ui/
    ├── __init__.py            # 导出 build_app、各 tab.build、run_standalone
    ├── shared.py              # 常量、默认 prompt、跨 Tab 复用工具函数
    ├── tab1_story.py          # 1、故事与剧本
    ├── tab2_characters.py     # 2、角色与场景
    ├── tab3_refs_shots.py     # 3、角色/场景图与分镜画面
    ├── tab4_voice_video.py    # 4、配音与分镜视频
    └── tab5_export.py         # 5、成片导出
```

## 三、各模块职责与接口

### 3.1 shared.py

- **常量**：`MAX_SEED`、`DEFAULT_NEG_PROMPT`、`Max_fenjin_num`、`DeFault_BGM_path`（及 BGM 根路径可配置化占位）。
- **默认文案**：`Default_story_prompt`、`Default_person_scene_prompt`、`DeFault_fenjin_prompt`、`DeFault_person_scene_text`、`DeFault_fenjin_text`。
- **通用 UI 小工具**（与 Gradio 事件兼容）：
  - `get_select_to_candi_image(img_ref, evt)`、`get_select_to_remove(img_gallery, evt)`、`get_select_to_candi_image2(evt)`；
  - `get_bgm_song(bgmlist)`（若依赖绝对路径，可接配置）；
  - 与 mask 相关的 `generate_fgmaskimg`、`get_default_maskimg()`、`Default_mask_img` / `Default_ip_position_infos`（Tab4 使用则从 shared 取）。
- **库读写**：`get_filemd5`、`savelib`、`saveperson_lib`、`savescene_lib`、`saveaction_lib`、`selectlib`、`selectperson_lib`、`selectscene_lib`、`selectaction_lib`，供 Tab3/Tab4 使用。

### 3.2 tab1_story.py — 1、故事与剧本

- **逻辑**：`_segments_from_story`、`llm_story`、`llm_story_and_sync`、`on_parse_novel`、`on_summarize_all`、`on_label_segment_characters_scenes`（依赖 state_segments / state_global_characters|scenes）；依赖 `llm`、`tool.novel_parser`、`shared`。
- **接口**：
  - `build(demo, ctx)`：在 `demo` 下挂 `gr.Tab('1、故事与剧本')`，用 `ctx` 中的 state 与共享控件，绑定事件，返回本 Tab 创建的、需给其它 Tab 用的组件的名字或引用（若有）。
  - `run_standalone()`：创建仅含 Tab1 的 `gr.Blocks()`，用占位 State，执行「本页测试」逻辑，`launch(server_name='0.0.0.0', server_port=8082)`（端口可参数化）。
- **本页测试**：`test_tab1_story(llm_modeltype)` 留在本模块，由 `build` 与 `run_standalone` 共用。

### 3.3 tab2_characters.py — 2、角色与场景

- **逻辑**：`_parse_person_scene_detailed`、`llm_person_scene`、`llm_person_scene_detailed`、`on_label_segment_characters_scenes`（若与 tab1 重复可抽到 shared 或仅在一处实现并由 tab1 调用）、`llm_fenjing`、`_parse_cinematic_shot_line`、`on_generate_cinematic_storyboards`、`person_scene_text_parse`、`fenjin_parse`；依赖 `llm`、`shared`。
- **接口**：`build(demo, ctx)`、`run_standalone()`；`fenjin_parse` 若被 Tab4 使用，由本模块或 shared 提供，Tab4 从 shared 或 tab2 导入。
- **本页测试**：`test_tab2_characters_scenes(llm_modeltype)` 在本模块。

### 3.4 tab3_refs_shots.py — 3、角色/场景图与分镜画面

- **逻辑**：风格/模板相关（`change_template_type`、`apply_select_stylename`、`get_select_template_prompt`、`get_select_name_template_prompt`、`register_img_template`、`register_img_to_candi`）；T2I 生成（`generate_no_ref`、`generate_adapter_person`、`generate_adapter_scene`、`get_ref_imgs`、`generate_adapter_person_scene`、mask 系 `generate_bgmaskimg`、`get_person_position_mask`、`generate_adapter_person_scene_with_mask_pro`）；9 宫格与分镜选图（`_fenjin_img_from_9grid`、`generate_single_fenjin_img`、`generate_all_fenjin_imgs`）；批量首图与 9 宫格（`batch_generate_character_scene_refs`、`batch_generate_9grid_for_characters_scenes`）。依赖 `mora.generate_image`、`mora.consistency_9grid`、`mora.sd_prompt.sdxl_styles`、`shared`，以及 ctx 中的 t2imodel / t2i_backend / t2i_model_id 等。
- **接口**：`build(demo, ctx)`；`run_standalone(t2imodel=None)`，可为 None 时用 `get_t2i_model(...)` 默认加载。
- **本页测试**：`test_tab3_refs_and_shots(t2i_backend, t2i_model_id)` 在本模块。

### 3.5 tab4_voice_video.py — 4、配音与分镜视频

- **逻辑**：`generate_single_tts`、`generate_tts`、`generate_single_fenjin_video`、`generate_all_fenjin_video`；分镜图从 9 宫格/库选取的逻辑由 tab3 提供的函数或 shared 中的工具在组装时串起来；库的加载/保存使用 shared 的 select*/save*；`content_text.change(fenjin_parse, ...)` 使用 tab2 的 `fenjin_parse`。依赖 `tts_utils`、`video_gen`、`video_utils`、`shared`，以及 `fenjin_parse`（来自 tab2 或 shared）。
- **接口**：`build(demo, ctx)`，ctx 中需包含 content_text、person_img_select、scene_img_select、state_*、共享参数（seed、guidance、video_backend 等）；`run_standalone()` 用占位 content_text 与占位图库，只跑 TTS + 一条 slideshow 分镜视频。
- **本页测试**：`test_tab4_tts_and_shot_video(...)` 在本模块。

### 3.6 tab5_export.py — 5、成片导出

- **逻辑**：`generate_final_video`、`get_bgm_song`（或使用 shared）。依赖 `video_utils.comb_video`、`shared`。
- **接口**：`build(demo, ctx)`，ctx 提供 videoclip 列表、BGM 等；`run_standalone()` 用本地已有分镜视频文件（如 test_tab4.mp4）合成一条短线成片。
- **本页测试**：`test_tab5_final_export()` 在本模块。

## 四、Context 约定（ctx）

主入口在 `story_generate.py` 中构造 `ctx`，传入各 tab 的 `build(demo, ctx)`。建议 `ctx` 为字典或简单对象，包含：

- **State**：`state_full_story`、`state_segments`、`state_global_characters`、`state_global_scenes`、`state_storyboards`、`state_shot_images`、`state_shot_audios`、`state_shot_videos`。
- **跨 Tab 控件**：`story_text`、`llm_modeltype`、`person_scene_text`、`content_text`、`t2i_backend`、`t2i_model_id`、`seed`、`guidance`、`width`、`height`、`num_inference_steps`、`num_img_per_prompt`；以及 Tab4 的 `person_img_select`、`scene_img_select`、`videoclip` 等（在组装时按 tab 依赖顺序创建并写入 ctx）。
- **运行时可注入**：`t2imodel`（可选，若为 None 则 tab3 在回调里用 `get_t2i_model(ctx['t2i_backend'], ctx['t2i_model_id'])`）。

各 tab 的 `build` 只从 `ctx` 读已有组件、向 `ctx` 写入本 Tab 新建且需给后续 Tab 使用的组件，避免 tab 之间直接 import 对方。

## 五、story_generate.py 精简后结构

```python
# 1) 导入
from story_ui import build_app

# 2) 入口函数
def generate_image_gr_demo(t2imodel):
    return build_app(t2imodel)

# 3) __main__
if __name__ == "__main__":
    t2imodel = ...
    generate_image_gr_demo(t2imodel).queue().launch(...)
```

`build_app(t2imodel)` 内部：创建 `gr.Blocks()`；创建全部 State 与共享控件；`ctx = {...}`；依次调用 `tab1.build(demo, ctx)`、…、`tab5.build(demo, ctx)`；返回 `demo`。

## 六、单 Tab 独立运行方式

在 Backend 目录下执行（需已安装 gradio、torch 等依赖）：

- Tab1：`python -m story_ui.tab1_story` → 默认 `http://0.0.0.0:8082`
- Tab2：`python -m story_ui.tab2_characters` → 默认 port 8083
- Tab3：`python -m story_ui.tab3_refs_shots` → 默认 port 8084
- Tab4：`python -m story_ui.tab4_voice_video` → 默认 port 8085
- Tab5：`python -m story_ui.tab5_export` → 默认 port 8086

各模块内 `run_standalone(server_name, server_port)` 可传参以改端口，便于多 Tab 同时联调。

## 七、实施顺序

1. 新建 `story_ui/`、`shared.py`，迁入常量与通用小工具、默认 prompt、库读写。
2. 实现 `tab1_story.py`（build + run_standalone + test），主入口临时只挂 Tab1，验证通过。
3. 实现 `tab2_characters.py`，挂 Tab1+Tab2，验证。
4. 实现 `tab3_refs_shots.py`，挂 Tab1～Tab3，验证。
5. 实现 `tab4_voice_video.py`，挂 Tab1～Tab4，验证。
6. 实现 `tab5_export.py`，挂齐五 Tab，验证。
7. 在 `story_generate.py` 中删除已迁移逻辑，改为调用 `build_app(t2imodel)`，做一次全流程回归。

---

以上方案兼顾「入口简洁」「环节边界清晰」「可单 Tab 运行调试」；实施时以可运行、可回滚为优先，接口（build/run_standalone）可先粗后细，再逐步把共享控件与 ctx 收敛到统一约定。
