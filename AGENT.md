# AGENT.md — 项目产品与结构说明

本文档描述本仓库**主要产品功能**、**代码结构**及**各模块职责**，便于 AI Agent 或开发者快速理解与改动的边界。

---

## 1. 主要产品功能

本系统是一个**儿童故事/小说可视化视频生产线**：从文字故事或长篇小说出发，经分镜设计、角色与场景设定、一致性画面生成、配音与分镜视频，最终合成可导出的成片。

### 1.1 端到端流程概览

```
故事/小说输入 → 分段与摘要 → 角色/场景抽取 → 电影化分镜 → 角色/场景首图与9宫格
    → 分镜画面选取 → 旁白 TTS + 字幕 → 分镜视频（幻灯片 / I2V）→ 成片导出
```

- **输入**：主题/梗概（由 LLM 扩写为完整故事），或长篇小说 txt（按章/段解析）。
- **输出**：带片头片尾、BGM、分镜串联的完整视频文件。

### 1.2 五大阶段（与 Gradio Tab 对应）

| 阶段 | Tab | 功能要点 |
|------|-----|----------|
| **阶段1** | Tab1 故事与剧本 | 主题/梗概扩写；长篇小说上传→解析为章/段；一键提炼各段摘要；按段标注涉及的角色/场景。 |
| **阶段2** | Tab2 角色与场景 | 从全文抽取角色、场景（含外形/画面描述、详细描述）；按段生成电影化分镜（旁白、角色、场景、画面 prompt、镜头类型、景别）。 |
| **阶段3** | Tab2→Tab3 | 角色/场景首图生成并写入 `set_lib/character/`、`set_lib/scene/`；可选「为全部角色/场景生成 9 宫格」写入各目录下 `grid_9/`。 |
| **阶段4** | Tab3 角色/场景图与分镜画面 | 分镜选图优先从 9 宫格取（`grid_9/{0~8}.png`，按景别/镜头类型映射）；无 9 宫格时可用旧版 IP-Adapter 路径（`use_legacy_consistency=True`）。 |
| **阶段5** | Tab4 配音与分镜视频 / Tab5 成片导出 | 每镜：旁白 TTS → 字幕 → 分镜视频（slideshow 或 I2V）；成片：分镜视频串联 + 片头/片尾 + BGM + 标题/故事名/结尾字。 |

---

## 2. 代码结构（顶层）

```
cursor_coding/
├── Backend/                    # 后端与主应用目录（运行故事生成时应以 Backend 为工作目录）
│   ├── story_generate.py       # Gradio 主入口、五 Tab 流程与状态
│   ├── llm.py                  # LLM 调用与故事/分镜类 prompt
│   ├── config_loader.py        # 统一配置（.env / config.toml）
│   ├── video_gen.py            # 分镜视频统一入口（slideshow / i2v / t2v）
│   ├── video_utils.py          # 幻灯片合成、I2V 后音字合成、成片拼接
│   ├── tts_utils.py            # TTS 与字幕（Edge-TTS、Qwen3-TTS 等）
│   ├── tool/
│   │   └── novel_parser.py     # 阶段1：长篇小说按章/段解析
│   ├── mora/
│   │   ├── generate_image.py   # 文生图多后端、get_t2i_model
│   │   ├── consistency_9grid.py# 阶段4：单参考图 → 9 宫格一致性图
│   │   └── iterate_generate_video.py  # I2V（如 SVD）实现
│   └── tts/
│       └── qwen3_tts.py        # Qwen3-TTS 等
├── source/                     # 字体等静态资源
└── AGENT.md                    # 本文件
```

---

## 3. 模块功能说明

### 3.1 `Backend/story_generate.py`

- **职责**：Gradio 主应用、五 Tab UI、跨 Tab 状态（`state_full_story`、`state_segments`、`state_global_characters`、`state_global_scenes`、`state_storyboards`、分镜图/音/视）。
- **入口**：`python story_generate.py`（需在 `Backend` 目录下执行，保证 `from tool.`、`from mora.` 等可用）。
- **关键依赖**：`mora.generate_image`（T2I）、`llm`（扩写/分镜/角色场景）、`video_gen`、`video_utils.comb_video`、`tts_utils`、`tool.novel_parser`、`mora.consistency_9grid`（可选）。
- **分镜图来源**：默认从 9 宫格取图（`_fenjin_img_from_9grid`，`set_lib/character|<场景>/grid_9/{0~8}.png`）；可通过 `use_legacy_consistency` 切回 IP-Adapter 路径。

### 3.2 `Backend/llm.py`

- **职责**：所有 LLM 调用与故事/分镜相关 prompt。
- **主要接口**：
  - `_generate_response(prompt, llm_provider, model_name)`：通用生成；商用 provider 的 api_key/base_url/model 由 `config_loader.get_llm_config` 或环境变量提供。
  - `expand_story_from_outline(outline)`：主题/梗概 → 完整儿童故事。
  - `summarize_segment(segment_content)`：单段摘要。
  - `extract_characters_scenes_detailed(full_story)`：全文 → 角色/场景（含外形描述、详细描述）。
  - `get_segment_characters_scenes(segment_content, character_names, scene_names)`：单段涉及的角色、场景列表。
  - `generate_cinematic_storyboard_for_segment(segment_content, character_names, scene_names)`：单段电影化分镜（含镜头类型、景别）。
- **Provider**：支持 g4f、moonshot、glm、openai、coze 等；配置见 `config_loader` 与环境变量。

### 3.3 `Backend/tool/novel_parser.py`

- **职责**：阶段1 长篇小说解析。
- **主要接口**：
  - `parse_novel_file(filepath)`：返回 `[List[{segment_id, chapter_title, content, summary}]]\)`，按常见章节标题或双换行切分，过长章节按 `max_segment_chars` 再拆。
  - `segments_to_full_story(segments)`：把各段 `content` 拼成全文，供全局角色/场景抽取。

### 3.4 `Backend/mora/consistency_9grid.py`

- **职责**：阶段4 角色/场景一致性 9 宫格生成。
- **主要接口**：`generate_9grid_from_reference(ref_image_path, subject_type, subject_desc, options, backend, local_generate_fn)`  
  - `subject_type`: `"character"` | `"scene"`  
  - `backend`: `"nano_banana"` | `"local"`（或环境变量 `USE_CONSISTENCY_BACKEND`）  
  - `local_generate_fn`：仅 `backend=="local"` 时使用，`(prompt, negative_prompt) -> PIL.Image`  
- **输出**：与 `GRID_9_PROMPT_SUFFIXES` 顺序一致的 9 张图（特写、中景、全景、侧脸、背影等）。

### 3.5 `Backend/mora/generate_image.py`

- **职责**：文生图/图生图多后端抽象与模型加载。
- **关键常量**：`T2I_BACKEND_OPEN_SOURCE` / `T2I_BACKEND_COMMERCIAL`，`T2I_OPEN_SOURCE_MODEL_IDS` / `T2I_COMMERCIAL_MODEL_IDS`。
- **统一入口**：`get_t2i_model(backend, model_id, name)`，返回具备 `generate_face_style(face_image, style_images, prompt, negative_prompt, ...)` 的实例。
- **开源 model_id**：flux_dev、flux_schnell、z_image_turbo、sdxl_lightning、sd15、ipadapter。
- **商用 model_id**：nano_banana_pro、tongyi、bytedance、kling 等。

### 3.6 `Backend/video_gen.py`

- **职责**：分镜视频生成的统一入口。
- **主要接口**：`generate_shot_video(images, audio_path, subtitle_path, output_dir, savename, backend, model_id, fadetime)`  
  - `backend`：`slideshow`（图+音+字合成）、`i2v`、`t2v`。  
  - `slideshow`：委托 `video_utils.comb_video.image_to_video_with_audio_subtitle`。  
  - `i2v`/`t2v`：内部调用 `mora.iterate_generate_video`（如 SVD）等，再经 `comb_video.video_audio_subtitle_to_video` 合音字。

### 3.7 `Backend/video_utils.py`

- **职责**：视频合成与成片拼接。
- **主要接口**（通过 `comb_video` 实例）：
  - `image_to_video_with_audio_subtitle(...)`：多图 + 音频 + 字幕 → 幻灯片式分镜视频。
  - `video_audio_subtitle_to_video(...)`：I2V 生成的原始视频 + 音频 + 字幕 → 带字幕分镜视频。
  - `generate_final_video(videoclipfiles, startimg, endimg, ..., bgm, starttext, storyname, endtext)`：分镜片段 + 片头/片尾 + BGM + 标题/结尾字 → 成片。

### 3.8 `Backend/tts_utils.py`

- **职责**：TTS 与字幕生成、中英互译（用于旁白/界面）。
- **主要接口**：
  - `synthesize(text, voice_or_speaker_id, rate, ..., backend, model_id)`：统一 TTS，返回 `(audio_path, subtitle_path)`；`backend` 为 edgetts | qwen3_tts 等。
  - `generate_text_audio(text, voice, datafilename, rate, outputdir, tts_backend, tts_model_id)`：分镜旁白单段配音，内部调用 `synthesize`。
- **语言/音色**：`CH_LANGUAGE_ID`、`EN_LANGUAGE_ID`；支持 Edge-TTS、Qwen3-TTS 等。

### 3.9 `Backend/config_loader.py`

- **职责**：从环境变量和可选 `.env`、`config.toml` 加载配置，禁止在业务代码中硬编码密钥。
- **主要接口**：
  - `get_llm_config(provider)` → `(api_key, base_url, model_name)`，供 `llm.py` 使用。
  - `get_t2i_config(model_id)` → `(api_key, base_url, model_name)`，供商用 T2I 后端使用。

### 3.10 `Backend/mora/iterate_generate_video.py`

- **职责**：图生视频（I2V）等实现，被 `video_gen._run_i2v_t2v` 调用。
- **当前能力**：例如 `svd_image_to_video(first_image, raw_path, ...)`，与 `video_gen` 中 `model_id in ("svd", "wan2.2")` 对应；扩展点可在此接入 HunyuanVideo、CogVideoX、Luma、Kling 等。

---

## 4. 数据与目录约定

- **故事/状态**：在内存中由 Gradio State 在 Tab 间传递，无持久化数据库。
- **角色/场景资产**：
  - 首图与 9 宫格：`set_lib/character/<角色名>/`、`set_lib/scene/<场景名>/`，9 宫格在对应目录下 `grid_9/0.png`～`grid_9/8.png`。
  - 分镜选图逻辑见 `_fenjin_img_from_9grid`（景别/镜头类型可映射到 0～8，默认用 0）。
- **输出目录**：`data/audio/`（TTS）、`data/video/`（分镜视频与成片），路径在 `story_generate.py`、`video_gen.py`、`video_utils` 中写死或参数传入。

---

## 5. 运行与配置要点

- **工作目录**：运行 `story_generate.py` 时，当前目录应为 `Backend`，以便 `from tool.`、`from mora.`、`config_loader` 等导入正常。
- **配置方式**：LLM / T2I 等密钥与端点通过环境变量或 `Backend/.env`、`Backend/config.toml` 配置，详见 `config_loader` 与各模块文档/注释。
- **启动方式**：主入口为 `Backend/story_generate.py`，默认以 Gradio 启动在 `0.0.0.0:8082`，T2I 模型由环境变量 `T2I_BACKEND`、`T2I_MODEL` 或 `get_t2i_model` 参数指定。

---

## 6. 扩展与修改边界（给 Agent 的提示）

- **新增分镜视频后端**：在 `video_gen.py` 的 `_run_i2v_t2v` 或等价入口中增加 `backend`/`model_id` 分支，视频生成实现可放在 `mora/iterate_generate_video.py`。
- **新增 T2I 模型**：在 `mora/generate_image.py` 的 `get_t2i_model` 中增加分支，并保证返回对象实现 `generate_face_style(...)` 约定。
- **新增 LLM provider**：在 `llm.py` 的 `_generate_response` 中增加分支，并在 `config_loader.get_llm_config` 或环境中提供对应 api_key/base_url/model。
- **修改分镜图逻辑**：分镜选图在 `story_generate.py` 的 `_fenjin_img_from_9grid`、`generate_single_fenjin_img`、`generate_all_fenjin_imgs` 中；若改为始终走 IP-Adapter，设置 `use_legacy_consistency = True` 并恢复相应分支代码。
- **9 宫格景别映射**：`GRID_9_PROMPT_SUFFIXES` 与 0～8 的语义在 `mora/consistency_9grid.py`；分镜选用哪一格在 `_fenjin_img_from_9grid(grid_index=...)`，当前默认 0，可按「镜头类型/景别」传入不同索引。

以上为当前代码反映的主要产品功能、结构与模块职责；实现细节以仓库内具体文件为准。
