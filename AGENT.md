# AI 视频生成系统架构升级指引（AGENT.md）

本文档为整体升级方案的提炼与指引，供开发与 AI Agent 在实施时统一遵循。完整细节可参考项目内更详细的方案文档。

---

## 一、升级目标概览

在现有「故事 → 角色/场景 → 分镜 → 图/音/视频 → 成片」管线基础上：

1. **模型层可插拔**：LLM、文生图/图生图、文生视频/图生视频、TTS 均支持「商用 + 开源」多后端，**通过参数指定** provider / model。
2. **前端按流程拆 Tab**：5 个一级 Tab 对应 5 个阶段，中间结果通过 State **自动导入下一环节**，每环节可**单独运行与验证**。
3. **每生成 Tab 单独可测**：每个有生成能力的 Tab 提供「本页测试」按钮与对应**测试函数**，用内置示例单页跑通，不依赖其它 Tab。

---

## 二、现状与主文件

| 模块 | 主文件 | 当前状况 |
|------|--------|----------|
| 大模型 | `Backend/llm.py` | g4f、moonshot、glm、openai、coze；部分 api_key 硬编码 |
| 文生图/图生图 | `Backend/mora/generate_image.py` | get_t2i_model → sdxl_lightning / sd15 / ipadapter，均为本地 |
| 图生视频 | `Backend/mora/iterate_generate_video.py` | SVD、I2VGen-XL、AnimateDiff 等，未接入主流程 |
| 分镜“视频” | `Backend/video_utils.py` | image_to_video_with_audio_subtitle 为图+音+字幕幻灯片，非模型 I2V |
| TTS | `Backend/tts_utils.py`、`Backend/tts/edgetts_demo.py` | 仅 Edge-TTS |
| 一致性图 | `Backend/mora/consistency_9grid.py` | 9 宫格，nano_banana / local |
| 前端 | `Backend/story_generate.py` | 单一大 Tab「故事动画生成」，内嵌多子 Tab，阶段边界不清 |

---

## 三、升级要点速查

### 3.1 大模型（LLM）

- **改造文件**：`Backend/llm.py`
- **配置**：api_key、base_url、model 从环境变量或 `config.toml` / `.env` 读取，**删除硬编码密钥**。
- **参数化**：所有对外接口统一支持 `llm_provider`、`model_name`，向下传到 `_generate_response`。
- **商用**：OpenAI、Anthropic、Google Gemini、Moonshot、GLM、通义等。
- **开源**：DeepSeek、Qwen、Llama/Mistral（OpenAI 兼容或 LiteLLM）。

### 3.2 文生图 / 图生图（T2I / I2I）

- **改造文件**：`Backend/mora/generate_image.py`
- **统一接口**：如 `generate(prompt, negative_prompt, ...)`、`generate_from_image(image, prompt, ...)`，再按 backend 分发。
- **商用（指定）**：Google **Nano Banana Pro**（Gemini 3 Pro Image）、**阿里通义万相**、**字节**、**Kling**（按各自开放 API 接入）。
- **开源（指定）**：**Flux**（Dev/Schnell）、**阿里 Z-Image-Turbo**；保留 sdxl_lightning、sd15、ipadapter。
- **参数**：`get_t2i_model(..., backend="open_source"|"commercial", model_id=...)`，UI 提供「商用/开源」+「具体模型」下拉。

### 3.3 文生视频 / 图生视频（T2V / I2V）

- **改造文件**：`Backend/video_utils.py`、`Backend/mora/iterate_generate_video.py`，可新建或扩展 `Backend/video_gen.py`
- **统一入口**：`generate_shot_video(images, audio_path, subtitle_path, backend="slideshow"|"i2v"|"t2v", model_id=...)`
  - `slideshow`：沿用现有图+音+字幕合成。
  - `i2v` / `t2v`：调用真实 I2V/T2V 模型再合成。
- **开源（指定）**：**Wan 2.2** 为主选（T2V/I2V/TI2V，Diffusers/ComfyUI）；可选 SVD、CogVideoX、AnimateDiff 等。
- **商用**：Runway、Kling、Minimax、Luma 等若开放 API，按 backend + model_id 接入。

### 3.4 TTS

- **改造文件**：`Backend/tts_utils.py`、`Backend/tts/` 下实现
- **统一接口**：`synthesize(text, voice_or_speaker_id, rate, output_path, subtitle_path=None, backend=None, model_id=None)`
- **必须新增**：**Qwen3-TTS**（阿里云百炼 Qwen-TTS 实时语音），backend/model_id 如 `qwen3_tts` / `qwen_tts`。
- **保留/可选**：Edge-TTS（默认开源）、CosyVoice、OpenVoice、ElevenLabs、Azure、OpenAI TTS 等。
- **调用处**：`generate_text_audio` 及所有调用方增加 `tts_backend`、`tts_model_id` 并下传。

---

## 四、前端 Tab 与 State 设计

### 4.1 五个一级 Tab

```
Tab1: 故事与剧本     — 主题/梗概扩写、小说解析、按段摘要
Tab2: 角色与场景     — 角色/场景抽取、按段标注、电影化分镜
Tab3: 角色/场景图与分镜画面 — 设定图、9 宫格、每镜选图
Tab4: 配音与分镜视频 — TTS、幻灯片或 I2V 生成分镜短片
Tab5: 成片导出       — 片头片尾、BGM、分镜列表合成
（可选）Tab0: 全局设置 — LLM / T2I / T2V / TTS 的 provider、model
```

### 4.2 State 与跨 Tab 流转

- 使用命名清晰的 `gr.State`：`state_full_story`、`state_segments`、`state_global_characters`、`state_global_scenes`、`state_storyboards`、`state_shot_images`、`state_shot_audios`、`state_shot_videos`。
- Tab1 输出 → Tab2 默认输入；Tab2 输出 → Tab3；Tab3 选定图 → Tab4；Tab4 分镜视频列表 → Tab5 自动带入，无需用户手动传递。

### 4.3 每个生成 Tab 的单独测试（必须）

每个有生成能力的 Tab 需提供：

- **「本页测试」按钮**
- **对应测试函数**，用内置示例仅在本 Tab 内跑通生成链路，结果在本 Tab 展示

| Tab | 测试函数 | 行为简述 |
|-----|----------|----------|
| Tab1 | `test_tab1_story()` | 固定主题/梗概 → 扩写或分段提炼 → 本 Tab 展示故事/分段 |
| Tab2 | `test_tab2_characters_scenes()` | 固定短故事 → 角色场景/电影化分镜 → 本 Tab 展示 |
| Tab3 | `test_tab3_refs_and_shots()` | 固定 prompt + 当前 T2I → 示例角色/场景/分镜图 → Gallery 展示 |
| Tab4 | `test_tab4_tts_and_shot_video()` | 固定旁白 TTS + 可选示例图+幻灯片/I2V → 示例音频/分镜视频 → 本 Tab 展示 |
| Tab5 | `test_tab5_final_export()` | 内置示例分镜路径或占位片段 → 合成短线成片 → 本 Tab 展示；缺片段时可提示先跑 Tab4 测试 |

测试函数**不依赖其它 Tab 产出**，尽量不写入跨 Tab State，专注「单页可验证」。

---

## 五、实施顺序与关键文件改动摘要

**推荐顺序**：  
1）LLM 配置与 provider/model 参数化 → 2）T2I/I2I 抽象与商用/开源后端 → 3）TTS 抽象与多引擎（含 Qwen3-TTS）→ 4）I2V/T2V 统一入口与 Wan2.2 等后端 → 5）前端 Tab 重划、State 流转与各 Tab 测试函数 → 6）联调与文档。  
其中 2、3 可并行；5 依赖 1–4 的接口基本稳定。

**关键文件与改动一句话**：

| 方向 | 主要文件 | 改动摘要 |
|------|----------|----------|
| LLM | `Backend/llm.py` | 配置外置、去硬编码 key；扩展 provider/model；对外函数统一增加 llm_provider、model_name |
| T2I/I2I | `Backend/mora/generate_image.py` | 抽象接口；商用：Nano Banana Pro、阿里、字节、Kling；开源：Flux、Z-Image-Turbo；get_t2i_model(backend, model_id) |
| TTS | `Backend/tts_utils.py`、`Backend/tts/` | 统一 synthesize；**新增 Qwen3-TTS**；调用方增加 tts_backend、tts_model_id |
| I2V/T2V | `Backend/video_utils.py`、`Backend/mora/iterate_generate_video.py`、可选 `Backend/video_gen.py` | 统一入口 slideshow/i2v/t2v；**开源主选 Wan2.2**；商用按需 HTTP 接入 |
| 前端 | `Backend/story_generate.py` | 5 个一级 Tab；State 命名与跨 Tab 传递；**每生成 Tab 实现 test_tab* 与「本页测试」按钮**；模型选择与各环节绑定 |
| 配置 | `config.toml.example` 或 `.env.example`（项目根或 Backend） | 列齐各模块 key、base_url、默认 model，便于部署与安全审查 |

---

## 六、方案既定选型（必须遵循）

1. **文生图/图生图**  
   - 商用：Google Nano Banana Pro，以及阿里（通义万相）、字节、Kling。  
   - 开源：Flux，以及阿里 Z-Image-Turbo 等。

2. **文生视频/图生视频**  
   - 开源以 **Wan 2.2** 等为主选，在统一入口与模型下拉中可选。

3. **TTS**  
   - 必须增加 **Qwen3-TTS**（阿里云百炼 Qwen-TTS 实时语音）。

4. **每个生成 Tab**  
   - 必须有**单独测试功能**：对应 `test_tab1_story()` … `test_tab5_final_export()` 及「本页测试」按钮，用内置示例单页验证，不依赖其它 Tab。

---

## 七、风险与兼容

- 商用 API：可配置开关或回退到开源默认，文档中说明计费与限额。
- 开源大模型/大图/视频模型：通过配置或环境变量控制加载，避免默认占满显存。
- 非 OpenAI 兼容 API（如 Coze）：在「参数指定 provider」框架内单独分支或适配。
- **向后兼容**：新增参数均设默认值，不改参数时行为与当前一致。

---

*本文档为架构升级的指引摘要，实施时以本仓库实际代码与更完整方案文档为准。*
