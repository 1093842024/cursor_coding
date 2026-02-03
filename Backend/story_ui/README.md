# story_ui — 五大环节 UI 模块

本包包含故事可视化视频生产线的五个 Gradio Tab 的界面与逻辑，由 `Backend/story_generate.py` 通过 `build_app(t2imodel)` 组装后启动。

## 环境与依赖（uv）

在项目根目录执行（需已安装 [uv](https://github.com/astral-sh/uv)）：

```bash
uv sync
```

在 **Backend** 目录下用 `uv run` 启动各子页面：

```bash
cd Backend
uv run story_ui/tab1_story.py    # 默认 8082
uv run story_ui/tab2_characters.py  # 8083
uv run story_ui/tab3_refs_shots.py  # 8084
uv run story_ui/tab4_voice_video.py # 8085
uv run story_ui/tab5_export.py       # 8086
```

## 目录结构

- **shared.py** — 常量、默认 prompt、跨 Tab 小工具（如图库选择/移除、BGM、mask、库读写）
- **tab1_story.py** — 1、故事与剧本（主题扩写、小说解析、按段摘要、按段标注角色场景）
- **tab2_characters.py** — 2、角色与场景（角色/场景抽取、电影化分镜、fenjin_parse）
- **tab3_refs_shots.py** — 3、角色/场景图与分镜画面（设定图、9 宫格、分镜选图）
- **tab4_voice_video.py** — 4、配音与分镜视频（TTS、分镜视频、角色/场景/动作库）
- **tab5_export.py** — 5、成片导出（分镜串联、片头片尾、BGM）
- **app.py** — `build_app(t2imodel)` 组装入口

## 单 Tab 独立运行与调试

在 **Backend** 目录下执行（需已 `uv sync` 安装依赖）：

```bash
cd Backend
uv run story_ui/tab1_story.py    # 默认 8082
uv run story_ui/tab2_characters.py  # 8083
uv run story_ui/tab3_refs_shots.py  # 8084
uv run story_ui/tab4_voice_video.py # 8085
uv run story_ui/tab5_export.py       # 8086
```

各模块内 `run_standalone(server_name, server_port)` 可改端口。

## 契约说明

- 各 tab 提供 **build(ctx)**：在当前 `gr.Blocks()` 下挂一个 `gr.Tab(...)`，从 `ctx` 读已有 state/控件，向 `ctx` 写入本 Tab 需给后续 Tab 使用的组件。
- 主入口在 `app.build_app()` 中依次调用 `tab1.build(ctx)`～`tab5.build(ctx)`，通过 `ctx` 传递跨 Tab 的 state 与控件引用。
- 详细拆分方案见 `Backend/REFACTOR_PLAN.md`。
