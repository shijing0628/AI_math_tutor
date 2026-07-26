# AI Math Tutor

Grade 12 math AI tutor built with **Streamlit** + **LangGraph** + **Azure OpenAI**,
with optional **Supabase** progress sync.

Live demo: https://grade12-math-tutor.streamlit.app/

## Features

- **AI Agent tab** — ReAct agent chooses tools: progress, textbook search, teach, quiz, grade, save
- **Fixed LangGraph workflow** — Lesson/Quiz tabs: Teach → Quiz → Evaluate → Remediation / Practice / Master
- Cache-first lessons/quizzes (JSON) to reduce API token use
- Visual questions with textbook page images + photo upload grading
- Progress: local JSON + optional Supabase cloud sync

## Project structure

```text
app.py                      Streamlit UI
tutor_graph.py              Fixed tutoring StateGraph (Lesson/Quiz)
tutor_tools.py              Agent @tool skills
tutor_agent.py              create_react_agent + SQLite memory
textbook_index.py           PDF chunk index builder / search
textbook_chunks.json        Prebuilt textbook search index
profile_store.py            Profile load/save (JSON + Supabase)
supabase_setup.sql          Table + RLS policies for cloud sync
course_roadmap.json         Ordered concepts
page_images/                Textbook page PNGs
grade12math.pdf             Source textbook
run_tutor_agent_local.py    CLI agent playground
test_agent_e2e.py           End-to-end agent session test
AI-tutor-langgraph.ipynb    Earlier Colab / LangGraph practice notebook
.env.example                Secrets template (never commit real .env)
```

## Local setup

```bash
pip install -r requirements.txt
copy .env.example .env          # Windows
# cp .env.example .env          # macOS / Linux
```

Fill `.env`:

```env
AZURE_OPENAI_API_KEY=...
AZURE_ENDPOINT=...
AZURE_MODEL=...

# Optional cloud progress (after running supabase_setup.sql)
SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_KEY=YOUR_PUBLISHABLE_OR_ANON_KEY
```

Optional starter profile:

```bash
copy student_profile.example.json student_profile.json
```

If you change the PDF, rebuild the text index:

```bash
python textbook_index.py
```

Run the app:

```bash
python -m streamlit run app.py
```

### Local agent tests

```bash
python run_tutor_agent_local.py "What concept should I study next?"
python run_tutor_agent_local.py --interactive
python test_agent_e2e.py
```

## Streamlit Cloud deploy

1. Connect the GitHub repo and set main file to `app.py`.
2. In **App settings → Secrets**, add (TOML):

```toml
AZURE_OPENAI_API_KEY = "..."
AZURE_ENDPOINT = "https://....openai.azure.com/openai/v1"
AZURE_MODEL = "..."
SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
SUPABASE_KEY = "..."
```

3. Redeploy / reboot after changing secrets.

## Supabase progress sync

1. Run [`supabase_setup.sql`](supabase_setup.sql) in the Supabase SQL editor
   (creates `student_profiles` **and** RLS policies — required for writes).
2. Set `SUPABASE_URL` + `SUPABASE_KEY` in `.env` / Streamlit secrets.
3. Confirm rows appear under **Table Editor → student_profiles**.

Without Supabase, the app still works using local `student_profile.json`
(that file is gitignored and resets on Streamlit Cloud restarts).

## Security

- Never commit `.env`, API keys, or `student_profile.json`.
- Use `.env.example` as the template.
- Rotate any keys that were ever shared in chat or old notebooks.

---

## 架构说明：为什么这样设计？（中文）

### 一句话总结

本项目同时有**两套控制方式**，共用同一套“能力”（讲课、出题、批改、存进度）：

| 方式 | 入口 | 谁决定下一步 | 核心文件 |
|------|------|--------------|----------|
| **Workflow（固定流程）** | Lesson / Quiz 等 Tab + 按钮 | 代码写死的路线 | `tutor_graph.py` |
| **Agent（智能体）** | **AI Agent** Tab + 聊天 | LLM 自己选 Tool | `tutor_tools.py` + `tutor_agent.py` |

所以：**AI Agent 理论上能完成“学、考、批、存”等与其他 Tab 同类的事**；但 Visual（课本插图、画图、拍照上传批改）等还没完全收进 Agent，因此其他 Tab 仍保留。

### 分层结构（为什么拆成 Tools / Skills）

| 层级 | 文件 | 职责 |
|------|------|------|
| **界面** | `app.py` | 只负责按钮、聊天、展示；尽量不藏业务逻辑 |
| **能力 / Skills** | `tutor_tools.py` | 真正干活的 `@tool`：查进度、搜课本、讲课、出题、批改、存档 |
| **决策 / Agent** | `tutor_agent.py` | `create_react_agent`：看学生说什么，决定调用哪个 tool |
| **固定流程** | `tutor_graph.py` | Teaching → Quiz → 等人作答 → Evaluate → 分支 → Save |
| **课本检索** | `textbook_index.py` | PDF 切块 + 关键词搜索（不花 token） |
| **进度存储** | `profile_store.py` | 本地 JSON + 可选 Supabase 云同步 |

这样拆的原因：

1. **可复用**：同一个 `teach_concept`，Agent、CLI、以后的 API 都能用。  
2. **可测试**：可单独测 tool / 跑 `test_agent_e2e.py`，不必每次点完整 UI。  
3. **省 token**：查进度、存档、搜课本是纯 Python；讲课/出题优先读缓存。  
4. **可扩展**：以后加 `grade_photo` 等，只需加 tool，不必重写整张图。  
5. **职责清晰**：LLM 负责“选哪一步”；确定性逻辑（分数分流、写库）尽量用代码。

### `@tool` 是什么？为什么不是普通函数？

普通 Python 函数对 LLM 是“看不见的”。  
`@tool` 会把函数的**名字、参数、docstring（说明何时该用）**交给 Agent，模型按说明选工具，而不是瞎猜。

例如学生问“我该学什么”，Agent 会先调 `get_current_concept`，因为该 tool 的说明就是返回当前 roadmap 概念。

### 两层 LLM 调用（容易混淆）

```text
学生一句话
    ↓
【决策 LLM】tutor_agent 里的 ChatOpenAI
    “要不要调用 teach_concept？”
    ↓
【能力 LLM】tool 内部的 llm_fn（讲课 / 出题 / 批改）
    “真正生成讲义、试卷、评语”
```

中间还有**零 API 的 tool**：`get_current_concept`、`update_progress`、`search_textbook`。

### Agent 的七个 Tool

| Tool | 作用 | 是否调 API |
|------|------|------------|
| `get_current_concept` | 读 profile + roadmap | 否 |
| `list_roadmap_concepts` | 列出后续课题 | 否 |
| `search_textbook` | 搜课本原文片段 | 否 |
| `teach_concept` | 讲课（优先 lesson 缓存） | 缓存命中则否 |
| `make_quiz` | 出题（优先 quiz 缓存） | 缓存命中则否 |
| `grade_answer` | 批改（尽量结构化 JSON） | 是 |
| `update_progress` | 写 JSON / Supabase | 否（只写库） |

`tutor_agent.py` 里的 `SYSTEM_PROMPT` 规定策略，例如：学新内容时先搜课本再讲课；批完必须 `update_progress`。

### Workflow vs Agent（对照）

```text
Workflow（tutor_graph.py）—— 路线写死：
  Teaching → Quiz → AwaitAnswer(interrupt) → Evaluate
      → Remediation / Practice / Coach → SaveProgress → END

Agent（tutor_tools + tutor_agent）—— 模型拼路线：
  “我该学啥” → get_current_concept
  “教我”     → search_textbook + teach_concept
  “考我”     → make_quiz
  “这是答案” → grade_answer + update_progress
```

能力重叠，**控制权不同**：一个在代码里，一个在模型里。

### 为什么还保留 Lesson / Quiz / Visual 等 Tab？

| Tab | Agent 能否替代 | 为何保留 |
|-----|----------------|----------|
| **AI Agent** | — | 自然语言一站式自动化 |
| **Lesson / Quiz** | 大部分能 | 按钮可控、适合演示固定 LangGraph；少一次“决策”开销 |
| **Ask Tutor** | 部分能 | 轻量答疑，不必每次跑全套 tool |
| **Visual Question** | 目前不能完全 | 插图、画图、拍照上传批改仍在此 Tab |
| **Progress** | 能读进度，UI 更直观 | 看 history / weak / 手动跳课 |

工程上常见做法：**先加 Agent，不立刻删稳定旧 UI**；确认够用后再考虑合并。

### 固定流程图（Lesson / Quiz）

```text
START
  → Teaching（缓存优先）
  → Quiz（缓存优先）
  → AwaitAnswer（interrupt，等人提交）
  → Evaluate
  → 分数分流（纯 Python，不调 API）
       <60  Remediation
       60–84 Practice
       ≥85  Coach（模板，不调 API）
  → SaveProgress（JSON / Supabase）
  → END
```

### 数据与部署注意

- **本地**：`.env` + `student_profile.json`；Agent 对话记忆在 `tutor_agent_memory.sqlite`（不入库）。  
- **云端 Streamlit**：必须用 **Secrets** 配 Azure / Supabase，不会读你电脑上的 `.env`。  
- **Supabase**：必须跑 `supabase_setup.sql`（含 **RLS 策略**），否则表一直是空的、写入会被 401 拒绝。  
- **先本地测再 push**：Streamlit Cloud 跟着 GitHub `main` 自动部署，边改边 push 容易把线上弄挂。
