# AI Math Tutor

Grade 12 math AI tutor built with **Streamlit** + **LangGraph** + **Azure OpenAI**.

## Features

- LangGraph workflow: Teach → Quiz → Evaluate → Remediation / Practice / Master
- Streamlit UI (Lesson, Chat, Quiz, Visual, Progress)
- JSON persistence for progress and content caches
- Cache-first design to reduce API token usage

## Setup

1. Clone the repo and install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a local `.env` from the example (do **not** commit real keys):

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux
```

3. Edit `.env` with your Azure OpenAI values:

```
AZURE_OPENAI_API_KEY=...
AZURE_ENDPOINT=...
AZURE_MODEL=...
```

4. Optional starter profile:

```bash
copy student_profile.example.json student_profile.json   # Windows
```

5. Run the app:

```bash
python -m streamlit run app.py
```

## Main files

| File | Role |
|------|------|
| `app.py` | Streamlit UI |
| `tutor_graph.py` | LangGraph tutoring workflow |
| `course_roadmap.json` | Ordered concepts |
| `AI-tutor-langgraph.ipynb` | Earlier Colab / LangGraph practice notebook |
| `readme.txt` | Chinese notes on the graph design |

## Security

- Secrets live only in `.env` (gitignored).
- Use `.env.example` as the template.
- Rotate any keys that were ever pasted into notebooks or chat.


tutor_graph.py 是整个 AI 家教的"大脑"，用 LangGraph 把教学流程做成了一张状态图（StateGraph）。Streamlit（app.py）只负责界面，真正的教学决策流程都在这张图里跑。

### 整体流程图
开始 (START)
   ↓
Teaching（讲课）
   ↓
Quiz（出题）
   ↓
AwaitAnswer（暂停，等学生作答）← interrupt 中断点
   ↓
Evaluate（批改评分）
   ↓ 按分数分流（条件边）
   ├── 分数 < 60  → Remediation（重新教，更简单）
   ├── 60 ~ 84   → Practice（针对弱点练习）
   └── ≥ 85      → Coach（鼓励，准备进入下一课）
   ↓
SaveProgress（把进度写进 student_profile.json）
   ↓
结束 (END)

### 每个部分的逻辑
1. TutorState（状态）
图里流动的"数据包"，是一个 TypedDict，包含：学生姓名、当前概念（concept）、章节、讲课内容、试卷、学生答案、得分、评价结果、分支内容等。每个节点读状态、改状态，然后传给下一个节点。

2. Teaching 节点 —— 缓存优先，省 token
先查 lesson_cache.json：如果这个概念的课已经生成过，直接从缓存读，不调用 API（lesson_from_cache: True）。
缓存没有才调用 Azure OpenAI 生成讲解（解释、例子、常见错误、一道练习题），生成后写回缓存。
3. Quiz 节点 —— 同样缓存优先
先查 quiz_cache.json，有就直接用；没有才让 LLM 出一份卷子（选择题、判断题、应用题 + 答案），然后存缓存。
4. AwaitAnswer 节点 —— 关键的"暂停"
answer = interrupt(payload)
这是 LangGraph 的 interrupt（人机交互中断）。图跑到这里会停下来，把状态存进 checkpointer，一直等到 Streamlit 里学生点"Submit quiz for grading"，再用 Command(resume=学生答案) 把答案送回来，图从这里继续往下走。这就解决了笔记本原版"必须提前伪造学生答案"的问题。

5. Evaluate 节点 —— 批改
把试卷 + 学生答案发给 LLM，要求返回 JSON：mastery_score（分数）、feedback（评语）、强项、弱点、建议练习、next_action。如果 JSON 解析失败，兜底给 0 分走补救路线。

6. route_after_evaluation —— 纯 Python 分流，不花钱
分数 ≥ 85 → "coach"
60 ~ 84  → "practice"
< 60     → "remediation"
这是条件边（conditional edge），只是普通的 if 判断，不调用任何 API。

7. 三个分支节点
Remediation（< 60）：让 LLM 用更简单的方式重新讲一遍 + 一道很容易的练习。
Practice（60–84）：针对评出来的弱点，给 2 道带提示的练习题。
Coach（≥ 85）：只用字符串模板生成鼓励语，完全不调 API——掌握了就没必要再花 token。
8. SaveProgress 节点 —— 写 JSON，纯 Python
往 history 里追加一条记录（概念、分数、时间、动作）。
分数 ≥ 85：加入 completed、移出 weak、current_index + 1（进入下一个概念）。
分数不够：加入 weak，停留在当前概念，等你重考。
最后写入 student_profile.json——这就是你在 Progress 页看到的内容。
省 API 费用的设计总结
环节	是否调 API
讲课 / 出题（缓存命中时）
否
等待答案、分数分流
否（纯 Python）
Coach 鼓励语
否（模板）
更新进度、写 JSON
否
首次生成课程 / 出题、批改、补救/练习内容
是（必要的才调）
文件底部的辅助函数
给 app.py 用的接口：

build_tutor_graph(llm_fn) —— 组装并编译图（带 MemorySaver 检查点，interrupt 必需）。
start_until_quiz() —— 从头跑到"等答案"处暂停。
submit_quiz_answer() —— 用学生答案恢复运行，跑完批改和分支。
is_waiting_for_answer() —— 判断图是不是正停在等答案。
make_thread_id() —— 每次做题生成一个会话 ID，LangGraph 靠它记住"暂停在哪里"。