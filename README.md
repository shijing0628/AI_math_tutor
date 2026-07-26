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
