"""
Grade 12 Math AI Tutor — local Streamlit app.

Uses grade12math.pdf page images + Azure OpenAI to teach by chapter,
chat with the student, quiz them, and persist progress in JSON.
"""

from __future__ import annotations

import base64
import json
import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from profile_store import load_profile, save_profile, supabase_enabled
from image_attachments import clear_pending_images, set_pending_images
from tutor_agent import build_tutor_agent as build_agent
from tutor_graph import (
    build_tutor_graph,
    get_graph_values,
    is_waiting_for_answer,
    make_thread_id,
    split_quiz_display,
    start_until_quiz,
    submit_quiz_answer,
)

# ============================================================
# Config
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE, override=True)

PROFILE_FILE = BASE_DIR / "student_profile.json"
ROADMAP_FILE = BASE_DIR / "course_roadmap.json"
LESSON_CACHE_FILE = BASE_DIR / "lesson_cache.json"
QUIZ_CACHE_FILE = BASE_DIR / "quiz_cache.json"
IMAGE_QUESTION_CACHE_FILE = BASE_DIR / "image_question_cache.json"
VISUAL_INDEX_FILE = BASE_DIR / "visual_index.json"
CHAT_HISTORY_FILE = BASE_DIR / "chat_history.json"
PAGE_IMAGES_DIR = BASE_DIR / "page_images"
PDF_FILE = BASE_DIR / "grade12math.pdf"

# All Azure settings come from .env (no secrets hardcoded here)
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "").strip()
AZURE_MODEL = os.getenv("AZURE_MODEL", "").strip()
PLACEHOLDER_KEYS = {
    "",
    "PASTE_YOUR_AZURE_OPENAI_API_KEY_HERE",
    "your_azure_openai_api_key_here",
}
DEFAULT_PROFILE = {
    "student_name": "Tom",
    "current_index": 0,
    "completed": [],
    "weak": [],
    "history": [],
}


# ============================================================
# JSON Helpers
# ============================================================

def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def clean_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


# ============================================================
# Azure LLM
# ============================================================

def get_api_key() -> str:
    # .env / process env first, then Streamlit Cloud secrets. Never shown in the UI.
    key = (os.environ.get("AZURE_OPENAI_API_KEY") or "").strip()
    if not key:
        try:
            key = str(st.secrets.get("AZURE_OPENAI_API_KEY", "")).strip()
        except Exception:
            key = ""
    if key in PLACEHOLDER_KEYS:
        return ""
    return key


def azure_config_error() -> str | None:
    if not get_api_key():
        return (
            "AZURE_OPENAI_API_KEY is not set. "
            f"Locally: edit `{ENV_FILE}` and restart Streamlit. "
            "On Streamlit Cloud: add it under App settings → Secrets."
        )
    if not AZURE_ENDPOINT:
        return f"AZURE_ENDPOINT is missing in `{ENV_FILE}`."
    if not AZURE_MODEL:
        return f"AZURE_MODEL is missing in `{ENV_FILE}`."
    return None


def get_client() -> OpenAI | None:
    if azure_config_error():
        return None
    return OpenAI(api_key=get_api_key(), base_url=AZURE_ENDPOINT)


def call_llm(prompt: str, system: str | None = None) -> str:
    err = azure_config_error()
    if err:
        return err

    client = get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=AZURE_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content or ""


def call_llm_with_images(prompt: str, image_paths: list[str], system: str | None = None) -> str:
    """Vision call: send textbook page images with the prompt."""
    err = azure_config_error()
    if err:
        return err

    client = get_client()
    content = [{"type": "text", "text": prompt}]
    for rel in image_paths[:3]:
        path = BASE_DIR / rel if not Path(rel).is_absolute() else Path(rel)
        if not path.exists():
            continue
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    response = client.chat.completions.create(
        model=AZURE_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content or ""


def call_llm_with_uploaded_images(
    prompt: str,
    uploaded_files: list,
    system: str | None = None,
) -> str:
    """Vision call: send student-uploaded photos (Streamlit UploadedFile) with the prompt."""
    err = azure_config_error()
    if err:
        return err

    client = get_client()
    content = [{"type": "text", "text": prompt}]
    for uf in uploaded_files[:3]:
        b64 = base64.b64encode(uf.getvalue()).decode("utf-8")
        mime = uf.type or "image/png"
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            }
        )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    response = client.chat.completions.create(
        model=AZURE_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content or ""


# ============================================================
# Tutor Logic
# ============================================================

def get_current_item(roadmap: list, profile: dict):
    if not roadmap:
        return None
    idx = int(profile.get("current_index", 0))
    if idx < 0 or idx >= len(roadmap):
        return None
    return roadmap[idx]


def concept_images(concept: str, visual_index: dict) -> list[str]:
    paths = visual_index.get(concept, []) or []
    return [p for p in paths if (BASE_DIR / p).exists()]


def list_all_page_images() -> list[str]:
    if not PAGE_IMAGES_DIR.exists():
        return []
    files = sorted(
        PAGE_IMAGES_DIR.glob("page_*.png"),
        key=lambda p: int(re.sub(r"\D", "", p.stem) or 0),
    )
    return [f"page_images/{p.name}" for p in files]


def save_visual_index(visual_index: dict) -> None:
    save_json(VISUAL_INDEX_FILE, visual_index)
    st.session_state.visual_index = visual_index


def infer_diagram_from_question(question: str) -> dict | None:
    """Build a simple plot when a question mentions labeled points like A(-2, 5)."""
    matches = re.findall(
        r"([A-Za-z])\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)",
        question or "",
    )
    if len(matches) < 2:
        return None
    labels = [m[0] for m in matches[:4]]
    points = [[float(m[1]), float(m[2])] for m in matches[:4]]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    pad = 2
    return {
        "type": "line_points",
        "points": points,
        "labels": labels,
        "show_line": True,
        "x_range": [min(xs) - pad, max(xs) + pad],
        "y_range": [min(ys) - pad, max(ys) + pad],
        "title": "Diagram for this question",
    }


def ensure_diagram(img_q: dict) -> dict:
    if not isinstance(img_q, dict):
        return img_q
    diagram = img_q.get("diagram")
    if isinstance(diagram, dict) and diagram.get("type") not in (None, "", "none"):
        return img_q
    inferred = infer_diagram_from_question(img_q.get("question", ""))
    if inferred:
        img_q = dict(img_q)
        img_q["diagram"] = inferred
    return img_q


def render_diagram(diagram: dict | None):
    """Draw a coordinate diagram for visual questions (points / line / function)."""
    if not diagram or not isinstance(diagram, dict):
        return
    dtype = (diagram.get("type") or "none").lower()
    if dtype in ("", "none"):
        return

    x_range = diagram.get("x_range") or [-5, 5]
    y_range = diagram.get("y_range") or [-5, 5]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.axhline(0, color="#444", linewidth=1)
    ax.axvline(0, color="#444", linewidth=1)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(diagram.get("title") or "Coordinate diagram")

    points = diagram.get("points") or []
    labels = diagram.get("labels") or []
    for i, pt in enumerate(points):
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        x, y = float(pt[0]), float(pt[1])
        ax.scatter([x], [y], s=70, zorder=5, color="#c0392b")
        label = labels[i] if i < len(labels) else f"P{i + 1}"
        ax.annotate(
            f"{label}({x:g}, {y:g})",
            (x, y),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
        )

    if diagram.get("show_line") and len(points) >= 2:
        xs = [float(p[0]) for p in points[:2]]
        ys = [float(p[1]) for p in points[:2]]
        if xs[1] != xs[0]:
            m = (ys[1] - ys[0]) / (xs[1] - xs[0])
            b = ys[0] - m * xs[0]
            x_line = np.linspace(x_range[0], x_range[1], 200)
            ax.plot(x_line, m * x_line + b, color="#2980b9", linewidth=2)
        else:
            ax.axvline(xs[0], color="#2980b9", linewidth=2)

    expr = (diagram.get("function") or "").strip()
    if expr:
        # Safe-ish eval for simple math expressions in x
        allowed = {
            "x": None,
            "np": np,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "sqrt": np.sqrt,
            "abs": np.abs,
            "exp": np.exp,
            "log": np.log,
            "pi": np.pi,
            "e": np.e,
        }
        try:
            x_vals = np.linspace(x_range[0], x_range[1], 400)
            safe_expr = (
                expr.replace("^", "**")
                .replace("Math.", "")
            )
            y_vals = eval(  # noqa: S307 - controlled student/tutor diagram expr
                safe_expr,
                {"__builtins__": {}},
                {**allowed, "x": x_vals},
            )
            ax.plot(x_vals, y_vals, color="#8e44ad", linewidth=2, label=f"y = {expr}")
            ax.legend(loc="best")
        except Exception:
            ax.text(
                0.5,
                0.02,
                f"Could not plot function: {expr}",
                transform=ax.transAxes,
                ha="center",
                fontsize=8,
                color="red",
            )

    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def is_llm_error(text: str) -> bool:
    t = (text or "").strip()
    return t.startswith("AZURE_") or "is not set" in t or "is missing in" in t


def get_lesson(concept: str, chapter: str, lesson_cache: dict) -> str:
    if concept in lesson_cache:
        return lesson_cache[concept]

    prompt = f"""
You are a friendly Grade 12 math tutor (like a real classroom teacher).
Teach this concept from chapter "{chapter}":

{concept}

Include:
1. Simple explanation (plain language, not textbook-dense)
2. Real-world example
3. Visual analogy if useful
4. Common mistakes
5. One practice question (do NOT give the answer yet)

Keep it focused so the student does not need to read every textbook page.
Use clear steps and student-friendly language.
"""
    lesson = call_llm(prompt, system="You are an expert Grade 12 mathematics teacher.")
    if not is_llm_error(lesson):
        lesson_cache[concept] = lesson
        save_json(LESSON_CACHE_FILE, lesson_cache)
    return lesson


def get_quiz(concept: str, quiz_cache: dict) -> str:
    if concept in quiz_cache:
        return quiz_cache[concept]

    prompt = f"""
Create a Grade 12 math quiz for this concept:

{concept}

Include:
1. Two multiple choice questions
2. One true/false question
3. One application question
4. One short explanation question
5. Answer key at the end (clearly labeled)

Make it suitable for a student learning this topic for the first time.
"""
    quiz = call_llm(prompt, system="You are an expert Grade 12 mathematics teacher.")
    if not is_llm_error(quiz):
        quiz_cache[concept] = quiz
        save_json(QUIZ_CACHE_FILE, quiz_cache)
    return quiz


def get_image_question(
    concept: str,
    image_paths: list[str],
    image_question_cache: dict,
    force_refresh: bool = False,
) -> dict:
    if not force_refresh and concept in image_question_cache:
        return image_question_cache[concept]

    prompt = f"""
You are a Grade 12 mathematics teacher.

Concept: {concept}

Look at the textbook page image(s) provided (graphs, diagrams, or worked examples).
Create ONE visual / graph-based assessment grounded in what you see on the page.

Return JSON only in this exact format:
{{
  "question": "...",
  "answer_key": "...",
  "skills_tested": ["..."],
  "page_reference": "what on the page the question uses",
  "diagram": {{
    "type": "line_points",
    "points": [[-2, 5], [4, -1]],
    "labels": ["A", "B"],
    "show_line": true,
    "function": "",
    "x_range": [-6, 6],
    "y_range": [-4, 8],
    "title": "short title"
  }}
}}

diagram.type options:
- "line_points": use for points / lines on a coordinate plane
- "function": set function like "x**2 - 2*x" (python/numpy style, use ** for powers)
- "none": only if the textbook figure itself is enough and no extra plot is needed

Requirements:
1. The question must require interpreting a graph, curve, diagram, or visual.
2. Not only multiple choice.
3. Include a clear answer key.
4. Always fill diagram so the student can SEE the figure (unless type is none and pages are provided).
5. Grade 12 appropriate language.
6. No markdown fences.
"""

    if image_paths:
        result = call_llm_with_images(
            prompt,
            image_paths,
            system="You are an expert Grade 12 mathematics teacher who reads textbook figures carefully.",
        )
    else:
        result = call_llm(
            prompt
            + "\n\nNo textbook page image was attached. Invent a realistic visual question "
            "AND fill the diagram field so a plot can be drawn for the student.",
            system="You are an expert Grade 12 mathematics teacher.",
        )

    try:
        result_json = clean_json(result)
    except Exception:
        result_json = {
            "question": result,
            "answer_key": "",
            "skills_tested": [],
            "page_reference": "",
            "diagram": {"type": "none"},
        }

    result_json = ensure_diagram(result_json)

    if not is_llm_error(str(result_json.get("question", ""))):
        image_question_cache[concept] = result_json
        save_json(IMAGE_QUESTION_CACHE_FILE, image_question_cache)

    return result_json


def evaluate_answer(concept: str, quiz: str, image_question: dict, student_answer: str) -> dict:
    prompt = f"""
You are a Grade 12 mathematics teacher.
Evaluate the student's answer carefully.

Concept:
{concept}

Text Quiz:
{quiz}

Visual / Graph Question:
{json.dumps(image_question, ensure_ascii=False)}

Student Answer:
{student_answer}

Return JSON only:
{{
  "mastery_score": 85,
  "feedback": "Detailed feedback for the student.",
  "strengths": ["..."],
  "weaknesses": ["..."],
  "recommended_practice": ["..."],
  "next_action": "mastered/practice/remediation"
}}

Scoring:
- 85 to 100: mastered
- 60 to 84: practice
- below 60: remediation

No markdown.
"""
    result = call_llm(prompt, system="You are a careful Grade 12 math grader.")
    try:
        return clean_json(result)
    except Exception:
        return {
            "mastery_score": 0,
            "feedback": result,
            "strengths": [],
            "weaknesses": [],
            "recommended_practice": [],
            "next_action": "remediation",
        }


def chat_reply(
    concept: str,
    chapter: str,
    student_name: str,
    messages: list[dict],
    user_text: str,
) -> str:
    history_txt = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages[-12:]
    )
    prompt = f"""
You are a real Grade 12 math teacher tutoring {student_name}.
Current chapter: {chapter}
Current concept: {concept}

Be encouraging, clear, and Socratic when helpful.
Do not dump entire textbook pages — teach efficiently.
If the student is stuck, give a hint first, then more detail if they ask.
Use short paragraphs and steps.

Recent conversation:
{history_txt}

Student: {user_text}

Reply as the teacher only.
"""
    return call_llm(prompt, system="You are a patient, expert Grade 12 mathematics tutor.")


def apply_evaluation(profile: dict, concept: str, evaluation: dict) -> dict:
    score = int(evaluation.get("mastery_score", 0) or 0)
    action = (evaluation.get("next_action") or "").lower()

    profile.setdefault("history", []).append(
        {
            "concept": concept,
            "score": score,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "next_action": action,
        }
    )

    if score >= 85 or action == "mastered":
        if concept not in profile.setdefault("completed", []):
            profile["completed"].append(concept)
        if concept in profile.get("weak", []):
            profile["weak"] = [c for c in profile["weak"] if c != concept]
        profile["current_index"] = int(profile.get("current_index", 0)) + 1
    else:
        if concept not in profile.setdefault("weak", []):
            profile["weak"].append(concept)

    save_profile(profile, PROFILE_FILE)
    return profile


# ============================================================
# UI helpers
# ============================================================

def show_page_images(paths: list[str], caption_prefix: str = "Textbook") -> None:
    if not paths:
        st.info("No textbook page images mapped for this concept yet.")
        return
    cols = st.columns(min(len(paths), 3))
    for i, rel in enumerate(paths[:3]):
        path = BASE_DIR / rel
        if not path.exists():
            continue
        with cols[i % len(cols)]:
            st.image(str(path), caption=f"{caption_prefix}: {path.name}", use_container_width=True)


def get_tutor_graph():
    """Build (or reuse) the LangGraph tutor compiled with current Azure credentials."""
    fingerprint = f"{get_api_key()}|{AZURE_ENDPOINT}|{AZURE_MODEL}"
    if (
        "tutor_graph" not in st.session_state
        or st.session_state.get("tutor_graph_fingerprint") != fingerprint
    ):
        st.session_state.tutor_graph = build_tutor_graph(call_llm, BASE_DIR)
        st.session_state.tutor_graph_fingerprint = fingerprint
        st.session_state.graph_threads = {}
    return st.session_state.tutor_graph


def refresh_caches_from_disk() -> None:
    st.session_state.lesson_cache = load_json(LESSON_CACHE_FILE, {})
    st.session_state.quiz_cache = load_json(QUIZ_CACHE_FILE, {})
    st.session_state.image_question_cache = load_json(IMAGE_QUESTION_CACHE_FILE, {})


def ensure_quiz_ready(
    concept: str,
    chapter: str,
    profile: dict,
    *,
    force_refresh_lesson: bool = False,
    force_refresh_quiz: bool = False,
) -> dict:
    """
    Run the LangGraph through Teaching → Quiz and pause for the student answer.
    Cache hits inside the graph avoid extra API calls.
    """
    graph = get_tutor_graph()
    threads = st.session_state.setdefault("graph_threads", {})

    if force_refresh_lesson or force_refresh_quiz:
        threads.pop(concept, None)

    tid = threads.get(concept)
    if tid and is_waiting_for_answer(graph, tid) and not (
        force_refresh_lesson or force_refresh_quiz
    ):
        return get_graph_values(graph, tid)

    tid = make_thread_id(profile.get("student_name", "Student"), concept)
    threads[concept] = tid
    result = start_until_quiz(
        graph,
        {
            "student_name": profile.get("student_name", "Student"),
            "concept": concept,
            "chapter": chapter,
            "profile": profile,
            "force_refresh_lesson": force_refresh_lesson,
            "force_refresh_quiz": force_refresh_quiz,
        },
        tid,
    )
    refresh_caches_from_disk()
    return result


def init_session() -> None:
    if "profile" not in st.session_state:
        st.session_state.profile = load_profile(PROFILE_FILE)
    if "roadmap" not in st.session_state:
        st.session_state.roadmap = load_json(ROADMAP_FILE, [])
    if "lesson_cache" not in st.session_state:
        st.session_state.lesson_cache = load_json(LESSON_CACHE_FILE, {})
    if "quiz_cache" not in st.session_state:
        st.session_state.quiz_cache = load_json(QUIZ_CACHE_FILE, {})
    if "image_question_cache" not in st.session_state:
        st.session_state.image_question_cache = load_json(IMAGE_QUESTION_CACHE_FILE, {})
    if "visual_index" not in st.session_state:
        st.session_state.visual_index = load_json(VISUAL_INDEX_FILE, {})
    if "graph_threads" not in st.session_state:
        st.session_state.graph_threads = {}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_json(CHAT_HISTORY_FILE, {})
    if "last_evaluation" not in st.session_state:
        st.session_state.last_evaluation = None


# ============================================================
# Streamlit App
# ============================================================

def main():
    st.set_page_config(
        page_title="Grade 12 Math AI Tutor",
        page_icon="📘",
        layout="wide",
    )
    init_session()

    profile = st.session_state.profile
    roadmap = st.session_state.roadmap

    st.title("Grade 12 Math AI Tutor")
    st.caption(
        "LangGraph workflow: Teach → Quiz → Grade → Remediation/Practice/Master · "
        "progress & caches saved in JSON · Azure OpenAI only when cache misses"
    )

    # ---- Sidebar ----
    with st.sidebar:
        st.header("Setup")
        cfg_err = azure_config_error()
        if cfg_err:
            st.error(cfg_err)
        else:
            st.success("Azure config ready")
            st.caption(f"Model: {AZURE_MODEL}")

        st.divider()
        st.header("Student")
        name = st.text_input("Name", value=profile.get("student_name", "Tom"))
        if name != profile.get("student_name"):
            profile["student_name"] = name
            save_profile(profile, PROFILE_FILE)

        concept_labels = [
            f"{i + 1}. {item.get('concept', 'Unknown')}" for i, item in enumerate(roadmap)
        ]
        current_idx = int(profile.get("current_index", 0))
        if roadmap:
            current_idx = min(max(current_idx, 0), len(roadmap) - 1)
            selected = st.selectbox(
                "Jump to concept",
                options=list(range(len(roadmap))),
                index=current_idx,
                format_func=lambda i: concept_labels[i],
            )
            if selected != profile.get("current_index"):
                profile["current_index"] = selected
                save_profile(profile, PROFILE_FILE)
                st.session_state.last_evaluation = None
                st.session_state.graph_threads = {}
                st.rerun()

        st.metric("Completed", len(set(profile.get("completed", []))))
        st.metric("Weak topics", len(profile.get("weak", [])))
        if profile.get("weak"):
            st.caption("Needs practice: " + ", ".join(profile["weak"][-5:]))

        st.divider()
        st.write(f"PDF: `{'grade12math.pdf' if PDF_FILE.exists() else 'missing'}`")
        st.write(f"Page images: `{len(list(PAGE_IMAGES_DIR.glob('*.png'))) if PAGE_IMAGES_DIR.exists() else 0}`")
        st.write(f"Roadmap topics: `{len(roadmap)}`")

        if st.button("Reload data from disk"):
            for k in (
                "profile",
                "roadmap",
                "lesson_cache",
                "quiz_cache",
                "image_question_cache",
                "visual_index",
                "chat_history",
                "tutor_graph",
                "tutor_graph_fingerprint",
                "graph_threads",
            ):
                st.session_state.pop(k, None)
            st.rerun()

    if not roadmap:
        st.error(
            f"No course roadmap found at `{ROADMAP_FILE}`. "
            "Add course_roadmap.json from your Colab project."
        )
        return

    item = get_current_item(roadmap, profile)
    if item is None:
        st.success("You finished every concept in the roadmap. Great work!")
        st.json(profile)
        return

    chapter = item.get("chapter", "")
    concept = item.get("concept", "")

    # Refresh visual assets from disk so page links / diagram cache stay current
    st.session_state.visual_index = load_json(VISUAL_INDEX_FILE, {})
    st.session_state.image_question_cache = load_json(IMAGE_QUESTION_CACHE_FILE, {})
    images = concept_images(concept, st.session_state.visual_index)

    st.subheader(f"{chapter}")
    st.write(f"**Current concept:** {concept}")
    progress = (int(profile.get("current_index", 0)) + 1) / max(len(roadmap), 1)
    st.progress(min(progress, 1.0), text=f"Topic {profile.get('current_index', 0) + 1} of {len(roadmap)}")

    tab_agent, tab_lesson, tab_chat, tab_quiz, tab_visual, tab_progress = st.tabs(
        ["AI Agent", "Lesson", "Ask Tutor", "Quiz", "Visual Question", "Progress"]
    )

    # ---- AI Agent (autonomous tutor) ----
    with tab_agent:
        st.markdown("### Autonomous tutor agent")
        st.caption(
            "One chat does everything: the agent decides when to check your progress, "
            "search the textbook, teach, quiz, grade, and save your results. "
            "Try: \"What should I study?\" → \"Teach me\" → \"Quiz me\" → paste your answers."
        )
        if supabase_enabled():
            st.caption("Cloud sync: progress is also saved to Supabase.")

        agent_error = azure_config_error()
        if agent_error:
            st.error(agent_error)
        else:
            if "agent_graph" not in st.session_state:
                with st.spinner("Starting agent..."):
                    st.session_state.agent_graph = build_agent(persistent=True)
                st.session_state.agent_display = []
            if "agent_display" not in st.session_state:
                st.session_state.agent_display = []
            agent_graph = st.session_state.agent_graph

            # One durable conversation thread per student
            agent_thread = {
                "configurable": {
                    "thread_id": f"agent:{profile.get('student_name', 'Student')}"
                }
            }

            # ---- Conversation (always visible) ----
            st.markdown("#### Conversation")
            history = st.session_state.get("agent_display") or []
            if not history:
                st.info(
                    "No messages yet. Type below, or attach a photo of your work, "
                    "then chat with the tutor."
                )
            else:
                for msg in history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                        if msg.get("tools"):
                            with st.expander("Tools used"):
                                for line in msg["tools"]:
                                    st.code(line, language="text")

            # ---- Optional photo attachment (does not replace chat) ----
            with st.expander("Optional: attach a photo of your work", expanded=False):
                st.caption(
                    "Can't type the math? Snap a photo of your handwriting or graph "
                    "and the agent will read and grade it."
                )
                agent_photos = st.file_uploader(
                    "Upload a photo (JPG/PNG, up to 3)",
                    type=["png", "jpg", "jpeg", "webp"],
                    accept_multiple_files=True,
                    key="agent_photo_upload",
                )
                use_agent_cam = st.toggle("Use camera instead", key="agent_cam_toggle")
                agent_cam = None
                if use_agent_cam:
                    agent_cam = st.camera_input("Take a photo", key="agent_cam_input")

                pending_photos = list(agent_photos or [])
                if agent_cam is not None:
                    pending_photos.append(agent_cam)

                send_photo = False
                if pending_photos:
                    cols = st.columns(min(len(pending_photos), 3))
                    for col, photo in zip(cols, pending_photos[:3]):
                        with col:
                            st.image(photo, use_container_width=True)
                    send_photo = st.button(
                        f"Send {len(pending_photos)} photo(s) to the tutor",
                        key="agent_send_photo",
                    )
                else:
                    pending_photos = []
                    send_photo = False

            agent_msg = st.chat_input("Talk to your tutor agent...")

            # A turn happens on a typed message OR a photo-only send.
            user_text = None
            if agent_msg:
                user_text = agent_msg
            elif send_photo:
                user_text = "I uploaded a photo of my work. Please look at it and help me."

            if user_text:
                image_payload = [
                    {"bytes": p.getvalue(), "mime": (p.type or "image/png")}
                    for p in pending_photos[:3]
                ]
                display_text = user_text
                if image_payload:
                    display_text += f"\n\n_[{len(image_payload)} photo(s) attached]_"
                st.session_state.agent_display.append(
                    {"role": "user", "content": display_text}
                )
                with st.chat_message("user"):
                    st.markdown(display_text)

                # Give the agent a text hint plus the raw bytes via the holder,
                # since tool-call arguments cannot carry image data themselves.
                model_text = user_text
                if image_payload:
                    model_text += (
                        f"\n\n[{len(image_payload)} photo(s) attached] "
                        "Use review_student_photo to read them."
                    )
                    set_pending_images(image_payload)

                with st.chat_message("assistant"):
                    with st.spinner("Agent is working (may call several tools)..."):
                        try:
                            result = agent_graph.invoke(
                                {"messages": [{"role": "user", "content": model_text}]},
                                config=agent_thread,
                            )
                        finally:
                            clear_pending_images()
                    messages = result.get("messages") or []
                    tool_lines = []
                    for m in messages:
                        for call in getattr(m, "tool_calls", None) or []:
                            name = call.get("name", "?")
                            args = call.get("args", {})
                            tool_lines.append(
                                f"{name}({json.dumps(args, ensure_ascii=False)[:160]})"
                            )
                    last = messages[-1] if messages else None
                    reply = str(getattr(last, "content", "") or "")
                    st.markdown(reply)
                    if tool_lines:
                        with st.expander("Tools used"):
                            for line in tool_lines[-8:]:
                                st.code(line, language="text")
                st.session_state.agent_display.append(
                    {"role": "assistant", "content": reply, "tools": tool_lines[-8:]}
                )
                # Progress may have changed via update_progress
                st.session_state.profile = load_profile(PROFILE_FILE)
                st.rerun()

            if st.session_state.get("agent_display") and st.button(
                "Reset agent conversation"
            ):
                st.session_state.agent_display = []
                st.session_state.pop("agent_graph", None)
                st.rerun()

    # ---- Lesson ----
    with tab_lesson:
        st.markdown("### Textbook pages for this concept")
        show_page_images(images)
        st.caption("Lesson content is produced by the LangGraph **Teaching** node (cache-first).")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            load_btn = st.button("Load / Generate Lesson", type="primary", use_container_width=True)
        with col_b:
            refresh_lesson = st.button("Regenerate Lesson (uses API)", use_container_width=True)

        if refresh_lesson and concept in st.session_state.lesson_cache:
            del st.session_state.lesson_cache[concept]
            save_json(LESSON_CACHE_FILE, st.session_state.lesson_cache)

        if load_btn or refresh_lesson or concept in st.session_state.lesson_cache:
            with st.spinner("Preparing lesson via LangGraph..."):
                gstate = ensure_quiz_ready(
                    concept,
                    chapter,
                    profile,
                    force_refresh_lesson=bool(refresh_lesson),
                )
            lesson = gstate.get("teaching_content") or ""
            if gstate.get("lesson_from_cache"):
                st.caption("Lesson loaded from cache (no API call).")
            st.markdown("### Tutor lesson")
            st.markdown(lesson)

            practice = st.text_area(
                "Try the practice question here (optional notes)",
                key=f"practice_{concept}",
                height=100,
            )
            if st.button("Ask tutor to check my practice attempt"):
                if not practice.strip():
                    st.warning("Write your attempt first.")
                else:
                    with st.spinner("Checking..."):
                        feedback = call_llm(
                            f"""Student is learning: {concept}
Lesson context:
{lesson}

Student practice attempt:
{practice}

Give brief feedback: what is correct, what to fix, and the worked solution.
""",
                            system="You are a Grade 12 math teacher.",
                        )
                    st.markdown(feedback)

    # ---- Chat ----
    with tab_chat:
        st.markdown("### Conversation with your tutor")
        st.caption("Ask questions about this concept. Chat is saved so you can continue later.")

        chat_key = concept
        chats = st.session_state.chat_history.setdefault(chat_key, [])

        for msg in chats:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_msg = st.chat_input("Ask your tutor anything about this topic...")
        if user_msg:
            chats.append(
                {
                    "role": "user",
                    "content": user_msg,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            )
            with st.chat_message("user"):
                st.markdown(user_msg)
            with st.chat_message("assistant"):
                with st.spinner("Tutor is thinking..."):
                    reply = chat_reply(
                        concept,
                        chapter,
                        profile.get("student_name", "Student"),
                        chats,
                        user_msg,
                    )
                    st.markdown(reply)
            chats.append(
                {
                    "role": "assistant",
                    "content": reply,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            )
            st.session_state.chat_history[chat_key] = chats
            save_json(CHAT_HISTORY_FILE, st.session_state.chat_history)

        if chats and st.button("Clear chat for this concept"):
            st.session_state.chat_history[chat_key] = []
            save_json(CHAT_HISTORY_FILE, st.session_state.chat_history)
            st.rerun()

    # ---- Quiz ----
    with tab_quiz:
        st.markdown("### Concept quiz")
        st.caption(
            "LangGraph: **Quiz** → pause for your answers → **Evaluate** → "
            "Remediation / Practice / Master → save progress to JSON."
        )
        c1, c2 = st.columns(2)
        with c1:
            gen_quiz = st.button("Load / Generate Quiz", type="primary", use_container_width=True)
        with c2:
            regen_quiz = st.button("Regenerate Quiz", use_container_width=True)

        if regen_quiz and concept in st.session_state.quiz_cache:
            del st.session_state.quiz_cache[concept]
            save_json(QUIZ_CACHE_FILE, st.session_state.quiz_cache)

        threads = st.session_state.setdefault("graph_threads", {})
        has_active = concept in threads and is_waiting_for_answer(
            get_tutor_graph(), threads[concept]
        )

        if gen_quiz or regen_quiz or concept in st.session_state.quiz_cache or has_active:
            with st.spinner("Preparing quiz via LangGraph..."):
                gstate = ensure_quiz_ready(
                    concept,
                    chapter,
                    profile,
                    force_refresh_quiz=bool(regen_quiz),
                )
            quiz = gstate.get("quiz") or ""
            if gstate.get("quiz_from_cache"):
                st.caption("Quiz loaded from cache (no API call).")

            display_quiz, answer_part = split_quiz_display(quiz)
            st.markdown(display_quiz)
            with st.expander("Show answer key (teacher view)"):
                st.markdown(answer_part or "_Answer key is included in the generated quiz text._")

            student_answer = st.text_area(
                "Write your quiz answers here",
                height=180,
                key=f"quiz_ans_{concept}",
            )
            if st.button("Submit quiz for grading", type="primary"):
                if not student_answer.strip():
                    st.warning("Enter your answers first.")
                else:
                    graph = get_tutor_graph()
                    tid = threads.get(concept)
                    if not tid or not is_waiting_for_answer(graph, tid):
                        st.warning("Quiz session expired. Click Load / Generate Quiz again.")
                    else:
                        with st.spinner("Grading via LangGraph (Evaluate → branch → Save)..."):
                            result = submit_quiz_answer(graph, student_answer, tid)
                        threads.pop(concept, None)
                        evaluation = result.get("evaluation") or {}
                        st.session_state.last_evaluation = evaluation
                        if result.get("profile"):
                            profile = result["profile"]
                            st.session_state.profile = profile
                        refresh_caches_from_disk()
                        st.session_state.profile = profile

                        st.success(f"Mastery score: {evaluation.get('mastery_score', '?')}")
                        st.markdown(evaluation.get("feedback", ""))
                        st.json(
                            {
                                "strengths": evaluation.get("strengths"),
                                "weaknesses": evaluation.get("weaknesses"),
                                "recommended_practice": evaluation.get("recommended_practice"),
                                "next_action": evaluation.get("next_action"),
                            }
                        )

                        if result.get("coach_message"):
                            st.info(result["coach_message"])
                        if result.get("branch_content"):
                            st.markdown("### Follow-up (Remediation / Practice)")
                            st.markdown(result["branch_content"])

                        if evaluation.get("next_action") == "mastered" or int(
                            evaluation.get("mastery_score", 0) or 0
                        ) >= 85:
                            st.balloons()
                            st.info(
                                "Marked mastered and progress advanced. "
                                "Open Lesson for the next topic."
                            )
                        else:
                            st.info(
                                "Progress saved. Load the quiz again when you are ready to retry "
                                "(lesson/quiz stay cached — no regenerate needed)."
                            )
    # ---- Visual ----
    with tab_visual:
        st.markdown("### How visual learning works")
        st.info(
            "Curve / graph questions use **two visuals**:\n"
            "1. **Textbook page scans** from `grade12math.pdf` (real figures in the book)\n"
            "2. **Auto-drawn diagrams** (coordinate plots the tutor generates for the question)\n\n"
            "Use this tab for graph questions. You do **not** need to draw by hand — "
            "read the diagram, then type your answer."
        )

        st.markdown("#### Textbook pages for this concept")
        if not images:
            st.warning(
                "No pages were linked yet. Pick pages below and click **Save page links**."
            )
        else:
            show_page_images(images, caption_prefix="Figure page")

        all_pages = list_all_page_images()
        with st.expander("Link / change textbook pages for this concept", expanded=not bool(images)):
            selected_pages = st.multiselect(
                "Choose page images that contain graphs or diagrams for this topic",
                options=all_pages,
                default=images,
                key=f"page_pick_{concept}",
            )
            if st.button("Save page links", key=f"save_pages_{concept}"):
                st.session_state.visual_index[concept] = selected_pages
                save_visual_index(st.session_state.visual_index)
                st.success("Saved. Reloading…")
                st.rerun()

            preview_page = st.selectbox(
                "Preview any textbook page",
                options=["(none)"] + all_pages,
                key=f"preview_page_{concept}",
            )
            if preview_page != "(none)":
                st.image(
                    str(BASE_DIR / preview_page),
                    caption=preview_page,
                    use_container_width=True,
                )

        images = concept_images(concept, st.session_state.visual_index)

        st.markdown("#### Visual quiz question")
        v1, v2 = st.columns(2)
        with v1:
            gen_vis = st.button(
                "Create question from page image",
                type="primary",
                use_container_width=True,
            )
        with v2:
            regen_vis = st.button("Regenerate visual question", use_container_width=True)

        if gen_vis or regen_vis or concept in st.session_state.image_question_cache:
            with st.spinner("Preparing visual question + diagram..."):
                img_q = get_image_question(
                    concept,
                    images,
                    st.session_state.image_question_cache,
                    force_refresh=regen_vis or gen_vis,
                )
                img_q = ensure_diagram(img_q)
                # Persist inferred diagram for next open
                st.session_state.image_question_cache[concept] = img_q
                save_json(IMAGE_QUESTION_CACHE_FILE, st.session_state.image_question_cache)

            st.markdown("##### Diagram (auto-drawn for this question)")
            diagram = img_q.get("diagram")
            if diagram and (diagram.get("type") or "").lower() not in ("", "none"):
                render_diagram(diagram)
            else:
                st.caption("No auto-diagram for this item — use the textbook page figure above.")

            st.markdown("##### Question")
            st.markdown(img_q.get("question", ""))
            if img_q.get("page_reference"):
                st.caption(f"Based on: {img_q['page_reference']}")
            if img_q.get("skills_tested"):
                st.write("Skills:", ", ".join(img_q["skills_tested"]))

            with st.expander("Answer key (after you try)"):
                st.markdown(img_q.get("answer_key") or "_No answer key yet._")

            vis_ans = st.text_area(
                "Your answer to the visual question (you can also upload a photo below)",
                height=140,
                key=f"vis_ans_{concept}",
            )

            st.markdown("##### Or upload a photo of your handwritten work / sketch")
            uploaded_photos = st.file_uploader(
                "Take a picture of your work on paper and upload it (JPG/PNG, up to 3)",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True,
                key=f"vis_photo_{concept}",
            )
            use_camera = st.toggle("Use camera instead", key=f"vis_cam_toggle_{concept}")
            camera_photo = None
            if use_camera:
                camera_photo = st.camera_input(
                    "Take a photo of your work",
                    key=f"vis_cam_{concept}",
                )

            photos = list(uploaded_photos or [])
            if camera_photo is not None:
                photos.append(camera_photo)
            if photos:
                cols = st.columns(min(len(photos), 3))
                for i, ph in enumerate(photos[:3]):
                    with cols[i % len(cols)]:
                        st.image(ph, caption=f"Your work {i + 1}", use_container_width=True)

            if st.button("Check visual answer"):
                if not vis_ans.strip() and not photos:
                    st.warning("Write your answer or upload a photo of your work first.")
                else:
                    grade_prompt = f"""Concept: {concept}
Visual question: {img_q.get('question')}
Answer key: {img_q.get('answer_key')}
Student typed answer (may be empty if they uploaded a photo): {vis_ans}

Give supportive feedback and a short worked solution. Score out of 10.
"""
                    with st.spinner("Checking..."):
                        if photos:
                            check = call_llm_with_uploaded_images(
                                grade_prompt
                                + "\nThe student also attached photo(s) of their handwritten "
                                "work or graph sketch. Read the handwriting/sketch carefully, "
                                "grade based on BOTH the photo and any typed answer, and point "
                                "out anything in the sketch that is right or wrong.",
                                photos,
                                system="You are a Grade 12 math teacher who reads handwritten "
                                "student work carefully.",
                            )
                        else:
                            check = call_llm(
                                grade_prompt,
                                system="You are a Grade 12 math teacher.",
                            )
                    st.markdown(check)

        st.divider()
        st.markdown("#### Graph sketch lab (optional)")
        st.caption(
            "If a quiz says “sketch the graph”, use this to see the curve yourself — "
            "then describe what you notice in your answer."
        )
        gcol1, gcol2 = st.columns([2, 1])
        with gcol1:
            fn_expr = st.text_input(
                "Function y = … (use ** for powers, e.g. -x + 3 or x**2 - 4)",
                value="-x + 3",
                key=f"graph_lab_fn_{concept}",
            )
        with gcol2:
            xmax = st.number_input("x max", value=6.0, key=f"graph_lab_xmax_{concept}")
        if st.button("Draw this graph", key=f"draw_graph_{concept}"):
            render_diagram(
                {
                    "type": "function",
                    "function": fn_expr,
                    "points": [],
                    "labels": [],
                    "show_line": False,
                    "x_range": [-float(xmax), float(xmax)],
                    "y_range": [-float(xmax), float(xmax)],
                    "title": f"y = {fn_expr}",
                }
            )

    # ---- Progress ----
    with tab_progress:
        st.markdown("### Saved learning record")
        st.write(
            "Progress is stored in `student_profile.json`, chat in `chat_history.json`, "
            "and generated lessons/quizzes in cache JSON files. "
            "The Lesson/Quiz path is driven by LangGraph (`tutor_graph.py`)."
        )
        st.json(profile)

        hist = profile.get("history") or []
        if hist:
            st.markdown("#### Recent scores")
            for h in hist[-10:][::-1]:
                st.write(
                    f"- **{h.get('concept')}** — score {h.get('score')} "
                    f"({h.get('timestamp', '')}) → {h.get('next_action', '')}"
                )

        if st.button("Mark current concept complete & go next"):
            if concept not in profile.setdefault("completed", []):
                profile["completed"].append(concept)
            profile["current_index"] = int(profile.get("current_index", 0)) + 1
            save_profile(profile, PROFILE_FILE)
            st.session_state.profile = profile
            st.rerun()


if __name__ == "__main__":
    main()
