"""
ReAct tutor agent (Steps 2-4 of the agent upgrade).

The agent uses LangGraph's ``create_react_agent`` so the LLM decides
which tool to call (teach, quiz, grade, textbook search, progress)
instead of following a fixed Teaching → Quiz → Evaluate pipeline.

Persistence: a SQLite checkpointer stores conversation state per
thread_id, so a quiz started in one turn can be graded in a later turn
(and survives app restarts).
"""

from __future__ import annotations

import base64
import os
import sqlite3
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from openai import OpenAI

from tutor_tools import build_tutor_tools

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
CHECKPOINT_DB = BASE_DIR / "tutor_agent_memory.sqlite"

SYSTEM_PROMPT = """You are a Grade 12 mathematics tutor agent.

You have tools for:
- looking up the student's current concept
- listing roadmap topics
- searching the textbook (free, no tokens)
- teaching a concept (cache-first)
- making a quiz (cache-first)
- grading an answer
- updating student progress
- reading a photo of the student's handwritten work or sketch (review_student_photo)

Guidelines:
1. Prefer tools over guessing. If the student asks what they should study,
   call get_current_concept first.
2. When the student asks to learn something: first call search_textbook with
   the concept name, then call teach_concept passing the excerpts as
   textbook_context. If search returns nothing, teach without context.
3. When they ask for a quiz/test, call make_quiz (use the concept name).
4. When the student sends quiz answers, call grade_answer with the quiz text
   from earlier in the conversation and their answer.
5. After grade_answer succeeds, ALWAYS call update_progress with the score
   and next_action so their profile stays current. Then, based on next_action:
   - mastered: congratulate and suggest the next concept
   - practice: give focused follow-up practice on their weak points
   - remediation: re-teach the concept more simply
6. When the student attaches a photo (you will see a note like
   "[photo attached]"), call review_student_photo. If the photo answers a quiz
   from earlier in the conversation, pass that quiz text so it is graded, then
   call update_progress. Otherwise call it with no quiz to read the work first.
7. Keep replies short, encouraging, and focused on Grade 12 math.
8. Do not invent scores. Use grade_answer or review_student_photo for scoring.
"""


def load_azure_settings() -> dict[str, str]:
    """Load Azure OpenAI settings from .env / process environment."""
    load_dotenv(ENV_FILE, override=True)
    return {
        "api_key": (os.getenv("AZURE_OPENAI_API_KEY") or "").strip(),
        "endpoint": (os.getenv("AZURE_ENDPOINT") or "").strip(),
        "model": (os.getenv("AZURE_MODEL") or "").strip(),
    }


def make_openai_llm_fn(settings: dict[str, str] | None = None):
    """
    Build a simple ``(prompt, system=None) -> str`` callable for tools
    that generate lesson / quiz / grade content via the OpenAI SDK.
    """
    cfg = settings or load_azure_settings()
    if not cfg["api_key"] or not cfg["endpoint"] or not cfg["model"]:
        raise RuntimeError(
            "Missing Azure settings. Set AZURE_OPENAI_API_KEY, "
            "AZURE_ENDPOINT, and AZURE_MODEL in .env"
        )

    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["endpoint"])
    model = cfg["model"]

    def llm_fn(prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(model=model, messages=messages)
        return response.choices[0].message.content or ""

    return llm_fn


def make_openai_vision_fn(settings: dict[str, str] | None = None):
    """
    Build a ``(prompt, images, system=None) -> str`` callable that sends photos
    to the vision-capable chat model.

    ``images`` is a list of ``{"bytes": b"...", "mime": "image/png"}`` dicts.
    """
    cfg = settings or load_azure_settings()
    if not cfg["api_key"] or not cfg["endpoint"] or not cfg["model"]:
        raise RuntimeError(
            "Missing Azure settings. Set AZURE_OPENAI_API_KEY, "
            "AZURE_ENDPOINT, and AZURE_MODEL in .env"
        )

    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["endpoint"])
    model = cfg["model"]

    def vision_fn(prompt: str, images: list[dict], system: str | None = None) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in (images or [])[:3]:
            raw = img.get("bytes")
            if not raw:
                continue
            mime = img.get("mime") or "image/png"
            b64 = base64.b64encode(raw).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        response = client.chat.completions.create(model=model, messages=messages)
        return response.choices[0].message.content or ""

    return vision_fn


def make_chat_model(settings: dict[str, str] | None = None) -> ChatOpenAI:
    """Chat model used by the ReAct agent for tool selection / replies."""
    cfg = settings or load_azure_settings()
    if not cfg["api_key"] or not cfg["endpoint"] or not cfg["model"]:
        raise RuntimeError(
            "Missing Azure settings. Set AZURE_OPENAI_API_KEY, "
            "AZURE_ENDPOINT, and AZURE_MODEL in .env"
        )
    return ChatOpenAI(
        model=cfg["model"],
        api_key=cfg["api_key"],
        base_url=cfg["endpoint"],
    )


def build_tutor_agent(base_dir: Path | None = None, persistent: bool = False):
    """
    Compile a ReAct agent with tutoring tools.

    Args:
        base_dir: Project root for cache/profile files.
        persistent: When True, conversation state is checkpointed to
            SQLite so multi-turn quiz flows survive reruns/restarts.
            Requires passing ``config={"configurable": {"thread_id": ...}}``
            on every invoke.

    Returns:
        A compiled LangGraph agent that accepts messages like::

            {"messages": [{"role": "user", "content": "..."}]}
    """
    root = Path(base_dir) if base_dir else BASE_DIR
    settings = load_azure_settings()
    llm_fn = make_openai_llm_fn(settings)
    vision_fn = make_openai_vision_fn(settings)
    model = make_chat_model(settings)
    tools = build_tutor_tools(
        llm_fn, base_dir=root, chat_model=model, vision_fn=vision_fn
    )

    checkpointer = None
    if persistent:
        # check_same_thread=False lets Streamlit's rerun threads share it
        conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
        checkpointer = SqliteSaver(conn)

    return create_react_agent(
        model,
        tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )


def run_agent_once(user_text: str, agent=None) -> str:
    """
    Send one user message to the agent and return the final assistant text.
    Useful for scripts and smoke tests.
    """
    graph = agent or build_tutor_agent()
    result: dict[str, Any] = graph.invoke(
        {"messages": [{"role": "user", "content": user_text}]}
    )
    messages = result.get("messages") or []
    if not messages:
        return ""
    last = messages[-1]
    content = getattr(last, "content", None)
    if content is None and isinstance(last, dict):
        content = last.get("content", "")
    return str(content or "")
