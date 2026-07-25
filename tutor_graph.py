"""
LangGraph tutoring workflow for the Grade 12 Math AI Tutor.

Pipeline:
  Teaching → Quiz → AwaitAnswer (human interrupt) → Evaluate
      → Remediation | Practice | Coach → SaveProgress → END

Lesson/quiz nodes are cache-first (JSON) to avoid unnecessary API calls.
Score routing and progress updates are pure Python (no LLM).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

LLMFn = Callable[..., str]

BASE_DIR = Path(__file__).resolve().parent
PROFILE_FILE = BASE_DIR / "student_profile.json"
LESSON_CACHE_FILE = BASE_DIR / "lesson_cache.json"
QUIZ_CACHE_FILE = BASE_DIR / "quiz_cache.json"
IMAGE_QUESTION_CACHE_FILE = BASE_DIR / "image_question_cache.json"


class TutorState(TypedDict, total=False):
    student_name: str
    concept: str
    chapter: str
    teaching_content: str
    quiz: str
    image_question: dict
    student_answer: str
    mastery_score: int
    evaluation: dict
    branch_content: str
    coach_message: str
    next_action: str
    profile: dict
    force_refresh_lesson: bool
    force_refresh_quiz: bool
    lesson_from_cache: bool
    quiz_from_cache: bool


# ------------------------------------------------------------
# JSON helpers
# ------------------------------------------------------------

def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def clean_json(text: str) -> Any:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


def is_llm_error(text: str) -> bool:
    t = (text or "").strip()
    return t.startswith("AZURE_") or "is not set" in t or "is missing in" in t


# ------------------------------------------------------------
# Graph builder
# ------------------------------------------------------------

def build_tutor_graph(llm_fn: LLMFn, base_dir: Path | None = None):
    """Compile the tutoring StateGraph with an in-memory checkpointer."""
    root = Path(base_dir) if base_dir else BASE_DIR
    profile_file = root / "student_profile.json"
    lesson_cache_file = root / "lesson_cache.json"
    quiz_cache_file = root / "quiz_cache.json"
    image_q_cache_file = root / "image_question_cache.json"

    def teaching_node(state: TutorState) -> dict:
        concept = state["concept"]
        chapter = state.get("chapter", "")
        cache = load_json(lesson_cache_file, {})
        force = bool(state.get("force_refresh_lesson"))

        if force and concept in cache:
            del cache[concept]
            save_json(lesson_cache_file, cache)

        if concept in cache and not force:
            return {
                "teaching_content": cache[concept],
                "lesson_from_cache": True,
            }

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
        lesson = llm_fn(prompt, system="You are an expert Grade 12 mathematics teacher.")
        if not is_llm_error(lesson):
            cache[concept] = lesson
            save_json(lesson_cache_file, cache)
        return {
            "teaching_content": lesson,
            "lesson_from_cache": False,
        }

    def quiz_node(state: TutorState) -> dict:
        concept = state["concept"]
        cache = load_json(quiz_cache_file, {})
        force = bool(state.get("force_refresh_quiz"))

        if force and concept in cache:
            del cache[concept]
            save_json(quiz_cache_file, cache)

        if concept in cache and not force:
            return {
                "quiz": cache[concept],
                "quiz_from_cache": True,
                "image_question": load_json(image_q_cache_file, {}).get(concept, {}),
            }

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
        quiz = llm_fn(prompt, system="You are an expert Grade 12 mathematics teacher.")
        if not is_llm_error(quiz):
            cache[concept] = quiz
            save_json(quiz_cache_file, cache)
        return {
            "quiz": quiz,
            "quiz_from_cache": False,
            "image_question": load_json(image_q_cache_file, {}).get(concept, {}),
        }

    def await_answer_node(state: TutorState) -> dict:
        """Pause for the student to submit quiz answers (Streamlit resumes)."""
        payload = {
            "type": "await_quiz_answer",
            "concept": state.get("concept"),
            "quiz": state.get("quiz", ""),
        }
        answer = interrupt(payload)
        return {"student_answer": str(answer or "").strip()}

    def evaluate_node(state: TutorState) -> dict:
        concept = state["concept"]
        quiz = state.get("quiz", "")
        image_question = state.get("image_question") or {}
        student_answer = state.get("student_answer", "")

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
        result = llm_fn(prompt, system="You are a careful Grade 12 math grader.")
        try:
            evaluation = clean_json(result)
        except Exception:
            evaluation = {
                "mastery_score": 0,
                "feedback": result,
                "strengths": [],
                "weaknesses": [],
                "recommended_practice": [],
                "next_action": "remediation",
            }

        score = int(evaluation.get("mastery_score", 0) or 0)
        action = (evaluation.get("next_action") or "").lower().strip()
        if not action:
            if score >= 85:
                action = "mastered"
            elif score >= 60:
                action = "practice"
            else:
                action = "remediation"
            evaluation["next_action"] = action

        return {
            "evaluation": evaluation,
            "mastery_score": score,
            "next_action": action,
        }

    def route_after_evaluation(state: TutorState) -> str:
        score = int(state.get("mastery_score", 0) or 0)
        action = (state.get("next_action") or "").lower()
        if action == "mastered" or score >= 85:
            return "coach"
        if action == "practice" or score >= 60:
            return "practice"
        return "remediation"

    def remediation_node(state: TutorState) -> dict:
        concept = state["concept"]
        feedback = (state.get("evaluation") or {}).get("feedback", "")
        weaknesses = (state.get("evaluation") or {}).get("weaknesses", [])
        prompt = f"""
The student struggled with: {concept}
Mastery score: {state.get("mastery_score", 0)}
Feedback: {feedback}
Weaknesses: {weaknesses}

Reteach the concept more simply.
Include:
1. A gentler explanation
2. One very easy worked example
3. One easy practice question (no answer yet)
Keep it short and encouraging.
"""
        content = llm_fn(prompt, system="You are a patient Grade 12 math remediation tutor.")
        return {
            "branch_content": content,
            "coach_message": "",
            "next_action": "remediation",
        }

    def practice_node(state: TutorState) -> dict:
        concept = state["concept"]
        weaknesses = (state.get("evaluation") or {}).get("weaknesses", [])
        recommended = (state.get("evaluation") or {}).get("recommended_practice", [])
        prompt = f"""
The student partially understands: {concept}
Score: {state.get("mastery_score", 0)}
Weak points: {weaknesses}
Recommended practice: {recommended}

Give focused follow-up practice:
1. Brief tip on the weak points
2. Two practice questions with short hints (not full answers)
Keep it concise.
"""
        content = llm_fn(prompt, system="You are a Grade 12 math practice coach.")
        return {
            "branch_content": content,
            "coach_message": "",
            "next_action": "practice",
        }

    def coach_node(state: TutorState) -> dict:
        # Template only — no API call (saves tokens on mastery path)
        name = state.get("student_name") or "Student"
        concept = state.get("concept", "this topic")
        score = state.get("mastery_score", 0)
        msg = (
            f"Nice work, {name}! You mastered **{concept}** "
            f"(score {score}). Moving you to the next topic."
        )
        return {
            "coach_message": msg,
            "branch_content": "",
            "next_action": "mastered",
        }

    def save_progress_node(state: TutorState) -> dict:
        concept = state["concept"]
        evaluation = state.get("evaluation") or {}
        profile = dict(state.get("profile") or load_json(profile_file, {}))
        score = int(evaluation.get("mastery_score", state.get("mastery_score", 0)) or 0)
        action = (state.get("next_action") or evaluation.get("next_action") or "").lower()

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

        save_json(profile_file, profile)
        return {"profile": profile}

    graph = StateGraph(TutorState)
    graph.add_node("teaching", teaching_node)
    graph.add_node("quiz", quiz_node)
    graph.add_node("await_answer", await_answer_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("remediation", remediation_node)
    graph.add_node("practice", practice_node)
    graph.add_node("coach", coach_node)
    graph.add_node("save_progress", save_progress_node)

    graph.add_edge(START, "teaching")
    graph.add_edge("teaching", "quiz")
    graph.add_edge("quiz", "await_answer")
    graph.add_edge("await_answer", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {
            "remediation": "remediation",
            "practice": "practice",
            "coach": "coach",
        },
    )
    graph.add_edge("remediation", "save_progress")
    graph.add_edge("practice", "save_progress")
    graph.add_edge("coach", "save_progress")
    graph.add_edge("save_progress", END)

    return graph.compile(checkpointer=MemorySaver())


# ------------------------------------------------------------
# Streamlit / caller helpers
# ------------------------------------------------------------

def make_thread_id(student_name: str, concept: str) -> str:
    safe_name = (student_name or "student").replace(" ", "_")
    safe_concept = (concept or "concept").replace(" ", "_")[:40]
    return f"{safe_name}:{safe_concept}:{uuid.uuid4().hex[:8]}"


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def is_waiting_for_answer(graph, thread_id: str) -> bool:
    snap = graph.get_state(thread_config(thread_id))
    return bool(snap.next)


def get_graph_values(graph, thread_id: str) -> dict:
    snap = graph.get_state(thread_config(thread_id))
    return dict(snap.values or {})


def start_until_quiz(graph, inputs: dict, thread_id: str) -> dict:
    """Run Teaching → Quiz, then pause at AwaitAnswer."""
    return graph.invoke(inputs, config=thread_config(thread_id))


def submit_quiz_answer(graph, answer: str, thread_id: str) -> dict:
    """Resume after interrupt with the student's quiz answer."""
    return graph.invoke(Command(resume=answer), config=thread_config(thread_id))


def split_quiz_display(quiz: str) -> tuple[str, str]:
    """Hide answer key from the student view when possible."""
    display_quiz = quiz or ""
    answer_part = ""
    for marker in ("Answer Key", "Answers:", "**Answer"):
        if marker in display_quiz:
            parts = display_quiz.split(marker, 1)
            display_quiz = parts[0].rstrip()
            answer_part = marker + parts[1]
            break
    return display_quiz, answer_part
