"""
Tutor tools for a ReAct-style Grade 12 math agent.

Step 1 of the agent upgrade: wrap tutoring actions as LangChain `@tool`
callables so an LLM agent can decide which skill to use.

Design notes:
- Lesson and quiz tools are cache-first (same JSON files as tutor_graph.py)
  to avoid burning API tokens on repeated content.
- Progress tools are pure Python (no LLM call).
- Tools that need an LLM receive it through `build_tutor_tools(llm_fn)`.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from image_attachments import get_pending_images
from profile_store import load_profile, save_profile
from textbook_index import search_chunks
from tutor_graph import (
    BASE_DIR,
    clean_json,
    is_llm_error,
    load_json,
    save_json,
)

LLMFn = Callable[..., str]

# Vision callable: (prompt, images, system=None) -> str, where images is a
# list of {"bytes": <raw image bytes>, "mime": "image/png"} dicts.
VisionFn = Callable[..., str]


class EvaluationResult(BaseModel):
    """Structured grading result (replaces fragile JSON string parsing)."""

    mastery_score: int = Field(description="Score from 0 to 100", ge=0, le=100)
    feedback: str = Field(description="Detailed feedback for the student")
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    recommended_practice: list[str] = Field(default_factory=list)
    next_action: str = Field(
        description="One of: mastered (>=85), practice (60-84), remediation (<60)"
    )


def build_tutor_tools(
    llm_fn: LLMFn,
    base_dir: Path | None = None,
    chat_model=None,
    vision_fn: VisionFn | None = None,
) -> list:
    """
    Build the tool list bound to a concrete LLM callable.

    Args:
        llm_fn: Callable like ``llm_fn(prompt, system=None) -> str``.
        base_dir: Project root (defaults to this package directory).
        chat_model: Optional LangChain chat model. When given, grading uses
            ``with_structured_output(EvaluationResult)`` for reliable JSON.
        vision_fn: Optional callable ``vision_fn(prompt, images, system=None)``
            that sends the pending photos to a vision model. When provided, the
            ``review_student_photo`` tool is enabled.
    """
    root = Path(base_dir) if base_dir else BASE_DIR
    lesson_cache_file = root / "lesson_cache.json"
    quiz_cache_file = root / "quiz_cache.json"
    profile_file = root / "student_profile.json"
    roadmap_file = root / "course_roadmap.json"

    @tool
    def get_current_concept() -> str:
        """Return the student's current roadmap concept and chapter from their profile."""
        profile = load_profile(profile_file)
        roadmap = load_json(roadmap_file, [])
        if not roadmap:
            return "No course roadmap is available."
        idx = int(profile.get("current_index", 0))
        if idx < 0 or idx >= len(roadmap):
            return "The student has finished every concept in the roadmap."
        item = roadmap[idx]
        return json.dumps(
            {
                "index": idx,
                "chapter": item.get("chapter", ""),
                "concept": item.get("concept", ""),
                "student_name": profile.get("student_name", "Student"),
                "completed_count": len(set(profile.get("completed", []))),
                "weak": profile.get("weak", []),
            },
            ensure_ascii=False,
        )

    @tool
    def list_roadmap_concepts(limit: int = 10) -> str:
        """List upcoming concepts from the course roadmap (default: next 10)."""
        profile = load_profile(profile_file)
        roadmap = load_json(roadmap_file, [])
        start = int(profile.get("current_index", 0))
        slice_ = roadmap[start : start + max(1, min(limit, 20))]
        lines = [
            f"{start + i}. [{item.get('chapter', '')}] {item.get('concept', '')}"
            for i, item in enumerate(slice_)
        ]
        return "\n".join(lines) if lines else "No remaining concepts."

    @tool
    def search_textbook(query: str, k: int = 3) -> str:
        """
        Search the Grade 12 textbook (grade12math.pdf) for passages matching
        a query. Free keyword search — costs no API tokens. Use this to
        ground lessons and examples in the real textbook wording.
        """
        results = search_chunks(query, k=max(1, min(k, 5)))
        if not results:
            return "No matching textbook passages found."
        parts = [
            f"[page {r['page']}] {r['text'][:600]}"
            for r in results
        ]
        return "\n\n".join(parts)

    @tool
    def teach_concept(concept: str, chapter: str = "", textbook_context: str = "") -> str:
        """
        Teach a Grade 12 math concept with a short lesson.
        Uses the lesson cache when available (no extra API call).
        Pass textbook_context (from search_textbook) to ground the lesson
        in the real textbook; cached lessons ignore the context.
        """
        concept = (concept or "").strip()
        if not concept:
            return "Please provide a concept name."

        cache = load_json(lesson_cache_file, {})
        if concept in cache:
            return f"[from cache]\n{cache[concept]}"

        context_block = ""
        if textbook_context.strip():
            context_block = f"""
Textbook excerpts to ground your lesson (quote or adapt where useful):
{textbook_context.strip()[:2500]}
"""

        prompt = f"""
You are a friendly Grade 12 math tutor.
Teach this concept from chapter "{chapter or "Functions and Models"}":

{concept}
{context_block}
Include:
1. Simple explanation
2. Real-world example
3. Common mistakes
4. One practice question (do NOT give the answer yet)

Keep it focused and student-friendly.
"""
        lesson = llm_fn(prompt, system="You are an expert Grade 12 mathematics teacher.")
        if not is_llm_error(lesson):
            cache[concept] = lesson
            save_json(lesson_cache_file, cache)
        return lesson

    @tool
    def make_quiz(concept: str) -> str:
        """
        Create a Grade 12 math quiz for a concept.
        Uses the quiz cache when available (no extra API call).
        """
        concept = (concept or "").strip()
        if not concept:
            return "Please provide a concept name."

        cache = load_json(quiz_cache_file, {})
        if concept in cache:
            return f"[from cache]\n{cache[concept]}"

        prompt = f"""
Create a Grade 12 math quiz for this concept:

{concept}

Include:
1. Two multiple choice questions
2. One true/false question
3. One application question
4. One short explanation question
5. Answer key at the end (clearly labeled)

Suitable for a first-time learner of this topic.
"""
        quiz = llm_fn(prompt, system="You are an expert Grade 12 mathematics teacher.")
        if not is_llm_error(quiz):
            cache[concept] = quiz
            save_json(quiz_cache_file, cache)
        return quiz

    @tool
    def grade_answer(concept: str, quiz: str, student_answer: str) -> str:
        """
        Grade a student's quiz answer for a concept.
        Returns JSON with mastery_score, feedback, and next_action.
        """
        concept = (concept or "").strip()
        student_answer = (student_answer or "").strip()
        if not concept or not student_answer:
            return "Need both concept and student_answer to grade."

        prompt = f"""
You are a Grade 12 mathematics teacher.
Evaluate the student's answer carefully.

Concept:
{concept}

Quiz:
{quiz}

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
        # Preferred path: structured output guarantees a valid schema
        if chat_model is not None:
            try:
                structured = chat_model.with_structured_output(EvaluationResult)
                evaluation = structured.invoke(prompt)
                return evaluation.model_dump_json(indent=2)
            except Exception:
                pass  # fall through to the plain-text path

        result = llm_fn(prompt, system="You are a careful Grade 12 math grader.")
        try:
            evaluation = EvaluationResult(**clean_json(result))
            return evaluation.model_dump_json(indent=2)
        except Exception:
            return EvaluationResult(
                mastery_score=0,
                feedback=result,
                next_action="remediation",
            ).model_dump_json(indent=2)

    @tool
    def update_progress(concept: str, mastery_score: int, next_action: str) -> str:
        """
        Save tutoring progress (JSON file + Supabase when configured).
        Advances current_index when mastered (score >= 85 or next_action=mastered).
        Pure Python — does not call the LLM.
        """
        profile = load_profile(profile_file)
        score = int(mastery_score or 0)
        action = (next_action or "").lower().strip()

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
            message = (
                f"Mastered '{concept}'. "
                f"Progress advanced to index {profile['current_index']}."
            )
        else:
            if concept not in profile.setdefault("weak", []):
                profile["weak"].append(concept)
            message = f"Recorded score {score} for '{concept}' as weak / needs practice."

        save_profile(profile, profile_file)
        return message

    @tool
    def review_student_photo(concept: str = "", quiz: str = "") -> str:
        """
        Read the student's most recently attached photo of handwritten math work
        or a graph/diagram sketch. Use this whenever the student uploaded or took
        a photo instead of typing their answer.

        Behavior:
        - If `quiz` text is given, grade the work in the photo against that quiz
          and return JSON with mastery_score, feedback, and next_action (same
          schema as grade_answer). Afterwards, call update_progress.
        - If `quiz` is empty, return a plain-language transcription/summary of
          what the student wrote or drew, so you can decide what to do next.
        """
        images = get_pending_images()
        if not images:
            return (
                "No photo is attached. Ask the student to upload or take a photo "
                "of their work, then try again."
            )
        if vision_fn is None:
            return "Vision is not available in this configuration."

        if quiz.strip():
            prompt = f"""
You are a Grade 12 mathematics teacher grading a photo of a student's work.
First read the handwriting / diagram in the image(s) carefully, then grade it
against the quiz below.

Concept:
{concept}

Quiz:
{quiz}

Return JSON only (no markdown):
{{
  "mastery_score": 85,
  "feedback": "What you saw in the photo and detailed feedback.",
  "strengths": ["..."],
  "weaknesses": ["..."],
  "recommended_practice": ["..."],
  "next_action": "mastered/practice/remediation"
}}

Scoring: 85-100 mastered, 60-84 practice, below 60 remediation.
"""
            raw = vision_fn(
                prompt,
                images,
                system="You are a careful Grade 12 math grader reading photos of student work.",
            )
            try:
                evaluation = EvaluationResult(**clean_json(raw))
                return evaluation.model_dump_json(indent=2)
            except Exception:
                return EvaluationResult(
                    mastery_score=0,
                    feedback=raw,
                    next_action="remediation",
                ).model_dump_json(indent=2)

        prompt = (
            "Transcribe and summarize the student's handwritten math work or "
            "sketch in the image(s). List each step or equation you can read, "
            "and describe any diagram/graph. Do not grade yet; just report what "
            "is on the paper as clearly as possible."
        )
        return vision_fn(
            prompt,
            images,
            system="You read photos of Grade 12 math work and transcribe them faithfully.",
        )

    tools = [
        get_current_concept,
        list_roadmap_concepts,
        search_textbook,
        teach_concept,
        make_quiz,
        grade_answer,
        update_progress,
    ]
    if vision_fn is not None:
        tools.append(review_student_photo)
    return tools
