"""
End-to-end local test for the fully automated agent (Steps 3-6).

Simulates a complete tutoring session in one persistent thread:
1. Ask what to study            -> get_current_concept
2. Ask to learn it              -> search_textbook + teach_concept
3. Ask for a quiz               -> make_quiz
4. Send strong answers          -> grade_answer + update_progress

Run:
    python test_agent_e2e.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from profile_store import load_profile
from tutor_agent import build_tutor_agent

BASE_DIR = Path(__file__).resolve().parent

THREAD = {"configurable": {"thread_id": "e2e-test-thread"}}


def safe_print(text: str) -> None:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(str(text).encode(encoding, errors="replace").decode(encoding, errors="replace"))


def show_turn(result: dict, label: str) -> str:
    messages = result.get("messages") or []
    tools = []
    for m in messages:
        for call in getattr(m, "tool_calls", None) or []:
            tools.append(call.get("name", "?"))
    last = messages[-1] if messages else None
    reply = str(getattr(last, "content", "") or "")
    safe_print(f"\n=== {label} ===")
    safe_print(f"tools called (whole thread so far): {tools}")
    safe_print(f"agent reply (first 400 chars):\n{reply[:400]}")
    return reply


def main() -> int:
    agent = build_tutor_agent(persistent=True)

    before = load_profile()
    safe_print(
        f"profile before: index={before.get('current_index')} "
        f"history={len(before.get('history', []))}"
    )

    def ask(text: str, label: str) -> str:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": text}]},
            config=THREAD,
        )
        return show_turn(result, label)

    ask("What concept should I study next?", "1. progress check")
    ask("Teach me this concept using the textbook.", "2. teach with textbook")
    quiz_reply = ask("Give me a quiz on it.", "3. quiz")

    # A deliberately generic strong answer; the grader sees the quiz text
    ask(
        "Here are my quiz answers: "
        "1) A  2) A  3) True  "
        "4) Using point-slope form y - y1 = m(x - x1) with the given point and slope, "
        "then simplifying to slope-intercept form. "
        "5) Slope-intercept form y = mx + b is best when you know slope and intercept; "
        "point-slope is best when you know a point and the slope.",
        "4. grade + save",
    )

    after = load_profile()
    safe_print(
        f"\nprofile after: index={after.get('current_index')} "
        f"history={len(after.get('history', []))}"
    )
    if len(after.get("history", [])) > len(before.get("history", [])):
        safe_print("OK: update_progress wrote a new history entry.")
    else:
        safe_print("WARNING: no new history entry (agent may not have called update_progress).")

    _ = quiz_reply
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
