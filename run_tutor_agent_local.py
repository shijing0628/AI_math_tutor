"""
Local smoke test / playground for the ReAct tutor agent (Steps 1–2).

Usage:
    python run_tutor_agent_local.py
    python run_tutor_agent_local.py "What concept should I study next?"
    python run_tutor_agent_local.py --interactive

This does not start Streamlit. It only exercises tutor_tools + tutor_agent.
"""

from __future__ import annotations

import argparse
import sys

from tutor_agent import build_tutor_agent


def safe_print(text: str) -> None:
    """Print text safely on Windows consoles that are not UTF-8."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def print_tool_trace(result: dict) -> None:
    """Print which tools the agent called (helpful for learning ReAct)."""
    messages = result.get("messages") or []
    safe_print("\n--- tool trace ---")
    found = False
    for msg in messages:
        # AIMessage with tool_calls
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            found = True
            for call in tool_calls:
                name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "?")
                args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
                safe_print(f"  tool -> {name}({args})")
        # ToolMessage results (truncated)
        msg_type = getattr(msg, "type", None) or msg.__class__.__name__
        if msg_type in ("tool", "ToolMessage"):
            found = True
            name = getattr(msg, "name", "tool")
            content = str(getattr(msg, "content", ""))[:180].replace("\n", " ")
            safe_print(f"  result <- {name}: {content}...")
    if not found:
        safe_print("  (no tool calls — the model answered directly)")
    safe_print("--- end trace ---\n")


def run_once(prompt: str) -> None:
    safe_print(f"User: {prompt}\n")
    agent = build_tutor_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    print_tool_trace(result)
    messages = result.get("messages") or []
    last = messages[-1] if messages else None
    content = getattr(last, "content", "") if last is not None else ""
    safe_print(f"Agent:\n{content}")


def run_interactive() -> None:
    print("Interactive tutor agent. Type 'quit' to exit.\n")
    agent = build_tutor_agent()
    history: list[dict] = []
    while True:
        try:
            user = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user:
            continue
        if user.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break
        history.append({"role": "user", "content": user})
        result = agent.invoke({"messages": history})
        print_tool_trace(result)
        messages = result.get("messages") or []
        # Keep full message history for multi-turn tool use
        history = messages
        last = messages[-1] if messages else None
        content = getattr(last, "content", "") if last is not None else ""
        print(f"Agent> {content}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local ReAct tutor agent test")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="What concept should I study next? Please check my current progress.",
        help="Single-turn user prompt",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Multi-turn chat loop",
    )
    args = parser.parse_args(argv)

    try:
        if args.interactive:
            run_interactive()
        else:
            run_once(args.prompt)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
