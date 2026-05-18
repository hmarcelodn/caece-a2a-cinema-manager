import os
import sys

from beeai_framework.agents.requirement.types import RequirementAgentRunState
from beeai_framework.tools.types import ToolOutput


def is_cinema_debug() -> bool:
    return os.getenv("CINEMA_DEBUG", "").lower() in ("1", "true", "yes")


def _truncate(text: str, max_len: int = 200) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}…"


def _tool_label(tool) -> str:
    if tool is None:
        return "unknown"
    return getattr(tool, "name", type(tool).__name__)


def _output_summary(output: ToolOutput | None) -> str:
    if output is None:
        return "(no output)"
    try:
        text = output.get_text_content()
    except Exception:
        return "(unreadable output)"
    if not text:
        return "(empty)"
    return f"{len(text)} chars: {_truncate(text)}"


def log_agent_steps(state: RequirementAgentRunState) -> None:
    if not is_cinema_debug():
        return

    print(f"[cinema-manager] run finished: {len(state.steps)} step(s)", file=sys.stderr)
    for step in state.steps:
        tool_name = _tool_label(step.tool)
        status = "ERROR" if step.error else "OK"
        err = f" — {step.error}" if step.error else ""
        task_preview = ""
        if isinstance(step.input, dict) and "task" in step.input:
            task_preview = f' task="{_truncate(str(step.input["task"]), 120)}"'
        elif step.input is not None:
            task_preview = f' input="{_truncate(str(step.input), 120)}"'
        print(
            f"[cinema-manager] [step {step.iteration}] {tool_name} → {status}{err}{task_preview}",
            file=sys.stderr,
        )
        if step.output is not None:
            print(
                f"[cinema-manager]   output: {_output_summary(step.output)}",
                file=sys.stderr,
            )
