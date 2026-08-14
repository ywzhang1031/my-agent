from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, TextIO

from .agent_loop import AgentEvent


class LineEditor:
    def __init__(
        self,
        history_path: str | Path,
        *,
        readline_module: Any | None = None,
        input_func: Callable[[str], str] = input,
        history_length: int = 500,
    ) -> None:
        self.history_path = Path(history_path)
        self.input_func = input_func
        self.history_length = history_length
        self.readline = readline_module if readline_module is not None else _load_readline()

    def __enter__(self) -> LineEditor:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def start(self) -> None:
        if self.readline is None:
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        documentation = getattr(self.readline, "__doc__", "") or ""
        binding = "bind -e" if "libedit" in documentation else "set editing-mode emacs"
        self.readline.parse_and_bind(binding)
        self.readline.set_history_length(self.history_length)
        if self.history_path.exists():
            try:
                self.readline.read_history_file(str(self.history_path))
            except OSError:
                pass

    def read(self, prompt: str) -> str:
        return self.input_func(prompt)

    def close(self) -> None:
        if self.readline is None:
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.readline.write_history_file(str(self.history_path))
            os.chmod(self.history_path, 0o600)
        except OSError:
            pass


class StreamRenderer:
    def __init__(
        self,
        *,
        output: TextIO = sys.stdout,
        status_output: TextIO = sys.stderr,
        answer_prefix: str = "assistant> ",
        show_thinking: bool = False,
    ) -> None:
        self.output = output
        self.status_output = status_output
        self.answer_prefix = answer_prefix
        self.show_thinking = show_thinking
        self._line_open = False
        self._mode: str | None = None
        self._model_content_seen = False
        self._reasoning_status_seen = False

    def handle(self, event: AgentEvent) -> None:
        if event.kind == "model_started":
            self._close_line()
            self._mode = None
            self._model_content_seen = False
            self._reasoning_status_seen = False
        elif event.kind == "reasoning_delta":
            if self.show_thinking:
                self._write_delta("reasoning", "thinking> ", event.text)
            elif not self._reasoning_status_seen:
                self._status("thinking...")
                self._reasoning_status_seen = True
        elif event.kind == "content_delta":
            self._model_content_seen = True
            self._write_delta("content", self.answer_prefix, event.text)
        elif event.kind == "tool_started":
            self._close_line()
            arguments = _short_json(event.data.get("arguments", {}))
            self._status(f"tool> {event.data.get('name', '')} {arguments}".rstrip())
        elif event.kind == "tool_finished":
            self._close_line()
            status = event.data.get("status") or (
                "success" if event.data.get("ok") else "failed"
            )
            duration = event.data.get("duration_ms")
            suffix = f" ({duration} ms)" if duration is not None else ""
            self._status(f"tool< {event.data.get('name', '')} {status}{suffix}".rstrip())
        elif event.kind == "provider_retry":
            self._close_line()
            attempt = event.data.get("attempt")
            delay = event.data.get("delay_seconds")
            self._status(f"retry> attempt {attempt} in {delay}s: {event.text}")
        elif event.kind == "context_compacted":
            self._close_line()
            self._status(
                "context> compacted "
                f"{event.data.get('compacted_messages', 0)} messages; "
                f"estimated {event.data.get('estimated_tokens', 0)}/"
                f"{event.data.get('input_budget', 0)} tokens"
            )
        elif event.kind == "turn_resumed":
            self._status(f"retry> resuming model step {event.data.get('step')}")
        elif event.kind == "turn_error":
            self._close_line()
            self._status(f"error> {event.text}")

    def finish(self, fallback_answer: str) -> None:
        if not self._model_content_seen and fallback_answer:
            self._write_delta("content", self.answer_prefix, fallback_answer)
        self._close_line()

    def _write_delta(self, mode: str, prefix: str, text: str) -> None:
        if self._mode != mode:
            self._close_line()
            self.output.write(prefix)
            self._line_open = True
            self._mode = mode
        self.output.write(text)
        self.output.flush()

    def _close_line(self) -> None:
        if self._line_open:
            self.output.write("\n")
            self.output.flush()
        self._line_open = False
        self._mode = None

    def _status(self, text: str) -> None:
        self.status_output.write(text + "\n")
        self.status_output.flush()


def _load_readline() -> Any | None:
    try:
        import readline
    except ImportError:
        return None
    return readline


def _short_json(value: Any, max_chars: int = 240) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"
