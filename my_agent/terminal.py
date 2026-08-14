from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import AnyFormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.styles import Style

from .agent_loop import AgentEvent
from .context import ContextSnapshot


@dataclass(frozen=True)
class SlashCommand:
    name: str
    description: str


class SlashCommandCompleter(Completer):
    def __init__(self, commands: Iterable[SlashCommand]) -> None:
        self.commands = tuple(commands)

    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent,
    ) -> Iterable[Completion]:
        prefix = document.text_before_cursor
        if not prefix.startswith("/") or any(character.isspace() for character in prefix):
            return
        lowered = prefix.lower()
        for command in self.commands:
            if command.name.lower().startswith(lowered):
                yield Completion(
                    command.name,
                    start_position=-len(prefix),
                    display=command.name,
                    display_meta=command.description,
                )


TERMINAL_STYLE = Style.from_dict(
    {
        "bottom-toolbar": "bg:#262626 #bcbcbc",
        "bottom-toolbar.label": "bg:#262626 #87afff bold",
        "bottom-toolbar.pending": "bg:#262626 #ffaf5f bold",
        "bottom-toolbar.hint": "bg:#262626 #8a8a8a",
        "completion-menu.completion": "bg:#303030 #eeeeee",
        "completion-menu.completion.current": "bg:#5f5faf #ffffff bold",
        "completion-menu.meta.completion": "bg:#303030 #a8a8a8",
        "completion-menu.meta.completion.current": "bg:#5f5faf #ffffff",
    }
)


class LineEditor:
    def __init__(
        self,
        history_path: str | Path,
        *,
        commands: Iterable[SlashCommand],
        status_provider: Callable[[], AnyFormattedText] | None = None,
        prompt_session: Any | None = None,
    ) -> None:
        self.history_path = Path(history_path)
        self.status_provider = status_provider
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.touch(mode=0o600, exist_ok=True)
        self._secure_history()
        self.session = prompt_session or PromptSession(
            history=FileHistory(str(self.history_path)),
            completer=SlashCommandCompleter(commands),
            complete_while_typing=True,
            complete_style=CompleteStyle.COLUMN,
            reserve_space_for_menu=10,
            enable_history_search=True,
            key_bindings=_completion_key_bindings(),
            style=TERMINAL_STYLE,
        )

    def __enter__(self) -> LineEditor:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def start(self) -> None:
        self._secure_history()

    def read(self, prompt: str) -> str:
        try:
            return self.session.prompt(
                prompt,
                bottom_toolbar=self.status_provider,
            )
        finally:
            self._secure_history()

    def close(self) -> None:
        self._secure_history()

    def _secure_history(self) -> None:
        if not self.history_path.exists():
            return
        try:
            os.chmod(self.history_path, 0o600)
        except OSError:
            pass


def format_context_status(
    snapshot: ContextSnapshot,
    *,
    pending: bool = False,
) -> list[tuple[str, str]]:
    percent = _format_percent(snapshot.estimated_tokens, snapshot.context_window_tokens)
    fragments = [
        ("class:bottom-toolbar", " "),
        ("class:bottom-toolbar.label", "context "),
        (
            "class:bottom-toolbar",
            f"{percent}  ~{_format_tokens(snapshot.estimated_tokens)}/"
            f"{_format_tokens(snapshot.context_window_tokens)}",
        ),
    ]
    if pending:
        fragments.append(("class:bottom-toolbar.pending", "  pending turn"))
    fragments.extend(
        [
            ("class:bottom-toolbar.hint", "  |  / commands  |  Tab select "),
        ]
    )
    return fragments


def render_context_usage(
    snapshot: ContextSnapshot,
    *,
    last_provider_input: int = 0,
    width: int = 48,
    color: bool = True,
) -> str:
    if width < 10:
        raise ValueError("context bar width must be at least 10")
    breakdown = snapshot.breakdown
    rows = [
        ("System prompt", breakdown.system_prompt_tokens, "90"),
        ("Tool definitions", breakdown.tool_definition_tokens, "35"),
        ("Conversation summary", breakdown.summary_tokens, "36"),
        ("Conversation", breakdown.conversation_tokens, "33"),
        ("Protocol overhead", breakdown.protocol_tokens, "34"),
    ]
    percent = _format_percent(snapshot.estimated_tokens, snapshot.context_window_tokens)
    usage = (
        f"~{_format_tokens(snapshot.estimated_tokens)} / "
        f"{_format_tokens(snapshot.context_window_tokens)} tokens"
    )
    lines = [
        "Context Usage",
        f"  {percent} full".ljust(width + 3 - len(usage)) + usage,
        "  [" + _context_bar(rows, snapshot.context_window_tokens, width, color) + "]",
        "",
    ]
    for label, tokens, ansi_color in rows:
        marker = _paint("#", ansi_color, color)
        lines.append(f"  {marker} {label:<22}{_format_tokens(tokens):>10}")
    lines.extend(
        [
            "",
            f"  {'Input budget':<24}{_format_tokens(snapshot.input_budget):>10}",
            f"  {'Reserved output':<24}{_format_tokens(snapshot.reserve_output_tokens):>10}",
            f"  {'Last provider input':<24}{_format_tokens(last_provider_input):>10}",
            f"  Messages: {snapshot.active_messages} active, "
            f"{snapshot.summarized_messages} summarized",
        ]
    )
    return "\n".join(lines)


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


def _completion_key_bindings() -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add("tab")
    def select_next_completion(event: Any) -> None:
        buffer = event.app.current_buffer
        if buffer.complete_state:
            buffer.complete_next()
        else:
            buffer.start_completion(select_first=True)

    @bindings.add("s-tab")
    def select_previous_completion(event: Any) -> None:
        buffer = event.app.current_buffer
        if buffer.complete_state:
            buffer.complete_previous()
        else:
            buffer.start_completion(select_first=True)

    return bindings


def _context_bar(
    rows: list[tuple[str, int, str]],
    context_window_tokens: int,
    width: int,
    color: bool,
) -> str:
    parts: list[str] = []
    remaining = width
    for _, tokens, ansi_color in rows:
        if tokens <= 0 or remaining <= 0:
            continue
        columns = round(tokens / context_window_tokens * width)
        columns = min(remaining, max(1, columns))
        parts.append(_paint("#" * columns, ansi_color, color))
        remaining -= columns
    parts.append(_paint("-" * remaining, "90", color))
    return "".join(parts)


def _paint(text: str, ansi_color: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"\033[{ansi_color}m{text}\033[0m"


def _format_tokens(tokens: int) -> str:
    if tokens >= 1_000_000:
        return _trim_decimal(tokens / 1_000_000) + "M"
    if tokens >= 1_000:
        return _trim_decimal(tokens / 1_000) + "K"
    return str(tokens)


def _format_percent(used: int, limit: int) -> str:
    percent = used / limit * 100
    if 0 < percent < 1:
        return f"{percent:.1f}%"
    return f"{round(percent)}%"


def _trim_decimal(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _short_json(value: Any, max_chars: int = 240) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"
