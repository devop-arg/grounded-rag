"""Deterministic offline "model" for ``eval --generate --mock`` and the app smoke test.

It reads the prompt it is given (single source of truth: the real prompt
builder) and answers with one sentence citing the first four-plus words of the
first passage, which is always a verbatim quote.  No network, no randomness,
no key.
"""

from __future__ import annotations

import json
import re

from .llm import LLMResult

SOURCE_RE = re.compile(r"<<<SOURCE (\d+)>>>\n(.*?)\n<<<END SOURCE \1>>>", re.DOTALL)
QUOTE_WORDS = 8


def parse_sources(prompt: str) -> list[tuple[int, str]]:
    """``[(n, text)]`` of every ``<<<SOURCE n>>>`` block in the user message."""
    return [(int(number), text) for number, text in SOURCE_RE.findall(prompt)]


class MockLLM:
    """Offline :class:`generation.llm.ChatModel` implementation."""

    def __init__(self, quote_words: int = QUOTE_WORDS) -> None:
        self.quote_words = quote_words
        self.calls = 0

    def complete(self, messages: list[dict[str, str]], *, json_mode: bool = True) -> LLMResult:
        self.calls += 1
        prompt = messages[-1]["content"] if messages else ""
        sources = parse_sources(prompt)
        if not sources:
            payload = {"abstain": True, "sentences": []}
        else:
            number, text = sources[0]
            words = " ".join(text.split()).split()
            if len(words) < 4:
                payload = {"abstain": True, "sentences": []}
            else:
                quote = " ".join(words[: self.quote_words])
                payload = {
                    "abstain": False,
                    "sentences": [
                        {
                            "text": f"Mock answer grounded in passage {number}.",
                            "citations": [{"n": number, "quote": quote}],
                        }
                    ],
                }
        body = json.dumps(payload, ensure_ascii=False)
        return LLMResult(
            text=body,
            prompt_tokens=max(1, len(prompt) // 4),
            completion_tokens=max(1, len(body) // 4),
            model="mock",
        )
