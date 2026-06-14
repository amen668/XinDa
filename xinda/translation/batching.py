"""Three-level batching for translation requests.

Same logic as v1 (count → char → token), but operating on plain dicts so
batching is decoupled from any ORM session.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T", bound=dict)

DEFAULT_CHUNK_SIZE = 30
DEFAULT_MAX_CHARS = 10_000
DEFAULT_MAX_TOKENS = 5_000


def split_by_count(items: list[T], chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[list[T]]:
    return [items[i: i + chunk_size] for i in range(0, len(items), chunk_size)]


def split_by_chars(
    items: list[T],
    max_chars: int = DEFAULT_MAX_CHARS,
    text_key: str = "src_text",
) -> list[list[T]]:
    out: list[list[T]] = []
    cur: list[T] = []
    cur_len = 0
    for it in items:
        n = len(it.get(text_key, ""))
        if cur and cur_len + n > max_chars:
            out.append(cur)
            cur, cur_len = [], 0
        cur.append(it)
        cur_len += n
    if cur:
        out.append(cur)
    return out


def split_by_token(
    items: list[T],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    token_fn: Callable[[str], int] | None = None,
    text_key: str = "src_text",
) -> list[list[T]]:
    if token_fn is None:
        token_fn = lambda s: max(1, int(len(s) * 0.5))  # noqa: E731
    out: list[list[T]] = []
    cur: list[T] = []
    cur_tok = 0
    for it in items:
        t = token_fn(it.get(text_key, ""))
        if cur and cur_tok + t > max_tokens:
            out.append(cur)
            cur, cur_tok = [], 0
        cur.append(it)
        cur_tok += t
    if cur:
        out.append(cur)
    return out


def batch_all(
    items: list[T],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    token_fn: Callable[[str], int] | None = None,
    text_key: str = "src_text",
) -> list[list[T]]:
    """Apply all three batching strategies in sequence."""
    out: list[list[T]] = []
    for level1 in split_by_count(items, chunk_size):
        for level2 in split_by_chars(level1, max_chars, text_key):
            out.extend(split_by_token(level2, max_tokens, token_fn, text_key))
    return out
