# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-memory transport shared by formatters and the UnifiedLLM replay gate."""

from __future__ import annotations

import copy
from typing import Any
from uuid import uuid4

LLM_STATE_KEY = "_nooa_llm_state"


class ReplayCarryingMessage(dict[str, Any]):
    """A public wire message with internal metadata outside its mapping.

    Generic JSON serializers see only the ordinary dictionary fields. Built-in
    UnifiedLLM clients read the attributes, gate opaque state by issuer, and
    translate a provider-neutral cache boundary at the final wire edge.
    """

    __slots__ = (
        "llm_state",
        "reasoning",
        "cache_boundary_before",
        "replay_batch_id",
        "replay_batch_size",
    )

    def __init__(
        self,
        message: dict[str, Any],
        llm_state: dict[str, Any] | None = None,
        reasoning: str | None = None,
        cache_boundary_before: bool = False,
        *,
        replay_batch_id: str | None = None,
        replay_batch_size: int = 0,
    ):
        super().__init__(message)
        self.llm_state = copy.deepcopy(llm_state)
        self.reasoning = reasoning
        self.cache_boundary_before = cache_boundary_before
        self.replay_batch_id = replay_batch_id
        self.replay_batch_size = replay_batch_size


def carried_state(message: Any) -> dict[str, Any] | None:
    """Read the non-serializable sidecar from a rendered message."""
    state = getattr(message, "llm_state", None)
    return state if isinstance(state, dict) else None


def carry_replay_batch(
    messages: list[dict[str, Any]],
    llm_state: dict[str, Any] | None,
    reasoning: str | None,
) -> list[dict[str, Any]]:
    """Attach one replay envelope to a valid, JSON-safe Responses item batch."""
    batch_id = uuid4().hex
    size = len(messages)
    return [
        ReplayCarryingMessage(
            message,
            llm_state if index == 0 else None,
            reasoning if index == 0 else None,
            replay_batch_id=batch_id,
            replay_batch_size=size,
        )
        for index, message in enumerate(messages)
    ]


def carried_replay_batch(message: Any) -> tuple[str, int] | None:
    """Return a rendered Responses batch identity stored outside its mapping."""
    batch_id = getattr(message, "replay_batch_id", None)
    batch_size = getattr(message, "replay_batch_size", 0)
    if isinstance(batch_id, str) and isinstance(batch_size, int) and batch_size > 0:
        return batch_id, batch_size
    return None


def carried_reasoning(message: Any) -> str | None:
    """Read provider-exposed text reasoning from a rendered message."""
    reasoning = getattr(message, "reasoning", None)
    return reasoning if isinstance(reasoning, str) and reasoning else None


def carried_cache_boundary(message: Any) -> bool:
    """Return whether the volatile suffix begins at this rendered message."""
    return getattr(message, "cache_boundary_before", False) is True
