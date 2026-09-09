# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-memory transport shared by formatters and the UnifiedLLM replay gate."""

from __future__ import annotations

import copy
from typing import Any

LLM_STATE_KEY = "_nooa_llm_state"


class StateCarryingMessage(dict[str, Any]):
    """A public wire message with provider state outside its mapping.

    Generic JSON serializers see only the ordinary dictionary fields. Built-in
    UnifiedLLM clients read ``llm_state`` and apply their issuer gate before the
    request reaches a provider.
    """

    __slots__ = ("llm_state",)

    def __init__(self, message: dict[str, Any], llm_state: dict[str, Any]):
        super().__init__(message)
        self.llm_state = copy.deepcopy(llm_state)


def carried_state(message: Any) -> dict[str, Any] | None:
    """Read the non-serializable sidecar from a rendered message."""
    state = getattr(message, "llm_state", None)
    return state if isinstance(state, dict) else None
