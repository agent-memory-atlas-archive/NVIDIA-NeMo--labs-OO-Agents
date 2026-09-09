# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Issuer-scoped capture and replay of closed-provider reasoning state.

The event IR treats provider state as an opaque dictionary. This module is the
only code that opens its NOOA envelope or places the payload on provider wire
messages. Unknown issuers and compatibility mismatches fail closed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from typing import Any, Literal, cast
from urllib.parse import urlsplit

import litellm

from nooa._llm_state import (
    LLM_STATE_KEY,
    carried_reasoning,
    carried_state,
)

logger = logging.getLogger(__name__)

_STATE_VERSION = 1
_CHAT_FORMAT = "litellm-chat"
_RESPONSES_FORMAT = "openai-responses"
_ENCRYPTED_REASONING_INCLUDE = "reasoning.encrypted_content"
_INLINE_THOUGHT_SIGNATURE_SEPARATOR = "__thought__"
_SUPPORTED_PROVIDERS = {
    "openai",
    "azure",
    "anthropic",
    "gemini",
}


def _field(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def response_item_type(item: Any) -> str | None:
    value = _field(item, "type")
    return value if isinstance(value, str) else None


def opaque_item(item: Any) -> Any:
    """Detach one provider-owned item for durable storage."""
    # Inspect the type: permissive mocks/proxies synthesize arbitrary instance
    # attributes and can otherwise recurse forever here.
    if callable(getattr(type(item), "model_dump", None)):
        return opaque_item(item.model_dump(exclude_none=True))
    if isinstance(item, dict):
        return {key: opaque_item(value) for key, value in item.items()}
    if isinstance(item, (list, tuple)):
        return [opaque_item(value) for value in item]
    return copy.deepcopy(item)


def _normalized_endpoint(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return "default"
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        return value.rstrip("/")
    path = parsed.path.rstrip("/")
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}{query}"


def _effective_endpoint(provider: str | None, configured: Any, resolved: Any) -> str:
    endpoint = configured or getattr(litellm, "api_base", None)
    if not endpoint and provider == "openai":
        endpoint = (
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        )
    elif not endpoint and provider == "azure":
        endpoint = os.getenv("AZURE_API_BASE")
    elif not endpoint and provider == "anthropic":
        endpoint = (
            os.getenv("ANTHROPIC_API_BASE")
            or os.getenv("ANTHROPIC_BASE_URL")
            or "https://api.anthropic.com"
        )
    elif not endpoint and provider == "gemini":
        endpoint = os.getenv("GEMINI_API_BASE") or "https://generativelanguage.googleapis.com"
    return _normalized_endpoint(endpoint or resolved)


def _credential_fingerprint(provider: str, configured: Any, resolved: Any) -> str | None:
    credential = configured or resolved
    if not credential:
        try:
            credential = litellm.get_api_key(provider, resolved)
        except Exception as exc:  # noqa: BLE001 - missing identity disables replay
            logger.debug("Could not resolve opaque-state credential for %s: %s", provider, exc)
            credential = None
    if not credential and provider == "azure":
        credential = (
            os.getenv("AZURE_API_KEY")
            or os.getenv("AZURE_OPENAI_API_KEY")
            or os.getenv("AZURE_AD_TOKEN")
        )
    elif not credential and provider == "anthropic":
        credential = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
    elif not credential and provider == "gemini":
        credential = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    reveal = getattr(credential, "get_secret_value", None)
    if callable(reveal):
        credential = reveal()
    if not isinstance(credential, str) or not credential:
        return None
    return hashlib.sha256(credential.encode()).hexdigest()


def replay_scope(
    model: str,
    api_style: Literal["chat", "responses"],
    params: dict[str, Any],
    declared_scope: str | None = None,
) -> str | None:
    """Return a non-secret compatibility key for the effective issuer route.

    LiteLLM resolves routing identity. Model, endpoint, credential/account, and
    API style are exact by default. A declared scope may group verified model
    aliases, but never bypasses endpoint, account, provider, or API isolation.
    """
    configured_endpoint = params.get("api_base") or params.get("base_url")
    try:
        resolved_model, provider, resolved_key, resolved_endpoint = litellm.get_llm_provider(
            model=model,
            custom_llm_provider=params.get("custom_llm_provider"),
            api_base=configured_endpoint,
        )
    except Exception as exc:  # noqa: BLE001 - unknown routes fail closed
        logger.debug("Could not resolve opaque-state issuer for %r: %s", model, exc)
        return None
    if provider not in _SUPPORTED_PROVIDERS or (
        api_style == "responses" and provider not in {"openai", "azure"}
    ):
        return None

    credential = _credential_fingerprint(
        provider,
        params.get("api_key") or params.get("azure_ad_token"),
        resolved_key,
    )
    if credential is None:
        logger.debug("Opaque-state replay disabled for %r: unknown credential", model)
        return None

    account: dict[str, str] = {}
    if provider in {"openai", "azure"}:
        for key, value in (
            (
                "organization",
                params.get("organization")
                or params.get("openai_organization")
                or getattr(litellm, "organization", None)
                or os.getenv("OPENAI_ORGANIZATION"),
            ),
            (
                "project",
                params.get("project")
                or params.get("openai_project")
                or getattr(litellm, "project", None)
                or os.getenv("OPENAI_PROJECT"),
            ),
        ):
            if isinstance(value, str) and value:
                account[key] = value
    identity = {
        "route": declared_scope or resolved_model,
        "endpoint": _effective_endpoint(provider, configured_endpoint, resolved_endpoint),
        "credential": credential,
        "account": account,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"{api_style}:{provider}:sha256:{digest}"


def _envelope(scope: str | None, state_format: str, payload: dict[str, Any]) -> dict | None:
    if scope is None or not payload:
        return None
    return {
        "version": _STATE_VERSION,
        "scope": scope,
        "format": state_format,
        "payload": copy.deepcopy(payload),
    }


def _matching_payload(state: Any, scope: str | None, state_format: str) -> dict | None:
    if (
        scope is None
        or not isinstance(state, dict)
        or state.get("version") != _STATE_VERSION
        or state.get("scope") != scope
        or state.get("format") != state_format
        or not isinstance(state.get("payload"), dict)
    ):
        return None
    return cast(dict[str, Any], copy.deepcopy(state["payload"]))


def _is_state_only(state: Any) -> bool:
    return (
        isinstance(state, dict)
        and state.get("version") == _STATE_VERSION
        and isinstance(state.get("payload"), dict)
        and state["payload"].get("state_only") is True
    )


def _scope_provider(scope: str | None) -> str | None:
    if not isinstance(scope, str):
        return None
    parts = scope.split(":", 2)
    return parts[1] if len(parts) == 3 else None


def _sanitize_chat_payload(payload: dict[str, Any], scope: str | None) -> dict[str, Any]:
    """Keep only replay fields whose LiteLLM Chat semantics NOOA knows."""
    provider = _scope_provider(scope)
    clean: dict[str, Any] = {}
    fields_by_provider = {
        "openai": ("reasoning_items", "thinking_blocks"),
        "azure": ("reasoning_items", "thinking_blocks"),
        "anthropic": ("thinking_blocks",),
        "gemini": ("thinking_blocks",),
    }
    for key in fields_by_provider.get(provider or "", ()):
        value = payload.get(key)
        if isinstance(value, list) and value:
            clean[key] = opaque_item(value)

    provider_fields = (
        payload.get("provider_specific_fields")
        if provider in {"openai", "azure", "gemini"}
        else None
    )
    signatures = (
        provider_fields.get("thought_signatures") if isinstance(provider_fields, dict) else None
    )
    if (
        isinstance(signatures, list)
        and signatures
        and all(isinstance(signature, str) and signature for signature in signatures)
    ):
        clean["provider_specific_fields"] = {"thought_signatures": copy.deepcopy(signatures)}

    tool_state = payload.get("tool_calls") if provider in {"openai", "azure", "gemini"} else None
    if isinstance(tool_state, list):
        calls: list[dict[str, Any] | None] = []
        for item in tool_state:
            fields = item.get("provider_specific_fields") if isinstance(item, dict) else None
            signature = fields.get("thought_signature") if isinstance(fields, dict) else None
            calls.append(
                {"provider_specific_fields": {"thought_signature": signature}}
                if isinstance(signature, str) and signature
                else None
            )
        if any(item is not None for item in calls):
            clean["tool_calls"] = calls

    tool_call_ids = payload.get("tool_call_ids")
    if (
        clean
        and isinstance(tool_call_ids, list)
        and tool_call_ids
        and all(isinstance(call_id, str) and call_id for call_id in tool_call_ids)
    ):
        clean["tool_call_ids"] = copy.deepcopy(tool_call_ids)

    if clean and payload.get("state_only") is True:
        clean["state_only"] = True
    return clean


def _tool_call_state(tool_call: Any) -> dict[str, Any] | None:
    dumped = opaque_item(tool_call)
    if not isinstance(dumped, dict):
        return None
    fields = dumped.get("provider_specific_fields")
    signature = fields.get("thought_signature") if isinstance(fields, dict) else None
    call_id = dumped.get("id")
    if (
        not signature
        and isinstance(call_id, str)
        and _INLINE_THOUGHT_SIGNATURE_SEPARATOR in call_id
    ):
        signature = call_id.split(_INLINE_THOUGHT_SIGNATURE_SEPARATOR, 1)[1]
    if not isinstance(signature, str) or not signature:
        return None
    return {"provider_specific_fields": {"thought_signature": signature}}


def public_tool_call_id(value: Any, scope: str | None) -> str:
    """Return an application call id without LiteLLM's inline Gemini state."""
    call_id = _field(value, "id", "")
    if not isinstance(call_id, str):
        return str(call_id or "")
    if _scope_provider(scope) != "gemini":
        return call_id
    return call_id.split(_INLINE_THOUGHT_SIGNATURE_SEPARATOR, 1)[0]


def capture_chat_state(message: Any, scope: str | None) -> dict | None:
    payload: dict[str, Any] = {}
    for key in ("reasoning_items", "thinking_blocks"):
        value = _field(message, key)
        if isinstance(value, list) and value:
            payload[key] = opaque_item(value)

    provider_fields = _field(message, "provider_specific_fields")
    if isinstance(provider_fields, dict):
        payload["provider_specific_fields"] = provider_fields

    raw_tool_calls = list(_field(message, "tool_calls") or [])
    tool_state = [_tool_call_state(call) for call in raw_tool_calls]
    if any(item is not None for item in tool_state):
        payload["tool_calls"] = tool_state

    if payload and raw_tool_calls:
        tool_call_ids = [public_tool_call_id(call, scope) for call in raw_tool_calls]
        if not all(tool_call_ids) or len(set(tool_call_ids)) != len(tool_call_ids):
            return None
        payload["tool_call_ids"] = tool_call_ids

    payload = _sanitize_chat_payload(payload, scope)
    if not payload:
        return None
    if not _field(message, "content") and not _field(message, "tool_calls"):
        payload["state_only"] = True
    return _envelope(scope, _CHAT_FORMAT, payload)


def _strip_inline_signature(value: Any) -> Any:
    if isinstance(value, str) and _INLINE_THOUGHT_SIGNATURE_SEPARATOR in value:
        return value.split(_INLINE_THOUGHT_SIGNATURE_SEPARATOR, 1)[0]
    return value


def _strip_chat_state(message: dict[str, Any], *, strip_inline_signatures: bool) -> None:
    for key in ("reasoning_items", "thinking_blocks", "provider_specific_fields"):
        message.pop(key, None)
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            if strip_inline_signatures:
                call["id"] = _strip_inline_signature(call.get("id"))
            call.pop("provider_specific_fields", None)
            function = call.get("function")
            if isinstance(function, dict):
                function.pop("provider_specific_fields", None)
    if strip_inline_signatures and "tool_call_id" in message:
        message["tool_call_id"] = _strip_inline_signature(message["tool_call_id"])


def _restore_chat_state(message: dict[str, Any], payload: dict[str, Any]) -> None:
    expected_call_ids = payload.get("tool_call_ids")
    if expected_call_ids is not None:
        tool_calls = message.get("tool_calls")
        if (
            not isinstance(tool_calls, list)
            or [call.get("id") if isinstance(call, dict) else None for call in tool_calls]
            != expected_call_ids
        ):
            return
    elif isinstance(payload.get("tool_calls"), list):
        # Positional per-call state cannot safely survive public call mutation.
        return

    for key in ("reasoning_items", "thinking_blocks", "provider_specific_fields"):
        if key in payload:
            message[key] = copy.deepcopy(payload[key])
    tool_calls = message.get("tool_calls")
    tool_state = payload.get("tool_calls")
    if not isinstance(tool_calls, list) or not isinstance(tool_state, list):
        return
    if len(tool_calls) != len(tool_state):
        return
    for call, state in zip(tool_calls, tool_state, strict=True):
        if not isinstance(call, dict) or not isinstance(state, dict):
            continue
        fields = state.get("provider_specific_fields")
        if isinstance(fields, dict):
            call["provider_specific_fields"] = copy.deepcopy(fields)


def _merge_reasoning_text(message: dict[str, Any], reasoning: str | None) -> None:
    if not reasoning or message.get("role") != "assistant":
        return
    content = message.get("content")
    if isinstance(content, str):
        if content.strip() == reasoning.strip():
            return
        message["content"] = f"{reasoning}\n\n{content}" if content else reasoning
    elif isinstance(content, list):
        message["content"] = [{"type": "text", "text": reasoning}, *content]
    elif content is None:
        message["content"] = reasoning


def capture_responses_state(output: list[Any], scope: str | None) -> dict | None:
    # NOOA only knows the OpenAI/Azure Responses item contract. Other providers
    # may expose a similarly shaped API through a gateway, but that is not
    # evidence that their opaque state is wire-compatible.
    if _scope_provider(scope) not in {"openai", "azure"}:
        return None
    items: list[Any] = []
    order: list[dict[str, Any]] = []
    has_public_carrier = False
    for item in output:
        item_type = response_item_type(item)
        if item_type == "reasoning":
            order.append({"type": "reasoning", "index": len(items)})
            items.append(opaque_item(item))
        elif item_type == "function_call":
            call_id = _field(item, "call_id")
            if isinstance(call_id, str):
                order.append({"type": "function_call", "call_id": call_id})
                has_public_carrier = True
        elif item_type == "message":
            order.append({"type": "message"})
            has_public_carrier = True
    if not items:
        return None
    payload: dict[str, Any] = {"items": items, "order": order}
    if not has_public_carrier:
        payload["state_only"] = True
    return _envelope(scope, _RESPONSES_FORMAT, payload)


def responses_reasoning_text(output: list[Any]) -> str | None:
    """Return provider-visible Responses reasoning summaries as plain text."""
    texts: list[str] = []
    for item in output:
        if response_item_type(item) != "reasoning":
            continue
        for summary in _field(item, "summary", []) or []:
            text = _field(summary, "text")
            if isinstance(text, str) and text:
                texts.append(text)
    return "\n".join(texts) or None


def prepare_chat_messages(messages: list[dict[str, Any]], scope: str | None) -> list[dict]:
    """Strip private/raw state and restore only a matching Chat payload."""
    prepared: list[dict[str, Any]] = []
    for original in messages:
        state = copy.deepcopy(carried_state(original))
        reasoning = carried_reasoning(original)
        message = copy.deepcopy(dict(original))
        message.pop(LLM_STATE_KEY, None)
        source_scope = state.get("scope") if isinstance(state, dict) else None
        _strip_chat_state(
            message, strip_inline_signatures=_scope_provider(source_scope) == "gemini"
        )
        payload = _matching_payload(state, scope, _CHAT_FORMAT)
        if payload is not None:
            payload = _sanitize_chat_payload(payload, scope) or None
        if payload:
            _restore_chat_state(message, payload)
        else:
            _merge_reasoning_text(message, reasoning)
        if (
            payload is None
            and _is_state_only(state)
            and message.get("role") == "assistant"
            and not message.get("content")
            and not message.get("tool_calls")
        ):
            continue
        prepared.append(message)
    return prepared


def _clean_responses_batch(batch: Any) -> list[dict[str, Any]]:
    if not isinstance(batch, list):
        return []
    clean: list[dict[str, Any]] = []
    for original in batch:
        if not isinstance(original, dict) or response_item_type(original) == "reasoning":
            continue
        item = copy.deepcopy(original)
        item.pop(LLM_STATE_KEY, None)
        item.pop("reasoning_items", None)
        clean.append(item)
    return clean


def _demote_responses_reasoning(
    clean: list[dict[str, Any]], reasoning: str | None
) -> list[dict[str, Any]]:
    if not reasoning:
        return clean
    message = next((item for item in clean if item.get("role") == "assistant"), None)
    if message is not None:
        if isinstance(message.get("content"), list):
            index = clean.index(message)
            return [
                *clean[:index],
                {"role": "assistant", "content": reasoning},
                *clean[index:],
            ]
        _merge_reasoning_text(message, reasoning)
        return clean
    return [{"role": "assistant", "content": reasoning}, *clean]


def prepare_responses_batch(
    batch: Any,
    state: Any,
    scope: str | None,
    reasoning: str | None = None,
) -> list[dict[str, Any]]:
    """Restore a matching Responses payload among its public turn carriers."""
    clean = _clean_responses_batch(batch)
    payload = (
        _matching_payload(state, scope, _RESPONSES_FORMAT)
        if _scope_provider(scope) in {"openai", "azure"}
        else None
    )
    if payload is None:
        if _is_state_only(state) and not reasoning:
            return []
        return _demote_responses_reasoning(clean, reasoning)
    items = payload.get("items")
    order = payload.get("order")
    if not isinstance(items, list) or not isinstance(order, list):
        return _demote_responses_reasoning(clean, reasoning)
    if payload.get("state_only") is True:
        return [copy.deepcopy(item) for item in items if isinstance(item, dict)]

    calls = {
        item.get("call_id"): item
        for item in clean
        if item.get("type") == "function_call" and isinstance(item.get("call_id"), str)
    }
    message = next((item for item in clean if item.get("role") == "assistant"), None)
    emitted_calls: set[str] = set()
    emitted_message = False
    pending: list[dict[str, Any]] = []
    replay: list[dict[str, Any]] = []
    last_carrier_emitted = False

    for slot in order:
        if not isinstance(slot, dict):
            continue
        if slot.get("type") == "reasoning":
            index = slot.get("index")
            if isinstance(index, int) and 0 <= index < len(items):
                item = items[index]
                if isinstance(item, dict):
                    pending.append(copy.deepcopy(item))
            continue
        if slot.get("type") == "function_call":
            call_id = slot.get("call_id")
            carrier = calls.get(call_id) if isinstance(call_id, str) else None
            if carrier is not None and isinstance(call_id, str):
                replay.extend(pending)
                replay.append(copy.deepcopy(carrier))
                emitted_calls.add(call_id)
                last_carrier_emitted = True
            else:
                last_carrier_emitted = False
            pending = []
            continue
        if slot.get("type") == "message":
            if message is not None and not emitted_message:
                replay.extend(pending)
                replay.append(copy.deepcopy(message))
                emitted_message = True
                last_carrier_emitted = True
            else:
                last_carrier_emitted = False
            pending = []

    if last_carrier_emitted:
        replay.extend(pending)
    replay.extend(
        copy.deepcopy(item)
        for item in clean
        if not (
            item is message
            and emitted_message
            or item.get("type") == "function_call"
            and item.get("call_id") in emitted_calls
        )
    )
    return replay


def add_encrypted_reasoning_include(api_params: dict[str, Any], scope: str | None) -> None:
    """Request OpenAI encrypted reasoning only on endpoints known to support it."""
    configured = api_params.get("include")
    include = list(configured) if isinstance(configured, (list, tuple, set)) else []
    if configured is not None and not isinstance(configured, (list, tuple, set)):
        include.append(configured)
    if _ENCRYPTED_REASONING_INCLUDE in include:
        api_params["include"] = include
        return
    if scope and scope.startswith("responses:azure:"):
        include.append(_ENCRYPTED_REASONING_INCLUDE)
    elif scope and scope.startswith("responses:openai:"):
        endpoint = _effective_endpoint(
            "openai", api_params.get("api_base") or api_params.get("base_url"), None
        )
        if endpoint == "https://api.openai.com/v1":
            include.append(_ENCRYPTED_REASONING_INCLUDE)
    if include:
        api_params["include"] = include
