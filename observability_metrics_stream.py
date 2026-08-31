"""
title: Observability metrics (stream)
id: observability_metrics_stream.py
author: Willian Zhang
project_url: https://github.com/Willian-Zhang/openwebui-functions
funding_url: https://github.com/sponsors/Willian-Zhang
original_author: vigneshwarrvenkat
version: 0.10
required_open_webui_version: 0.5.17

Open WebUI **filter** (`type: filter`): streams timing hints using the existing
`status` event type (`event_emitter`), so **no frontend changes** are required.
Metrics appear under the assistant message status area like other tooling status.

Timing model:
- `inlet()` records the request start, so TTFT is measured from request
  submission (not from the first chunk seen, which is always ~0 ms).
- `stream()` observes real streaming chunks when the backend streams through
  OpenWebUI's chunk pipeline (OpenAI-compatible endpoints, generator pipes).
- Pipes that deliver content via `__event_emitter__` and return a plain string
  (e.g. manifold pipes) bypass the `stream()` hook: their chunks arrive as one
  burst at the end. For those, `outlet()` falls back to inlet→outlet wall time
  and skips throughput claims that would be meaningless.
- Provenance convention: "~" prefixes estimated values (len/4 heuristics);
  numbers without "~" are exact (provider usage) or directly measured
  (monotonic clock timings).
- Prompt processing (pp): prompt token count comes from usage (exact) or a
  len/4 estimate of the inlet messages. The prefill rate is prompt_tokens /
  TTFT — TTFT includes network/queue time, so it is a lower bound on the
  server's true prefill speed and is labeled "incl. latency". In the
  wall-time fallback (no observable TTFT) only the count is shown.

Debugging: turn on the `debug_prompt_breakdown` valve to emit an extra status
line at request start showing what the filter can see in the inlet body
(message count, text chars, image/video parts, tools JSON size). Use it to
locate "invisible" prompt tokens: anything not listed there (built-in/native
tool schemas, RAG context, image pads expanded server-side) is injected after
this filter runs and only shows up in the provider's exact usage counts.

Paste this file into Admin → Functions → New → type **Filter**.
Enable globally or attach via model filter IDs as usual.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Callable, Iterable, Mapping, MutableMapping
import asyncio
import json
import logging
import time

log = logging.getLogger(__name__)

AEmitter = Callable[[dict], asyncio.Future | Any]

# Drop bookkeeping for requests that never completed (disconnects, errors).
_STALE_AFTER_S = 2 * 60 * 60


def _iter_text_from_delta(delta: Mapping[str, Any]) -> Iterable[str]:
    parts: list[Any] = []
    for key in ("content", "reasoning_content", "reasoning", "thinking"):
        v = delta.get(key)
        if isinstance(v, str) and v:
            parts.append(v)
    audio = delta.get("audio")
    if isinstance(audio, Mapping):
        t = audio.get("transcript")
        if isinstance(t, str) and t:
            parts.append(t)
    return parts


def _stream_key(
    metadata: Mapping[str, Any] | None,
) -> tuple[str, str] | None:
    if not metadata:
        return None
    cid = metadata.get("chat_id")
    mid = metadata.get("message_id")
    if isinstance(cid, str) and isinstance(mid, str):
        return (cid, mid)
    return None


def _fmt_duration(seconds: float) -> str:
    # Durations are always measured, so no "~" prefix.
    return f"{max(0.0, seconds):.2f} s"


def _extract_usage_int(body: Mapping[str, Any], keys: tuple[str, ...]) -> int | None:
    """Best-effort token count from usage objects in an outlet body."""
    candidates: list[Any] = [body.get("usage")]
    messages = body.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, Mapping) and msg.get("role") == "assistant":
                candidates.append(msg.get("usage"))
                break
    for usage in candidates:
        if isinstance(usage, Mapping):
            for key in keys:
                ct = usage.get(key)
                if isinstance(ct, (int, float)) and ct > 0:
                    return int(ct)
    return None


def _prompt_breakdown(body: Mapping[str, Any]) -> str:
    """Human-readable inventory of everything countable in an inlet body."""
    n_msgs = 0
    text_chars = 0
    images = 0
    videos = 0
    messages = body.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, Mapping):
                continue
            n_msgs += 1
            content = msg.get("content")
            if isinstance(content, str):
                text_chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, Mapping):
                        continue
                    t = part.get("text")
                    if isinstance(t, str):
                        text_chars += len(t)
                    ptype = part.get("type")
                    if ptype in ("image", "image_url") or "image_url" in part:
                        images += 1
                    elif ptype in ("video", "video_url") or "video_url" in part:
                        videos += 1

    parts = [
        f"{n_msgs} msgs",
        f"text {text_chars} chars (~{max(1, round(text_chars / 4))} tok)",
    ]
    if images:
        parts.append(f"images {images} (server-side tokens, not counted)")
    if videos:
        parts.append(f"videos {videos} (server-side tokens, not counted)")

    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        try:
            tool_chars = len(json.dumps(tools))
        except Exception:
            tool_chars = 0
        parts.append(
            f"tools {len(tools)}"
            f" ({tool_chars} chars, ~{max(1, round(tool_chars / 4))} tok, not counted)"
        )
    else:
        # OpenWebUI attaches native tool specs in middleware *after* filter
        # inlets, so "none seen" here does not mean none were sent — built-in
        # integrated tools alone can add thousands of prompt tokens.
        parts.append("tools none seen (may be injected after this filter)")

    files = body.get("files")
    if isinstance(files, list) and files:
        parts.append(f"files {len(files)} (RAG text may be injected after this filter)")

    known = {"messages", "model", "stream", "stream_options", "tools", "files", "metadata"}
    extra = sorted(k for k in body if k not in known)
    if extra:
        parts.append("other keys: " + ", ".join(extra))
    return " · ".join(parts)


def _estimate_prompt_chars(body: Mapping[str, Any]) -> int:
    """Rough character count of everything sent to the model (inlet body)."""
    total = 0
    messages = body.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, Mapping):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, Mapping):
                        t = part.get("text")
                        if isinstance(t, str):
                            total += len(t)
    return total


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=101, description="Run near the end so counts see normalized chunks."
        )
        debug_prompt_breakdown: bool = Field(
            default=False,
            description=(
                "Emit a status line at request start listing what the inlet"
                " body contains (messages, text chars, images, tools size)."
                " Also logged at INFO level."
            ),
        )

    def __init__(self):
        self.valves = self.Valves()
        # Per (chat_id, message_id): timing + token estimate.
        self._state: MutableMapping[tuple[str, str], MutableMapping[str, Any]] = {}

    def _purge(self, key: tuple[str, str]) -> None:
        self._state.pop(key, None)

    def _purge_stale(self) -> None:
        now = time.monotonic()
        stale = [
            k
            for k, st in self._state.items()
            if now - st.get("mono_start", now) > _STALE_AFTER_S
        ]
        for k in stale:
            self._state.pop(k, None)

    async def _emit(
        self,
        emitter: AEmitter | None,
        *,
        description: str,
        done: bool,
    ) -> None:
        if not emitter:
            return
        payload = {
            "type": "status",
            "data": {
                "done": done,
                "action": "observability_metrics",
                "description": description,
                "hidden": False,
            },
        }
        res = emitter(payload)
        if asyncio.iscoroutine(res):
            await res

    async def inlet(
        self,
        body: dict,
        __metadata__: dict | None = None,
        __event_emitter__: AEmitter | None = None,
    ) -> dict:
        keys = _stream_key(__metadata__)
        if keys is None:
            return body

        self._purge_stale()
        # Request start: this is the reference point for TTFT and total time.
        self._state[keys] = {
            "mono_start": time.monotonic(),
            # Fallback prompt size if the provider never reports prompt_tokens.
            "prompt_chars_est": _estimate_prompt_chars(body),
        }

        if self.valves.debug_prompt_breakdown:
            breakdown = _prompt_breakdown(body)
            log.info("observability inlet %s: %s", keys, breakdown)
            await self._emit(
                __event_emitter__,
                description=f"debug inlet: {breakdown}",
                done=False,
            )
        return body

    async def stream(
        self,
        event: dict,
        __metadata__: dict | None = None,
        __event_emitter__: AEmitter | None = None,
    ) -> dict:
        emitter = __event_emitter__

        keys = _stream_key(__metadata__)
        if keys is None:
            return event

        st = self._state.setdefault(keys, {})
        mono_now = time.monotonic()

        # Fallback if inlet never ran (e.g. filter enabled mid-request).
        if "mono_start" not in st:
            st["mono_start"] = mono_now
            st["start_is_stream_local"] = True

        st["saw_stream"] = True

        choices = event.get("choices")
        delta: dict | None = None
        finish_reason = None
        usage = event.get("usage")
        try:
            if isinstance(choices, list) and choices:
                ch0 = choices[0]
                delta = (
                    dict(ch0.get("delta"))
                    if isinstance(ch0.get("delta"), dict)
                    else None
                )
                finish_reason = ch0.get("finish_reason")
        except Exception as e:
            log.debug("observability filter parse delta: %s", e)

        if isinstance(delta, dict):
            chars = "".join(_iter_text_from_delta(delta)).strip()
            if chars:
                if "mono_first_body" not in st:
                    st["mono_first_body"] = mono_now
                elif not st.get("ttft_emitted") and not st.get(
                    "start_is_stream_local"
                ):
                    # Emit TTFT only once a *second* text chunk proves this is
                    # a real stream — burst pipes (whole response in one chunk)
                    # would otherwise report total time as "first text".
                    st["ttft_emitted"] = True
                    ttft_s = float(st["mono_first_body"]) - float(st["mono_start"])
                    # Interim lines use full words; only the final summary is
                    # abbreviated. "~" marks estimated values.
                    desc = f"First model text after {_fmt_duration(ttft_s)}"
                    # Prompt tokens are only estimable here (usage arrives at
                    # the end); the final line recomputes pp with exact usage.
                    pc = st.get("prompt_chars_est", 0)
                    if isinstance(pc, int) and pc > 0 and ttft_s > 0:
                        pp_tok = max(1, round(pc / 4))
                        pp_rate = round(pp_tok / ttft_s)
                        desc += (
                            f" · prompt processing ~{pp_tok} tokens,"
                            f" ~{pp_rate} tokens/s"
                        )
                    await self._emit(emitter, description=desc, done=False)
                # Accumulate characters; estimate tokens once at the end so
                # many tiny chunks don't each count as >=1 token.
                st["approx_chars"] = st.get("approx_chars", 0) + len(chars)

        if isinstance(usage, Mapping) and usage:
            ct = usage.get("completion_tokens")
            if isinstance(ct, (int, float)) and ct > 0:
                st["completion_tokens"] = int(ct)
            pt = usage.get("prompt_tokens")
            if isinstance(pt, (int, float)) and pt > 0:
                st["prompt_tokens"] = int(pt)

        # Record when generation ended, but do NOT emit the summary here:
        # in the OpenAI stream protocol the usage chunk arrives in a separate
        # chunk *after* finish_reason (with empty choices), so finalizing on
        # finish_reason would discard exact token counts. The summary is
        # emitted from outlet(), which runs after the stream fully ends.
        raw_type = event.get("type")
        terminal = bool(
            (isinstance(finish_reason, str) and finish_reason)
            or raw_type in ("DONE", "[DONE]", "done")
            or event.get("done")
        )
        if terminal and "mono_end" not in st:
            st["mono_end"] = mono_now

        return event

    async def outlet(
        self,
        body: dict,
        __metadata__: dict | None = None,
        __event_emitter__: AEmitter | None = None,
    ) -> dict:
        keys = _stream_key(__metadata__)
        if keys is None:
            return body

        st = self._state.pop(keys, None)
        if st is None:
            return body

        mono_start = st.get("mono_start")
        if not isinstance(mono_start, (int, float)):
            return body

        mono_first = st.get("mono_first_body")
        mono_end = st.get("mono_end")
        approx_chars = st.get("approx_chars", 0)

        # Tokens: prefer streamed usage, then usage in the outlet body,
        # then the char/4 estimate.
        tokens = st.get("completion_tokens")
        if not isinstance(tokens, int):
            tokens = _extract_usage_int(body, ("completion_tokens", "output_tokens"))
        tokens_exact = isinstance(tokens, int)
        if not tokens_exact and approx_chars > 0:
            tokens = max(1, round(approx_chars / 4))
        # "~" prefix marks estimated values; exact values have none.
        tok_p = "" if tokens_exact else "~"

        # Prompt (prefill) side: same priority order.
        prompt_tokens = st.get("prompt_tokens")
        if not isinstance(prompt_tokens, int):
            prompt_tokens = _extract_usage_int(body, ("prompt_tokens", "input_tokens"))
        prompt_exact = isinstance(prompt_tokens, int)
        if not prompt_exact:
            prompt_chars = st.get("prompt_chars_est", 0)
            if isinstance(prompt_chars, int) and prompt_chars > 0:
                prompt_tokens = max(1, round(prompt_chars / 4))
        prompt_p = "" if prompt_exact else "~"

        streamed = isinstance(mono_first, (int, float)) and isinstance(
            mono_end, (int, float)
        )
        # A pipe that returns a string (instead of streaming) shows up as the
        # whole response arriving in one terminal burst: per-chunk timing
        # across that burst is meaningless, so fall through to wall time.
        looks_like_burst = (
            streamed
            and (float(mono_end) - float(mono_first)) < 0.1
            and approx_chars > 400
        )

        if streamed and not looks_like_burst:
            total_s = float(mono_end) - float(mono_start)

            ttft_s = None
            if not st.get("start_is_stream_local"):
                ttft_s = float(mono_first) - float(mono_start)

            done_part = f"Done {_fmt_duration(total_s)}"
            if st.get("start_is_stream_local"):
                # Timer started on first chunk, not at request submission.
                done_part += " (from first chunk)"
            summary_parts = [done_part]
            if isinstance(prompt_tokens, int) and prompt_tokens > 0:
                pp_part = f"pp {prompt_p}{prompt_tokens} tok"
                if ttft_s is not None and ttft_s > 0:
                    # TTFT includes network/queue time, so this is a lower
                    # bound on the server's true prefill rate.
                    pp_part += f", {prompt_p}{round(prompt_tokens / ttft_s)} tok/s"
                summary_parts.append(pp_part)
            if isinstance(tokens, int) and tokens > 0:
                gen_s = max(1e-9, float(mono_end) - float(mono_first))
                tg_rate = round(tokens / gen_s)
                summary_parts.append(
                    f"tg {tok_p}{tokens} tok, {tok_p}{tg_rate} tok/s"
                )
        else:
            # No per-chunk timing was observable (event-emitter pipe or
            # non-streaming response), so report wall time only. No TTFT
            # means no pp rate; tg over wall time includes latency.
            end = float(mono_end) if isinstance(mono_end, (int, float)) else time.monotonic()
            total_s = end - float(mono_start)
            summary_parts = [f"Done {_fmt_duration(total_s)} (wall)"]
            if isinstance(prompt_tokens, int) and prompt_tokens > 0:
                summary_parts.append(f"pp {prompt_p}{prompt_tokens} tok")
            if isinstance(tokens, int) and tokens > 0:
                avg_tps = round(tokens / max(1e-9, total_s))
                summary_parts.append(
                    f"tg {tok_p}{tokens} tok, {tok_p}{avg_tps} tok/s incl. latency"
                )

        await self._emit(
            __event_emitter__, description=" · ".join(summary_parts), done=True
        )
        return body
