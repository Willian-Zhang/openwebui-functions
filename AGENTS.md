# AGENTS.md

Guidance for coding agents and contributors working in this repository.

## What this repo is

A collection of **Open WebUI functions**. Each function is one self-contained
Python file at the repo root that users paste into
*Admin Panel → Functions → New*. There is no package, no build step, no test
suite, and no CI. The only runtime dependency is `pydantic`, provided by Open
WebUI itself.

Current functions:

- `observability_metrics_stream.py` — `type: filter`. Emits timing/token
  metrics (TTFT, prompt processing, generation rate, cache/reasoning tokens,
  cost) as `status` events on the assistant message.

## Hard constraints

1. **One file per function, fully self-contained.** No relative imports, no
   shared helper modules, no new third-party dependencies. Open WebUI executes
   the file in isolation; anything it needs must be inside it or in the
   standard library / `pydantic`.
2. **Keep the frontmatter docstring intact.** The module docstring at the top
   of each file is parsed by Open WebUI as metadata:

   ```
   title: ...
   id: ...
   author: ...
   project_url: ...
   funding_url: ...
   original_author: ...
   version: 0.11
   required_open_webui_version: 0.5.17
   ```

   Do not rename or reorder these keys. **Bump `version`** whenever behavior
   changes. Bump `required_open_webui_version` only if you start relying on a
   newer hook or API.
3. **Never break the user-facing status output without updating the docs.**
   The format of the emitted status lines is documented in `README.md` and in
   the module docstring; keep all three in sync.
4. **No frontend changes.** Functions must only use existing Open WebUI event
   types (`status`, etc.). If a feature would require a UI change, it does not
   belong here.
5. Python 3.11+ syntax is fine (`X | None`, `from __future__ import annotations`
   is already used). Avoid anything newer than what Open WebUI's Docker image
   ships.

## Open WebUI filter contract (for `observability_metrics_stream.py`)

The `Filter` class exposes:

- `class Valves(BaseModel)` — admin-configurable settings. Every field needs a
  `description`; it is shown in the UI.
- `async def inlet(body, __metadata__, __event_emitter__)` — runs before the
  request hits the model. Must return `body`.
- `async def stream(event, __metadata__, __event_emitter__)` — runs per chunk
  **only** when the backend streams through Open WebUI's chunk pipeline
  (OpenAI-compatible endpoints, generator pipes). Must return `event`.
- `async def outlet(body, __metadata__, __event_emitter__)` — runs after the
  full response. Must return `body`.

Hooks must be non-fatal: never raise out of a hook, never mutate `body` /
`event` in a way that changes model input or output. This function is
observe-only.

## Architecture notes

- **Per-request state** lives in `self._state`, keyed by
  `(chat_id, message_id)` from `__metadata__` (`_stream_key`). `inlet` creates
  it, `outlet` pops it, `_purge_stale` drops entries older than
  `_STALE_AFTER_S` for requests that never completed.
- **Timing** uses `time.monotonic()` only. `mono_start` is set in `inlet`
  (request submission), `mono_first_body` on the first non-empty text delta,
  `mono_end` on `finish_reason` / `[DONE]`.
- **TTFT is emitted only on the second text chunk**, and only if `inlet` ran.
  A single-chunk "burst" (a pipe returning a whole string) would otherwise
  report total time as TTFT. `outlet` applies the same `looks_like_burst`
  heuristic (`< 0.1 s` between first and last chunk with `> 400` chars) and
  falls back to wall time.
- **The final summary is emitted from `outlet`, not on `finish_reason`.** In
  the OpenAI stream protocol the `usage` chunk arrives *after*
  `finish_reason` with empty `choices`; finalizing early would lose exact
  counts.
- **Provenance convention:** `~` prefixes estimated values (char/4
  heuristics). Anything measured (durations) or provider-reported (usage,
  cost, cache/reasoning tokens) has no prefix. Preserve this when adding new
  fields.
- **Usage lookup fans out over candidates** (`_usage_candidates`): `body.usage`,
  the last assistant message's `usage`, then a depth/count-bounded recursive
  scan (`_find_usage_dicts`) for pipes that stash usage in odd places. Every
  usage-reading helper (`_extract_usage_int`, `_extract_usage_nested_int`,
  `_extract_usage_cost`) goes through this list. When adding a new usage
  field, accept **both** OpenAI-style (`prompt_tokens`, `*_tokens_details`)
  and Responses-API-style (`input_tokens`, `input_tokens_details`) names.
- **Text extraction** (`_iter_text_from_delta`) counts `content`,
  `reasoning_content`, `reasoning`, `thinking`, and `audio.transcript`. If a
  provider introduces a new delta field that carries model text, add it there.

## Style

- Match the existing code: type hints everywhere, `Mapping`/`MutableMapping`
  for read/write dict parameters, small pure helper functions at module level,
  `isinstance` guards before every dict access (bodies are untrusted and vary
  by provider).
- Comments explain *why* (protocol quirks, provider inconsistencies), not
  *what*. Keep that bar.
- Log with the module `log` at `DEBUG` for parse failures and `INFO` for
  opt-in debug output; never `print`.
- Interim status lines use full words ("First model text after …"); the final
  summary uses abbreviations (`pp`, `tg`, `tok`). Parts are joined with ` · `.

## Validating changes

There are no tests. At minimum:

```sh
python -m py_compile observability_metrics_stream.py
```

For behavioral changes, describe how you exercised the code (or state that
you could not) rather than claiming it works. Realistic verification means
pasting the file into a running Open WebUI instance and checking the status
line against a streaming OpenAI-compatible model, a non-streaming pipe, and a
provider that reports `usage` (ideally including `*_tokens_details` and
`cost`).

## Adding a new function

1. Create `<snake_case_name>.py` at the repo root with the frontmatter
   docstring (copy the key set from the existing file; set `version: 0.1`).
2. Follow the constraints above.
3. Add a row to the **Functions** table in `README.md` and a short section
   explaining what the user sees and how to configure it.
4. Add the file to the "Current functions" list in this document.
