# openwebui-functions

A collection of [Open WebUI](https://github.com/open-webui/open-webui) functions
(filters, pipes, actions). Each function is a single self-contained Python file
that you paste into the Open WebUI admin panel — no packaging or install step.

## Functions

| File | Type | Description |
| --- | --- | --- |
| [`observability_metrics_stream.py`](observability_metrics_stream.py) | Filter | Streams timing and token metrics (TTFT, prompt processing, generation speed, cache/reasoning tokens, cost) into the assistant message status area. |

### Observability metrics (stream)

Shows live performance metrics for every assistant response using the existing
`status` event type, so **no frontend changes** are required.

While streaming:

```
First model text after 0.84 s · prompt processing ~1200 tokens, ~1430 tokens/s
```

When the response finishes:

```
Done 12.40 s · pp 1187 tok, 1413 tok/s · tg 642 tok, 55 tok/s · cached 1024 tok · reasoning 210 tok · $0.0031
```

**Reading the output**

- `pp` — prompt processing (prefill): prompt token count and rate. The rate is
  `prompt_tokens / TTFT`, and TTFT includes network/queue time, so it is a
  lower bound on the server's real prefill speed.
- `tg` — text generation: completion token count and tokens/second measured
  from first text chunk to last.
- `~` prefix — the value is an **estimate** (`len(text) / 4` heuristic). Values
  without `~` are exact (provider-reported usage) or directly measured
  (monotonic clock).
- `cached`, `cache write`, `reasoning`, `N turns`, `N tool calls`, cost — only
  shown when the provider reports them in `usage`. Always exact.
- `(wall)` / `incl. latency` — the backend did not stream through Open WebUI's
  chunk pipeline (e.g. a manifold pipe that emits via `__event_emitter__` and
  returns a string). Only inlet→outlet wall time is measurable, so per-chunk
  throughput is not claimed.
- `(from first chunk)` — the filter was enabled mid-request and `inlet()` did
  not run, so the timer started at the first chunk instead of at submission.

**Provider compatibility**

Usage lookups accept both OpenAI Chat Completions names
(`prompt_tokens` / `completion_tokens`, `*_tokens_details`) and Responses-API
names (`input_tokens` / `output_tokens`, `input_tokens_details` /
`output_tokens_details`). Usage is searched in `body.usage`, the last assistant
message's `usage`, and finally via a bounded recursive scan of the outlet body
for pipes that stash usage in non-standard places.

**Valves**

| Valve | Default | Purpose |
| --- | --- | --- |
| `priority` | `101` | Run near the end of the filter chain so counts see normalized chunks. |
| `debug_prompt_breakdown` | `false` | Emit an extra status line at request start listing what the inlet body contains (message count, text chars, images/videos, tools JSON size, files). Useful for locating "invisible" prompt tokens injected after the filter runs (native tool schemas, RAG context, image pads). Also logged at `INFO`. |

## Installation

1. In Open WebUI go to **Admin Panel → Functions → New**.
2. Paste the contents of the function file (e.g. `observability_metrics_stream.py`).
3. Save, then enable it — either **globally**, or per model under
   **Workspace → Models → (model) → Filters**.

Requires Open WebUI **≥ 0.5.17** (the `stream()` filter hook). The only
dependency is `pydantic`, which Open WebUI already ships.

## Development

There is no build system; each file must remain importable as a standalone
module inside Open WebUI's function runtime.

```sh
python -m py_compile observability_metrics_stream.py   # syntax check
```

The frontmatter docstring at the top of each file (`title`, `id`, `version`,
`required_open_webui_version`, …) is parsed by Open WebUI — keep it intact and
bump `version` when changing behavior.

See [`AGENTS.md`](AGENTS.md) for conventions and architecture notes aimed at
contributors and coding agents.

## Credits

Originally based on work by **vigneshwarrvenkat**; maintained by
[Willian Zhang](https://github.com/Willian-Zhang).
