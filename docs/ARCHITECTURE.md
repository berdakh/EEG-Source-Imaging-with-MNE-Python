# Architecture

```
 user ──▶ CLI / Streamlit UI ──▶ FastAPI  ──▶ Agent loop ──▶ Provider (Claude | Ollama | rule)
                                               │   ▲
                                               ▼   │ tool results (JSON)
                                            Tools (typed, schema-described)
                                               │
                                               ▼
                                            Session state  ◀──▶  eegsi (MNE-Python)
                                            raw, epochs, evoked, cov, fwd, inv, stc, results, log
```

## Packages

| Package | Responsibility | Depends on |
| --- | --- | --- |
| `eegsi` | Pure analysis functions: filtering, bad channels, epoching, ICA, sphere/fsaverage forward models, inverse solutions, cluster statistics, HTML reports. Takes and returns MNE objects. | MNE, NumPy, SciPy |
| `eegagent` | The agent: session state, tool definitions, provider adapters, the loop, evals, API, UI. | `eegsi`, anthropic SDK, httpx, FastAPI |
| `eegmodels` | Registry-driven downloader and loader for open-weight EEG foundation models. Independent of the agent today; the plan is to expose `embed` as a tool. | huggingface_hub, pooch |

## Design decisions

**Tools are thin wrappers over a library.** Every tool is a few lines that validate preconditions,
call one `eegsi` function and record a log line. The analysis logic is testable without any LLM,
and the LLM never touches arrays.

**State lives in a `Session`, not in the conversation.** Tools read and write the session, so a
tool result can be small (a few numbers) while the objects stay in memory. `describe_state` lets
the model re-orient itself at any time, which matters for long multi-turn sessions and for
smaller local models.

**Errors are data.** A tool failure is returned as `{"error": ...}` with `is_error=True`, not
raised. The model reads it and recovers, typically by running the missing earlier step. The eval
case `recover_from_missing_step` tests exactly this.

**Provider-neutral history.** The loop stores messages in a small neutral format and each
provider converts to its native wire format. The Anthropic adapter keeps the native assistant
content (`raw`) so thinking blocks are replayed unchanged, as the API requires; Ollama's
adapter maps the same history onto its `tool_calls` / `tool` roles. Adding a provider is one
file with one method.

**Deterministic evals.** Cases are prompts plus checks that look at the tool trace and the final
session state, not at the wording of the answer. One check (`answer_reports_peak`) verifies that
the numbers in the prose match the tool output, which catches hallucinated results. The `rule`
provider runs the whole harness offline in CI; the real model runs when an API key is present.

**No MRI required.** The default head model is a 4-shell sphere with a volume source grid, so the
complete pipeline, including source localisation, runs on any machine in seconds. The
`fsaverage` option gives a proper BEM on the FreeSurfer template when the download is possible.

## Request flow for one user turn

1. `Agent.run(text)` appends the user message and calls `provider.complete(system, history, tools)`.
2. The provider returns a `Turn` with text and zero or more `ToolCall`s.
3. Each call is executed by `tools.run_tool(session, name, args)` in order; results are JSON strings.
4. All results of that turn are appended as one `tool` message (one user message with all
   `tool_result` blocks for Anthropic, which keeps parallel tool use working).
5. Repeat until the model stops calling tools or `max_steps` is reached, in which case the model
   is asked once more, without tools, for a summary.

## Extending

- **New tool:** decorate a function in `eegagent/tools.py` with `@tool(name, description, schema)`.
  It appears in every provider automatically and in `GET /tools`.
- **New analysis:** add a function to `eegsi`, test it on `eegsi.data.simulate_raw()`, then wrap it.
- **New eval case:** add an entry to `eegagent/evals/cases.yaml`; add a check function to
  `runner.py` if the existing ones do not cover it.
