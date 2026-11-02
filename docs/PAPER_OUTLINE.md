# Paper outline: "EEG-Agent: tool-grounded language agents for reproducible EEG source imaging"

A working outline for an arXiv preprint (cs.AI / q-bio.NC / eess.SP). Items marked TODO need
experiments that this repository makes possible but that have not been run yet.

## Abstract (draft)

Large language models can plan multi-step scientific analyses, but free-form code generation
makes results hard to audit and easy to hallucinate. We present EEG-Agent, an agent that performs
EEG preprocessing, forward and inverse modelling, and cluster statistics by calling a small set of
typed tools over MNE-Python. State lives in an analysis session rather than in the prompt, every
numeric claim in the agent's summary is traceable to a tool result, and the same agent runs on a
frontier API model or a local open-weight model. We release an eval suite with deterministic,
state-based checks and report TODO pass rates for TODO models.

## 1. Introduction

- Problem: EEG source imaging has many steps with interacting parameters; expertise is scarce.
- LLM agents can lower the barrier, but code-writing agents produce unverifiable analyses.
- Contribution 1: a tool-grounded agent architecture where the LLM never touches data.
- Contribution 2: a provider-agnostic design evaluated on API and local models.
- Contribution 3: an open eval suite with state-based grading, including a hallucination check.

## 2. Related work

- Tool use and function calling in LLMs; agent frameworks (ReAct-style loops).
- LLMs for scientific workflows and data analysis (code-interpreter style agents).
- EEG analysis software (MNE-Python, EEGLAB, FieldTrip, Brainstorm) and automation efforts.
- EEG foundation models (LaBraM, EEGPT, CBraMod, REVE) as future tools.

## 3. System

- 3.1 Analysis library (`eegsi`): what each stage computes; sphere vs fsaverage head models.
- 3.2 Session state and tool schemas (table of the 13 tools with parameters).
- 3.3 Agent loop, error-as-data recovery, step budget.
- 3.4 Providers: Anthropic Messages API with native tool use; Ollama chat API with tool calling.
- 3.5 Service layer: HTTP API and UI, Docker.

## 4. Evaluation

- 4.1 Eval cases (7 today; extend to ~30 covering parameter extraction, recovery, refusal to
  fabricate, multi-turn follow-ups).
- 4.2 Metrics: task pass rate, tool-order correctness, steps per task, tokens per task,
  hallucination rate (answer numbers vs tool outputs), wall-clock time.
- 4.3 Models: TODO Claude Opus 5.5 and Sonnet 5.5; TODO Qwen3 8B / 32B and Llama 3.x via Ollama.
- 4.4 Ablations: without `describe_state`; without error-as-data (raise instead); with a reduced
  system prompt; local model with and without few-shot tool examples.
- 4.5 Real data: MNE sample dataset (auditory/visual) and the lab's MRCP recordings; compare
  agent-chosen pipelines against the published manual pipeline (legacy scripts).

## 5. Results (TODO)

Tables: pass rate per model per case; cost and latency; failure taxonomy (wrong parameter,
skipped step, fabricated number, gave up).

## 6. Discussion

- Why state-based grading matters; limits of keyword grading.
- Small local models: where they fail (schema adherence, long tool outputs).
- Safety: the agent cannot run arbitrary code; what it still can get wrong (parameter choice).
- Toward agents that call EEG foundation models as tools (`eegmodels`).

## 7. Reproducibility

- Everything runs offline on synthetic data; CI runs the suite on each push.
- Exact model ids, prompts, tool schemas and eval cases are in the repository.

## Figures

1. Architecture diagram (docs/ARCHITECTURE.md).
2. Example transcript with tool trace and the produced report.
3. Pass rate per case per model (bar chart).
4. Failure taxonomy (stacked bars).
