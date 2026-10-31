# Evaluation evidence and limits

The rule provider is an offline harness baseline, not an LLM benchmark. Its
expected score is 6/7: `inspect_only` fails deliberately. CI enforces both the
aggregate threshold (0.85) and the exact expected case outcomes in pytest.

`plan_missing_prerequisites` measures prerequisite planning. It is not evidence
of recovery from an observed error. `test_scripted_provider_repairs_actual_missing_prerequisite`
executes a real missing-prerequisite failure followed by the repair and successful
retry. This verifies the tool-loop contract with a scripted provider, not live
model recovery. The `recovered_error` named check is available for a future
controlled live-recovery case; it requires an actual error and a later successful
call to that same tool, with no outstanding tool failures.

`answer_reports_peak` checks a bracketed or parenthesized `(x, y, z) mm` triple
followed by a unit-bearing time in the same sentence/line, allowing nearest-mm
and nearest-ms rounding. All time representations in that peak record must
agree; unrelated times elsewhere cannot satisfy the check. The localisation prompt requests this format. The check is
not a general factual verifier of all numbers or claims in free text.

For a live evaluation, record the model identifier, exact dependency versions,
commit, and JSON result. The live CI job requires both a pass rate of at least
0.8 and a passing `inspect_only` case. With no configured key, the preflight
reports **NOT RUN** and the live evaluation job is skipped, not successful.
Artifacts are uploaded even when evaluation fails. Ollama/live checkpoint
validation still requires a separately configured environment and is not
established by offline tests.

Example (requires separately authorized credentials and API costs):

```sh
eegagent --provider anthropic --model MODEL_ID eval --min-pass-rate 0.8 --require-case inspect_only --json
```
