# eegmodels

One registry, one downloader, one loader for open-weight EEG foundation models,
built to plug into MNE-Python objects once each checkpoint contract is validated.

> **Validation hold:** all eight bundled entries currently block inference. The
> corrected metadata exposes unresolved preprocessing/montage contracts rather
> than treating guesses as runnable defaults. `load`, `embed` and `predict` fail
> with an explanation before downloading/loading weights. This is an intentional
> behavior reduction, not completed pretrained-model support. See
> [MODEL_VALIDATION.md](../docs/MODEL_VALIDATION.md) for per-entry requirements.
> The six previously available entries affected are `steegformer-base`,
> `steegformer-large`, `steegformer-largev2`, `mirepnet`, `brainbert` and
> `mvpformer-s`. REVE and LaBraM still lack adapters; REVE now correctly reports
> `planned` rather than implying that access alone enables inference.
>
> `pull` remains separate for the six available entries. It does not certify
> inference compatibility. No model weights or live-provider calls were used to
> test this patch.

```bash
pip install -e ".[braindecode]"   # from the repository root; MNE is a core dependency

eegmodels list                        # what is in the registry
eegmodels info steegformer-base       # source, license, input contract
eegmodels pull steegformer-base       # download into ~/.cache/eegmodels
# Inference below remains blocked until the selected contract is validated:
# eegmodels embed steegformer-base sub-01_eeg.vhdr -o sub-01.npy
```

```python
import mne, eegmodels

raw = mne.io.read_raw("sub-01_eeg.vhdr", preload=True)
model = eegmodels.load("steegformer-base")  # currently raises AdapterError: unvalidated contract
z = model.embed(raw)  # (n_windows, ...) embeddings
```

`embed()` accepts an `mne.io.Raw`, an `mne.Epochs`, or a NumPy array
(`(n_channels, n_times)` or `(n_epochs, n_channels, n_times)` with `sfreq=`, and
optionally `ch_names=` and `units=`). The adapter enforces the model's input contract
from the registry: unit conversion, channel selection and ordering, resampling,
amplitude clipping and windowing. Model-specific filtering, rereferencing and
alignment listed in the registry are not performed by generic preparation.
`segment_seconds` and `channel_vocab_size` are descriptive metadata, not windowing
or montage implementations. Do not set `contract_verified` to true without implementing
and testing those requirements.

The Braindecode adapter explicitly requests `return_features=True` and returns
the `features` tensor as an array. Output may be token-wise, depending on the
architecture; it is not automatically pooled. Incompatible runtimes or a logits
response fail clearly rather than silently becoming embeddings. `predict` is
not supported without a separate validated task head and label mapping.

## Layout

| Path | Role |
| --- | --- |
| `registry/*.yaml` | One file per model: source, revision, license, input contract, adapter. |
| `registry.py` | Schema validation and lookup. Extra registry dirs via `EEGMODELS_REGISTRY`. |
| `download.py` | `pull()`: Hugging Face snapshots via `huggingface_hub`, direct URLs via `pooch` with SHA256 checks. Cache at `$EEGMODELS_CACHE` or `~/.cache/eegmodels`. |
| `adapters/base.py` | `BaseAdapter`: shared input-contract enforcement. Subclasses implement `load_weights()` and `forward()`. |
| `adapters/braindecode.py` | Loads any braindecode zoo model through `from_pretrained`. |
| `api.py` | `load(name)`. |
| `cli.py` | `eegmodels list / info / pull / embed`. |

## Adding a model

1. Copy an entry in `registry/` and edit it. The schema is documented at the top of `registry.py`.
2. If the model loads through braindecode, set `adapter.class: braindecode` and
   `adapter.kwargs.model_class`. Validate its preprocessing/montage and required
   `return_features` API against pinned checkpoint/runtime versions. Only set
   `input.contract_verified: true` after the acceptance checks in the validation
   document pass. Existing custom registries must explicitly declare this field too; omission
   defaults to false. A declaration is the registry author’s assertion, not an
   automated scientific validation.
3. Otherwise add a class in `adapters/`, register it in `adapters/__init__.py`, and point the
   entry at it. Only `load_weights()` and `forward()` are required.
4. Pin `source.revision` to a commit hash once you have validated the download.

## Registry status

| Status | Meaning |
| --- | --- |
| `available` | Source is downloadable; inference additionally requires a validated contract. |
| `gated` | Downloads after you request access on the model page and run `huggingface-cli login`. |
| `planned` | Entry exists for the input contract and references; adapter not written yet. `pull` refuses it. |

Unknown rates/windows are null, not paper-derived guesses. `contract_verified`
keeps published metadata separate from executable validation. The current broad
optional dependency constraint is not a compatibility guarantee; actual compatible
versions and checkpoint hashes must be recorded before any entry is enabled.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite needs no network and no model weights: URL downloads are exercised against a
local HTTP server and adapters are tested with an echo model.
