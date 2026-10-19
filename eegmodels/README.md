# eegmodels

One registry, one downloader, one loader for open-weight EEG foundation models,
built to plug straight into MNE-Python objects.

```bash
pip install -e ".[braindecode,mne]"   # from the repository root

eegmodels list                        # what is in the registry
eegmodels info steegformer-base       # source, license, input contract
eegmodels pull steegformer-base       # download into ~/.cache/eegmodels
eegmodels embed steegformer-base sub-01_eeg.vhdr -o sub-01.npy
```

```python
import mne, eegmodels

raw = mne.io.read_raw("sub-01_eeg.vhdr", preload=True)
model = eegmodels.load("steegformer-base")
z = model.embed(raw)           # (n_windows, ...) embeddings
```

`embed()` accepts an `mne.io.Raw`, an `mne.Epochs`, or a NumPy array
(`(n_channels, n_times)` or `(n_epochs, n_channels, n_times)` with `sfreq=`, and
optionally `ch_names=` and `units=`). The adapter enforces the model's input contract
from the registry: unit conversion, channel selection and ordering, resampling,
amplitude clipping and windowing. You never touch tensors.

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
   `adapter.kwargs.model_class` and you are done.
3. Otherwise add a class in `adapters/`, register it in `adapters/__init__.py`, and point the
   entry at it. Only `load_weights()` and `forward()` are required.
4. Pin `source.revision` to a commit hash once you have validated the download.

## Registry status

| Status | Meaning |
| --- | --- |
| `available` | Downloads and loads. |
| `gated` | Downloads after you request access on the model page and run `huggingface-cli login`. |
| `planned` | Entry exists for the input contract and references; adapter not written yet. `pull` refuses it. |

Entries marked `VERIFY` in their YAML carry sampling rate, window or channel values taken
from papers rather than confirmed against the released checkpoint. Check the model card
before trusting them.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite needs no network and no model weights: URL downloads are exercised against a
local HTTP server and adapters are tested with an echo model.
