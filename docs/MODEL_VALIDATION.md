# EEG model contract validation

This change separates **published metadata** from a **validated executable input
contract**. Sources below were inspected on 2026-10-31. No checkpoint was downloaded
or executed as part of the metadata audit. Offline adapter regression tests are
not evidence of pretrained-weight compatibility or scientific accuracy.

## Safe behavior

The bundled registry entries have `input.contract_verified: false`. Inference is
blocked until their remaining requirements are implemented and tested. Downloading
a checkpoint is distinct from validating it. Do not set this flag merely because a
model card is reachable or a mock-based test passes.

`channel_vocab_size` is the number of electrode identity slots in a model's
vocabulary, **not** a maximum supported channel count. `segment_seconds` describes
a tokenizer segment/patch; `window_seconds` describes a complete input example.
A null window does not establish arbitrary-length support. A null sampling rate
means the rate is unknown, not that automatic resampling is safe. Preprocessing
requirements are declarative evidence, not implemented transformations.

## Registry reconciliation

| Entry | Confirmed metadata | Remaining validation |
| --- | --- | --- |
| STEEGFormer base / large | Published port vocabulary has 145 slots, not 128. | Sampling rate, full input duration, channel-to-slot mapping, checkpoint-compatible construction, feature output. The base config's 22 channels and 1000 samples do not establish a universal rate/window contract. |
| STEEGFormer largeV2 | Published vocabulary has 256 slots. | Rate/window remain unknown. Current implementation says the slot electrode names are unpublished; requires a verified explicit `chan_pos_idx`, not an assumed identity mapping. |
| MIRepNet | 250 Hz, 1000 samples / 4 s, exact ordered 45-channel template, 256-dimensional encoder features. Port identifies source/checkpoint license as MIT. | Implement/validate 8–30 Hz filtering and Euclidean alignment outside the model; load/feature smoke test. Do not attach semantic labels to its three-output head: ordering is undocumented. |
| BrainBERT | 2048 Hz. Port consumes raw batch × channels × time, does STFT internally and pools channels. | Laplacian rereferencing and minimum input length for the pinned STFT settings; feature smoke test. No mandatory five-second window or single-channel cap is established. |
| MVPFormer-S | Published segmentation is 5 s / 2560 samples at 512 Hz. | Resolve port checkpoint full context. Original IBM S config uses 25 segments; do not treat a segment as the full window. Verify feature output rather than its random classification head. |
| REVE base | Official repository exists; 200 Hz, variable lengths and electrode layouts, electrode positions required. REVE Responsible Use License v1.0. | Custom adapter and position input not implemented; access status not tested. Remains planned. |
| LaBraM base | 200 Hz, microvolt input recipe, 200-sample / 1 s patches; upstream fine-tuning divides amplitudes by 100. | Real checkpoint URL/hash and adapter remain missing. Confirm task/context length and filtering; division is not amplitude clipping. Remains planned. |

## Primary sources

- STEEGFormer [base card](https://huggingface.co/braindecode/STEEGFormer-base),
  [base config](https://huggingface.co/braindecode/STEEGFormer-base/blob/main/config.json),
  [large card](https://huggingface.co/braindecode/STEEGFormer-large),
  [largeV2 card](https://huggingface.co/braindecode/STEEGFormer-largeV2), and
  [current montage mapping implementation](https://github.com/braindecode/braindecode/blob/master/braindecode/models/steegformer.py).
- MIRepNet [port card](https://huggingface.co/braindecode/mirepnet-pretrained) and
  [ordered channel list](https://huggingface.co/braindecode/mirepnet-pretrained/blob/main/mirepnet_channels.json).
- BrainBERT [port card](https://huggingface.co/braindecode/brainbert-pretrained) and
  [port implementation](https://github.com/braindecode/braindecode/blob/master/braindecode/models/brainbert.py).
- MVPFormer [port card](https://huggingface.co/braindecode/mvpformer-s-pretrained),
  [port implementation](https://github.com/braindecode/braindecode/blob/master/braindecode/models/mvpformer.py),
  [IBM S configuration](https://github.com/IBM/multi-variate-parallel-transformer/blob/main/configs/mvpformer_s_generative.yaml),
  [dataset time conversion](https://github.com/IBM/multi-variate-parallel-transformer/blob/main/eeg_datasets/longterm.py).
- REVE [official model card](https://huggingface.co/brain-bzh/reve-base) and
  [license](https://huggingface.co/brain-bzh/reve-base/blob/main/LICENSE).
- LaBraM [official instructions](https://github.com/935963004/LaBraM#running-experiments)
  and [fine-tuning implementation](https://github.com/935963004/LaBraM/blob/main/engine_for_finetuning.py).

These URLs may track changing upstream branches. Their current contents and model
card statements must be distinguished from immutable checkpoint evidence.

## Checklist before enabling a model

1. Pin the Hugging Face source to an immutable commit and record downloaded file
   hashes. The existing `revision: main` values are explicitly **not** a
   reproducibility guarantee. Do not invent hashes or reuse a hash from a different
   upstream checkpoint as the port's hash.
2. Pin a compatible Braindecode and Torch environment. Verify required class,
   `from_pretrained` and encoder-feature APIs exist in that version. Do not assume
   every model is supported by the project's broad `braindecode>=1.0` constraint.
3. Resolve signal rate, units, channel identities/order, rereferencing, filtering,
   normalization/alignment, patching and full context from the selected release.
   Implement missing transformations or explicitly require already-processed
   inputs, with validation. Do not silently reconstruct missing channels.
4. Load real weights and run a finite-output smoke test using a documented,
   non-sensitive synthetic or approved fixture. Record input shape, rate, units,
   ordered channel names/position mapping and output feature dimensions. Confirm
   embeddings are encoder features rather than classifier logits.
5. Test incorrect rates, lengths, missing/duplicate channels and mapping errors.
   Compare to the pinned upstream implementation where possible. Test both the
   intended batch shape and a second batch size.
6. Only then update `contract_verified` and preserve the exact versions, hashes,
   fixture provenance and results in a reviewable validation record. Passing this
   engineering check does not establish clinical validity or downstream accuracy.
