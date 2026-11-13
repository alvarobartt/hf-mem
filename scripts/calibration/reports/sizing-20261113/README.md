# Capacity recommendations and defaults, 2026-11-13

The headline GPU capacity includes the whole allocation inside a 90% budget, plus any separate initialization allowance and native engine minimum. Runtime `total` / `fit.required` remains a distinct allocation estimate. A native server can reserve spare VRAM as additional cache; observed use does not establish the minimum capacity required by the workload.

23 prospective configurations across vLLM, SGLang and llama.cpp have zero current fit/no-fit mismatches, including the hardware-unhinted recommendations. The 18 complete controlled allocation comparisons have 0.85% median absolute error and 5.22% maximum error. Four native-default vLLM/SGLang comparisons have 0.30% median and 1.50% maximum non-KV error; their automatic spare-cache reservation is excluded from allocation-error scoring.

| Engine | Fit checks | Controlled positive allocations | Median absolute allocation error | Maximum |
| --- | ---: | ---: | ---: | ---: |
| vllm | 6 | 4 | 1.37% | 3.26% |
| sglang | 6 | 4 | 0.58% | 1.00% |
| llama.cpp | 11 | 10 | 1.53% | 5.22% |

## What changed

- The physical recommendation and fit budget include CUDA initialization as well as weights, cache and workspaces. A model ID alone selects a fitting capacity class with the matching scheduler/prefill defaults. Inkling 2 selects its calibrated SGLang reference automatically; GGUF selects llama.cpp.
- Inkling storage includes final MXFP8 scale expansion, replicated encoder/router weights, the automatic ShortConv/radix state pool and prefill graph sharing. A storage audit follows packed unregistered Marlin tensors and independently measures request/state pools instead of calling missing cache “overhead.” NVFP4 final FP8 scales are not doubled.
- GGUF compute follows context and hidden/FFN dimensions. Runtime capacity classes distinguish T4, L4 and H200. Full sliding-window storage, quantized caches and two-sequence context pools are checked with native automatic fitting disabled.
- vLLM language-only mode excludes identified encoder weights from steady-state storage but includes a conservative, encoder-sized initialization allowance in physical capacity. The observation shows a cold-start peak; it does not prove those exact omitted tensors account for it. SGLang's unsupported Qwen3.5 standalone language-only launch is rejected by hf-mem, matching the pinned native engine.

## Prospective checks and exclusions

Forecasts and estimator/executed source hashes were frozen locally before Jobs submission. `job-verification.json` compares the actual Jobs API payload with those source bytes, immutable images, flavors and registration timestamps. These are local freezes, not public preregistrations. All scored raw allocation forecasts still replay byte for byte. Capacity-policy changes are evaluated separately: whole-allocation headroom was revised after the first H200 full-window GGUF result, and the language-only initialization allowance after the 2B result. The bounded GGUF run and subsequent 4B language-only run were frozen after those respective corrections.

Fresh Inkling holdouts change context, prefill and scheduler capacity from the calibration runs. TP=EP4 at 131,072 context / 4,096 prefill predicts 407.90 GiB versus 410.76 GiB sampled runtime (−0.70%); TP=EP8 at 524,288 / 4,096 predicts 487.83 versus 488.79 GiB (−0.20%). Only the attention pool is bounded to the workload; native convolution/radix allocation, FA4 and optimizer defaults remain enabled. Every rank's packed weights and auxiliary pools are audited. Earlier “non-cache” Inkling comparisons subtract attention KV only and therefore still include convolution states; they are not pure residual-overhead scores.

Default checks exercise full native configured context. Batch checks send two distinct full-length prompts and verify per-sequence prompt/decode counts. A server becoming healthy is insufficient: the SGLang T4 4B development case starts but refuses the requested context and is retained as a negative result. The independent GGUF 3B full-context T4 allocation also fails as predicted. One L4 language-only holdout was canceled before execution after more than 20 minutes of hardware unavailability and moved to H200; both registrations and the canceled attempt remain in the report. Only a verified Jobs API `CANCELED` status with a recorded null start time qualifies as an unrun diagnostic. Failed hook assertions are instrumentation failures, unsupported flags are unsupported options, and an HTTP timeout is inconclusive; none is recast as an OOM or silently dropped. Corrected batch-hook reruns retain the original failed records.

All current controlled recommendations cover sampled cold-start and runtime peaks. A direct H200 memory-limit test reserves otherwise spare VRAM with an independently audited CUDA allocation, leaving 3.4718 GiB free against the frozen 3.4720 GiB recommendation. llama.cpp completes 65,528 prompt tokens and eight decode tokens, with full sliding-window cache and no automatic fit/offload. A separate run at the previous 3.1248 GiB budget fails during startup. The reservation process is explicitly declared, all other foreign GPU processes are forbidden, and its constant baseline is subtracted. This tests the actual recommendation boundary rather than inferring a minimum from success on an almost empty H200.

The earlier 150 complete cross-engine configurations retain exactly the same weights, allocation and overhead bytes. Their retrospective 0.81% median total error includes calibration data; it is separate from the new prospective scores.

## Evidence and reproduction

- `sizing-evaluation.json` and `observations.csv` retain every classified outcome, original forecasts, current predictions and individual errors. `prior-regression.json` retains the 150-case replay.
- Each experiment includes its pinned specification, original forecast, Job handle, exact executed source and collected summary. `frozen-source/` contains the original estimator snapshots; `*registration.json` preserves the pre-submission protocols. Source snapshots are excluded from formatters to preserve their hashes.
- `public/native.log`, where present, contains clearly labeled diagnostic excerpts, not the full native log. Full raw NVML traces and complete logs remain local under `/tmp/hf-mem-default-calibration-20261113`; the collection summaries retain their integrity/completeness checks. The Jobs links in `job-verification.json` identify the executed runs. No new bucket upload is performed.

Replay predictions and the classification summary with Hub access (model headers only):

```bash
PYTHONPATH=src uv run python scripts/calibration/evaluate_sizing.py --directory scripts/calibration/reports/sizing-20261113
```

Recheck exact executed Job payloads without exporting environment variables or secrets:

```bash
uv run --with huggingface_hub python scripts/calibration/verify_jobs.py --directory scripts/calibration/reports/sizing-20261113
```

Preview any stored experiment with `matrix_run.py`, `llamacpp_run.py`, or `native_run.py --directory EXPERIMENT`. To launch a fresh paid run, copy its inputs to a new directory without `job.json` and add `--launch`; runners refuse to relaunch an existing handle. Collectors can re-audit the full retained logs/raw samples; compact checked-in summaries alone cannot reproduce the raw-trace audit. `sha256.json` lists the packaged artifact hashes.

These results concern the pinned engine images, models, workloads and devices. Sampled peaks are lower bounds at a target 10 ms interval; multi-GPU readings are sequential within each aggregate sample. No single sampled peak establishes a universal minimum. Hardware-unspecified sizing uses a capacity-based profile, not GPU autodetection. H100 remains unmeasured, and this does not validate arbitrary MIG, CPU/Metal, multimodal inputs, speculative decoding or non-default optimizer combinations. The 90% budget is hf-mem's sizing policy, not an alias for SGLang's static fraction or a claim about all engines' native defaults.
