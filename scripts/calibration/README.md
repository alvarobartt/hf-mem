# Runtime overhead calibration

The expanded cross-engine work below supersedes the original scheduler-only and capacity-tier coefficients. The older sections preserve the experiments and claims made for those source snapshots; their reported numbers are historical, not predictions from the current estimator.

The November 13 CLI follow-up adds native engine argument spellings, canonical `nvidia-*` GPU names, board-capacity validation, and a distinct `--language-model-only` mode that excludes identified encoder/projector weights. The V1–V5 source snapshots and reports below remain immutable historical records. Their `text_only=True` measurements retain encoder weights; they do **not** validate the new encoder-unloaded mode. The follow-up tests exercise that mode with exact synthetic tensor headers and GGUF projectors. The older source bundle awaiting public-upload authorization is unchanged; it does not silently acquire these later source changes.

## Capacity recommendations and defaults, 2026-11-13

The new [sizing report](reports/sizing-20261113/README.md) separates physical GPU capacity from runtime allocations and native spare-cache reservations. It validates model-ID defaults and explicit context, batching, cache and language-only settings against pinned vLLM, SGLang and llama.cpp images on T4, L4, L40S and H200, including actual Inkling TP=EP4/8. The headline recommendation includes whole-allocation headroom and any separate initialization allowance; an optional hardware hint refines the profile.

Fresh Inkling allocation errors are −0.70% and −0.20%. The report retains every original forecast and failed trial, native packed-weight and auxiliary-state audits, a constrained-memory recommendation check, and the unchanged 150-case earlier allocation replay. Its combined scores, prospective/calibration distinction and reproduction commands are in the linked report. The older Inkling section below preserves the original automatic-cache comparison; its “non-cache” totals still include convolution states and are not pure overhead scores.

## Inkling 2 Small native validation, 2026-11-13

Two Hub Jobs ran the requested `thinkingmachines/Inkling-2-Small` revision `aac974122acd33ddb0d3e1f54433a9c5dd2249ce` with `lmsysorg/sglang:dev-inkling-2-small@sha256:424c1da97f6d6eef1612006c77b2ca791657a8184c1fbdbc682ebf9485d3b32c`. Both use TP=EP, FA4, BF16 activations, page size 128, xgrammar, multimodal enabled, and the requested reasoning/tool parsers. The four-GPU server has context 262,144 and 64 running-request slots; the eight-GPU server has context 1,048,576 with the native scheduler default. No static-memory fraction, radix-cache setting or token-pool limit was overridden. Hub Jobs runs the Python server entry point directly and manages GPU visibility/networking; its available shared-memory capacity was about 2 TiB rather than an enforced Docker 16 GiB cap.

| GPUs / configured context | Original frozen hf-mem minimum | Native sampled runtime peak | Native sampled startup peak | Native reserved attention cache |
| --- | ---: | ---: | ---: | ---: |
| 4 / 262,144 | 423.95 GiB | 505.87 GiB | 509.82 GiB | 93.04 GiB |
| 8 / 1,048,576 | 651.19 GiB | 1,016.21 GiB | 1,026.26 GiB | 508.71 GiB |

The fit predictions agree with actual successful deployments. All three text probes per server complete with zero prefix-cache hits and eight generated tokens. The longest prompts contain exactly 258,048 and 1,044,480 tokens, respectively; these are near the configured limits, leaving admission headroom. Both servers retain the multimodal encoder, but these requests do not establish maximum-image activation peaks. Startup and runtime peaks are separate observations and must not be added. The values above aggregate every device in each NVML sampling iteration and subtract an empty-device baseline. GPU reads are sequential; the 10 ms target sampler yields lower bounds.

The original forecasts and source were frozen locally before both submissions and remain unchanged. Their raw differences from these reserved-server footprints are −16.19% and −35.92%. This is **not** a like-for-like total-memory accuracy score: native SGLang reserves a large KV pool from remaining VRAM, while hf-mem estimates the requested workload's cache. Subtracting each side's own attention-cache amount leaves the frozen **non-cache total** estimates 8.93% and 9.52% below measurement. Those remaining gaps include custom quantized storage, allocation/graph workspaces and other state; they are not pure overhead errors.

The deployment exposed two metadata issues, now fixed: `model_max_length` supplies the default context, and `local_layer_ids` / `sliding_window_size` identify the native eight full-attention and forty local-attention layers. Retrospective forecasts after this parsing correction are 386.61 / 493.85 GiB, with 10.66 / 34.66 GiB of logical attention KV, and still predict both configurations fit. These revised values were computed **after** the native results and are not new holdouts or a fit to the automatic reservation. No backend coefficients were tuned; the earlier 150 complete native configurations retain exactly the same numerical estimates.

The custom image reports SGLang `0.0.0.dev1+g797af7170.d20261110`, rather than the older calibration image. Its `inkling2` precision loader keeps most packed expert weights outside named PyTorch parameters/buffers. The ordinary parameter audit therefore sees only a small fraction of their storage. Residual-overhead scores are omitted, rather than treating these missing weights as overhead. The current estimator also does not fully describe this custom model's packed-expert routing, short-convolution state and FA4 prefill-graph workspaces. The separate `mtp.safetensors` file is already excluded by ordinary main-model discovery; no extra MTP subtraction was introduced.

[4-GPU Job](https://huggingface.co/jobs/alvarobartt/6af746dd29dd05ff036543e3), [8-GPU Job](https://huggingface.co/jobs/alvarobartt/6af7471029dd05ff03654438), [comparison JSON](reports/inkling2-20261113/comparison.json), [CSV](reports/inkling2-20261113/comparison.csv), and [frozen forecasts](reports/inkling2-20261113/original-forecasts.json) retain the complete settings and exclusions. Full sanitized logs, native sources, source snapshots and raw traces are local in `/tmp/hf-mem-inkling-validation-20261113`. No new public bucket upload is authorized.

Reproduce offline collection with `python scripts/calibration/native_collect.py --directory EXPERIMENT --log-file EXPERIMENT/live-private.log`; compare the stored forecasts with `python scripts/calibration/evaluate_native.py --directory ROOT`. `native_run.py --directory EXPERIMENT` previews and verifies the executed image/model pins, TP/EP controls and source hashes. Adding `--launch` submits paid compute and refuses an existing Job handle. Collection verifies complete device traces, rank startup, exact declared prompt/decode counts and raw sample accounting; it rejects an incomplete registered-weight audit as a basis for residual-overhead scoring.

## Expanded cross-engine calibration, 2026-11-12

The estimator accepts `--engine vllm|sglang` and `--gpu-type`. The GPU family distinguishes initialization, attention backend, scheduler, prefill and collective-buffer defaults that cannot be inferred reliably from VRAM capacity. Actual TP runs cover PCIe A10G and NVLink A100/H200, alongside T4, L4, L40S and RTX PRO 6000 single-GPU checks. Engine images are pinned by digest:

- vLLM 0.33.0: `vllm/vllm-openai@sha256:91ee972ab751fb7180bc71c175865e36b091686600dca783d52b2511399810b3`.
- SGLang 0.5.23: `lmsysorg/sglang@sha256:f7876bef696ccf9cf75a2bf7ccb82970f5fac577f082fb8ca95c9a59d2160eb5`.

`matrix_collect.py` requires every rank's initialization, model, cache and graph evidence; independent resolved controls; an empty process baseline; all requested prompt/decode tokens; and complete raw sample accounting. NVML aggregate peaks use device readings from the same sampling iteration. Stage peaks are not added. Two sequential prompts with different token IDs share a warmed server and are not independent deployment replicates. The sampler's 10 ms target interval makes observed peaks lower bounds.

Candidate V1 was fitted on 21 reference configurations, then checked on separate models/settings. Its A100 TP and small-model native-default misses led to V2. Original V1 forecasts remain unchanged. Both candidates' source and forecasts were frozen **locally** before their respective validation Jobs. New public source/forecast publication was blocked by automatic approval review and remains pending explicit approval; the local freeze is not a public preregistration. Earlier published baseline experiments retain their original public artifacts.

V2 separates request slots from graph-capture capacity, uses native hardware/topology prefill and graph limits, counts retained float32 logits by geometry, and uses measured per-engine initialization/collective reserves. Expanded query projections follow head count and head dimension instead of hidden size alone. Cross-shard tied dense embedding heads and ordinary Qwen3.5 MTP exclusions correct checkpoint-to-runtime weight discrepancies independently of overhead. SGLang GDN state slots include their padding slot and shard independently from attention KV heads; a TP count larger than the KV-head count must not duplicate all recurrent states. Spatially merged image reserves are bounded by the context that could admit the image.

SGLang's automatic recurrent-state/radix pool consumes a fraction of available VRAM; this is distinct from minimum workload storage. For an explicit scheduler capacity, the dedicated-state estimate matches the controlled `--disable-radix-cache` configuration. Without it, only the requested batch's minimum state is included and a warning identifies the automatic reservation. H100 initialization remains an interpolation; measured RTX PRO 6000 behavior does not validate all Blackwell GPUs. Packed GPT-OSS storage remains an explicit limitation of the older SGLang parameter audit, so that cohort's residual-overhead score is omitted.

V2's fresh cohort exposed larger MoE and multimodal tails despite low dense-model medians. V3 separates GDN decoder scratch from encoder activations, adds measured MoE collective buffers, accounts for head-dimension-dependent library workspaces, and removes the obsolete unhinted-hardware branch. It adds `--text-only`, retaining loaded multimodal weights. A paired 3,520×3,520 image produces 12,100 image tokens plus the prompt in both engines; the image raises the warmed-server peak by about 1.7–1.8 GiB. This is training/diagnostic evidence for V3, not its holdout result. Fresh V3 checks use a different GDN model, MLA/MoE transfer, TP8 MoE, a smaller real image on H200, and fresh T4 dimensions/settings.

All 20 V3 configurations passed native completeness gates. Its median absolute total/overhead errors were 0.99%/3.48% for vLLM and 2.69%/8.18% for SGLang. The largest SGLang GDN tail was still 25.8% total and 40.4% overhead error, so V3 was not treated as the final result. V4 separates the linear-attention prefill path from full-attention FA3, distinguishes Triton key-side scratch from recurrent value-state storage, and models the additional GDN collective reserve without double-counting the MoE reserve. It also preserves the original positional `Estimate(..., warnings)` argument. Fresh V4 checks cover larger contexts, TP2/8, a larger 9B GDN shape and a 4,032×4,032 actual image, on both engines.

V4's SGLang Hopper results exposed an overestimated decoder workspace before the image was processed. V5 bounds the empirical GDN prefill scratch reserve in the observed SM90 fused-kernel domain (float32 recurrent state, 128-dimensional key/value heads, no KDA configuration, up to 32,768 prefill tokens). This is a measured reserve, not a claim that the kernel allocates exactly 512 MiB. Three fresh V5 SGLang cases transfer the correction to different 4B/9B text settings and a real large image. The other eleven concurrent V4 forecasts are byte-for-byte equivalent under V5; those outcomes were not used to choose V5. The three V4 SGLang Hopper tuning cases are excluded from the final independent cohort.

The final independent check contains seven cases per engine. Source and forecasts were frozen locally before the corresponding Jobs; exact equivalence on the eleven unchanged paths is retained in the report. The previous estimator is the immutable source snapshot from the start of this expanded calibration.

| Engine | Cases | Median absolute total error, previous → final | Median absolute residual-overhead error, previous → final | Worst final total / overhead error |
| --- | ---: | ---: | ---: | ---: |
| vLLM | 7 | 22.67% → 2.12% | 59.08% → 4.05% | 6.35% / 14.06% |
| SGLang | 7 | 1.98% → 1.77% | 4.87% → 6.76% | 4.49% / 10.38% |
| Combined | 14 | 7.37% → 1.95% | 40.15% → 5.55% | 6.35% / 14.06% |

SGLang's median overhead tradeoff is retained: its worst overhead error improves from 48.11% to 10.38%. Combined P90 total/overhead errors are 4.49%/11.88%. These results meet the locally frozen targets (median total <3%, median overhead <10%, P90 overhead <25%); they do not establish universal accuracy.

The final-source **retrospective** replay covers 150 configurations from complete native datasets: 69 vLLM and 81 SGLang, thirteen checkpoints, seven GPU families (T4, A10G, L4, L40S, A100, H200 and RTX PRO 6000), and actual TP1/2/4/8. Its median total/overhead errors are 0.81%/3.04%, P90 3.11%/8.60%, and maxima 6.35%/20.26%. This includes training and older validation. One otherwise usable V2 TP4 row is excluded from these primary metrics because its dataset lost the earlier log prefix. The T4 startup timeout remains a diagnostic. No missing evidence was reconstructed. A separate legacy eight-model API replay gives median total errors of 0.46% (vLLM) and 1.10% (SGLang); its older SGLang context-boundary requests do not meet the current exact eight-decode-token protocol.

The machine-readable [summary](reports/crossengine-20261112/summary.json), [final independent observations](reports/crossengine-20261112/final-independent-validation.csv), [retrospective coverage](reports/crossengine-20261112/final-retrospective-coverage.json), and [versioned protocols](reports/crossengine-20261112/protocols) retain denominators, exclusions, original misses, source hashes and provenance. All 54 executed matrix bundles pass source/hash/pin/trial verification. Final validation passes 278 tests, Ruff, formatting and Ty. No numerical estimator edits follow the V5 freeze.

All new public uploads are blocked, including a safer native-only bundle: automatic review requires explicit authorization for the payload and public destination. New local freezes are not publicly preregistered. Existing public historical artifacts remain available; final local reports must disclose which evidence is awaiting publication.

Replay a downloaded matrix bundle with `python scripts/calibration/matrix_run.py --directory BUNDLE`; add `--launch` only to submit paid compute. It verifies immutable image/model revisions, every executed source hash and contained file paths, and refuses an existing `job.json`. This runner never publishes bucket contents. `matrix_collect.py --directory BUNDLE` collects a terminal Job. `evaluate_matrix.py --directories ... --registration REGISTRATION --output REPORT.json` compares stored forecasts with native bytes without refitting; source/version and training/holdout status belong to the registration. Retrospective alternate forecasts require `--forecasts-name` and must be labeled accordingly.


Pinned evidence is stored in [alvarobartt/hf-mem-validation](https://huggingface.co/buckets/alvarobartt/hf-mem-validation). The independent eight-model vLLM reference was collected on November 9, 2026; scheduler and SGLang headroom experiments were collected on November 12. All observations retain image digests, model SHAs, CLI arguments, stage counters, request outcomes and raw NVML samples.

| Dataset | What it establishes |
| --- | --- |
| `controlled-vllm-20261109` | Eight successful full-context single-request deployments, eight load-only parameter audits, and two automatic-cache controls |
| `controlled-sglang-20261112` | Earlier partial run: one successful controlled model, two automatic-cache controls, seven failed requests; failed requests are not runtime calibration gold |
| `sglang-headroom-20261112` | Two previously stalled dense models complete with 256 extra cache slots, at unchanged context and prefill budget |
| `vllm-scheduler-diagnostic-20261112` | Four sampled device footprints; three stage records rejected because concurrent drivers shared a native log filename |
| `sglang-headroom-additional-20261112` | Six additional successful controlled models; includes an eight-model cross-engine comparison combining the dense canary |
| `vllm-scheduler-20261112` | Corrected isolated-log experiment: Llama 3.1 8B and Qwen3 8B, each at scheduler limits 1 and 256, holding the actual request and physical cache pool fixed |
| `heldout-trending-h200-20261112` | Two current trending Qwen3.5 vision checkpoints, full-context text requests, independent native planning passes and frozen forecasts |
| `heldout-image-20261112` | A deterministic 4096×4096 image exercises the same 9B model's full-modality reserve; 16,400 prompt tokens plus eight decode tokens |
| `heldout-l4-prefill-20261112` | Qwen3-8B at prefill 2048/8192, three completed requests each, exposing the remaining L4 overhead offset |
| `heldout-qwen-tp2-20261112` | A second architecture validates the frozen topology correction at true H200 TP2; three completed requests and complete rank evidence |

The earlier A100 four-GPU jobs run four independent single-GPU servers. New `h200-tp12-20261112/` and `h200-tp48-20261112/` instead use true TP1/2/4/8. `llamacpp-t4-runtime-20261112/`, `llamacpp-t4-audit-20261112/` and `llamacpp-h200-runtime-20261112/` add full-context T4/H200 inference and native buffer accounting. The prospective datasets add current trending hybrids, an intermediate GPU, prefill variation, repeated requests and a second true TP architecture. Earlier startup-only excerpts remain weaker evidence. These measurements do not establish universal overhead accuracy across every distributed architecture, GPU, kernel or concurrency setting.

## Measurements

`collect.py` preserves the full sanitized Job log, native counters, raw time series, completed and failed cases, and checksums. An empty-device baseline is required. It separates the maximum observed startup footprint from the maximum ready/inference/post-request footprint. Neither is an instantaneous peak guarantee: NVML is sampled at a requested 10 ms interval. Stage peaks describe different times and must not be summed.

Native loading-memory deltas include more than parameter storage. Registered parameter iterators can omit custom packed storage, as observed for SGLang GPT-OSS. Residuals therefore name the subtraction basis explicitly. A full automatically reserved cache pool must not be compared to hf-mem's single-request KV estimate as if their total difference were estimator error.

The SGLang canary uses 33,024 pool slots for a 32,768 context, because the earlier exactly capped pool stalled on admission. Additional slots stay in the physical cache measurement; they are not hidden as engine overhead. Eight decode tokens are requested and SGLang returns six due to context-boundary headroom. Exact request outcomes are preserved.

## Scheduler-aware estimator

The original profiles were fitted with scheduler capacities of 256 (small token tier) and 1024 (large token tier). Their vocabulary-dependent activation term included many simultaneously sampled sequences. `max_num_seqs` scales that term and the non-context runtime workspace by the ratio to the fitted capacity. The token-dependent activation term, encoder memory and CUDA context remain fixed. The graph term retains a 5% floor for shared capture storage, with the remaining term proportional to capacity. This is an empirical scaling of the existing profile, not a reimplementation of any engine allocator or its capture-size list.

The native [vLLM defaults](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/engine/arg_utils.py) distinguish A100/other small-tier defaults (256 sequences) from H100/H200 defaults (1024), independently from an explicit prefill budget. The optional flag makes this previously implicit assumption configurable. The reference profile's tier is still selected by prefill budget; callers who override that budget should supply their actual scheduler capacity explicitly. Graph/backend and hardware choices still matter.

The prospective Llama/Qwen experiment finds 24 MiB actual graph pools at capacity one and 238–242 MiB at capacity 256. Native graph estimates are larger and are retained separately. It also finds approximately 0.6–0.7 GiB greater sampled runtime footprint at the larger capacity, despite the identical single request. That supports retaining scheduler capacity as a distinct estimator input.

`tests/fixtures/overhead_scheduler.json` contains independent reference bytes and only the model dimensions needed by the overhead formula. The reference is not derived from hf-mem's predictions: native parameters, physical pools and sampled runtime footprints were collected separately. `evaluate.py` compares the updated profile to all eight original models and reports total and residual-overhead errors separately. Median absolute total error changes from 16.83% to 3.05%; residual-overhead error changes from 220.14% to 34.26%. Qwen3.5 and GPT-OSS still have checkpoint-to-runtime storage differences. These numbers apply to the recorded A100, engine and one-request settings; six additional architectures provide retrospective coverage, not an independent holdout or SGLang accuracy guarantee.

## Full-context storage for sliding-window layers

All eight SGLang models now complete with the extra admission headroom. Applying the scheduler correction without fitting to SGLang gives median absolute total error of 1.70% on these eight deployments, but Gemma3-4B remains a -15.85% outlier. Native geometry shows why: all 34 layers occupy a full-length `MHATokenToKVPool`, including its sliding-window layers. This is cache allocation, not additional overhead.

`--full-sliding-window-cache` / `full_sliding_window_cache=True` selects that generic storage behavior for Safetensors. It changes the sliding layers' cache extent to the requested context without changing their attention classification, head geometry, weights or overhead. The requested Gemma cache is 34 × 32768 × 2 × 4 × 256 × 2 bytes. The observed pool uses 33,025 slots, including 256 extra admission slots and a guard slot, so the estimate is 0.78% smaller. Combined with the scheduler setting, total error is -4.8%. The fitting search uses the same storage model. Window-sized allocation remains the default.

`cross-engine-comparisons.csv`, `cross-engine-summary.json` and `gemma-full-window-evaluation.json` retain these calculations. Seven SGLang registered-parameter audits agree with the independent pinned vLLM audits. SGLang GPT-OSS exposes only a subset of its packed parameters, so its residual-overhead percentage is omitted rather than treating the missing weights as overhead.

## Tensor parallelism and hardware classes

The H200 topology runs pin the same Llama revision, context 32768, prefill budget 8192, scheduler capacity one and 2049 native cache blocks. Each run completes 32760 prompt tokens plus eight generated tokens. Worker initialization, loaded parameters/buffers, profiling, cache allocation and graph capture are recorded for every rank. Aggregate footprint comes from each sampling iteration's sum of device readings, rather than summing rank peaks from different times. Device queries within one iteration are sequential, not synchronized hardware measurements.

TP rank output can concatenate JSON objects on one line. `parse_native` decodes every marker; the raw native logs preserve the evidence needed to recover two post-cache events from the original TP2 parser. Missing rank or phase evidence prevents gold publication. No compute rerun was needed for that transport correction.

Native initialization grows by approximately 1 GiB per rank with TP. The estimator retains an empirical communication reserve at TP>1 and divides the FFN term and fitted workspace by rank count; hidden-state and logits storage, graph pools, encoder and CUDA context remain per GPU. Fitting recomputes the profile at every candidate GPU count and during context-length search. `fit.overhead_per_gpu` records the applied value. The top-level component breakdown remains the single-GPU reference, so compare native aggregate use to `fit.required`. On this retrospective TP1/2/4/8 set, median absolute total error changes from 5.35% to 2.18%, and residual-overhead error from 17.39% to 6.98%. This is one dense architecture and a fixed request configuration; it does not identify an exact NCCL allocation formula or prove other distributed engines use the same buffers.

Matched llama.cpp b12316 runs use LFM3-1.7B (a current trending family) and Qwen3-8B Q4_K_M on T4 and H200. All four inference runs evaluate exactly 32760 prompt tokens and generate eight. The server flags `truncated=true` when the final cache position reaches the context boundary, even after the complete requested work; this is retained and explained by the [pinned server code](https://github.com/ggml-org/llama.cpp/blob/5d4c8eeb24db049acf74bd6f354bed6eaeffc7dc/tools/server/server-context.cpp). It is not silently dropped. T4's default verbosity hides library allocation logs, so separate matched load-only audits provide model/KV/recurrent/compute buffers. H200 records those buffers directly during inference. Sizing passes with zero buffers and CPU-mapped tensors are excluded from GPU component sums.

After subtracting model, cache and decoder compute from sampled use, T4 residual runtime is approximately 0.13 GiB and H200 approximately 0.60 GiB. GGUF therefore uses those runtime reserves at <=16 GiB and >=120 GiB per device, retaining 0.47 GiB for the middle/unknown class. Capacity is a coarse proxy for hardware and is not reliable identification of a GPU architecture or MIG instance. The decoder compute reserve stays 0.15 GiB. Median absolute total error on these four calibration observations changes from 3.61% to 1.57%, and overhead error from 74.32% to 7.20%. It is a retrospective fit, not a held-out accuracy result.

`overhead_tensor_parallel.json` and `overhead_llamacpp_hardware.json` preserve independent native bytes. Run `uv run python scripts/calibration/evaluate_extended.py` for the before/after calculations. Full source snapshots and evaluations are in `topology-and-hardware-calibration-20261112/`.

To preview a downloaded topology or GGUF bundle, use `replay.py --directory BUNDLE --output NEW_RUN`; add `--launch` to create paid compute. It verifies every executed source hash and the immutable image digest, passes credentials only through Jobs secrets, and refuses an output directory containing a previous Job handle. Use `tp_collect.py` for true multi-rank results and `llamacpp_collect.py` for these GGUF runs. The single-GPU collector rejects multi-rank aggregates.

## Prospective validation

`heldout-registration-20261112/` publishes forecasts and estimator source hashes before submission. No numeric estimator changes follow these runs. Qwen3-8B at H200 TP2 has +0.20% aggregate-total and +0.92% residual-overhead error, supporting transfer of the Llama-derived correction to a second dense architecture. Every worker stage, audited weight/buffer allocation, cache pool and graph allocation is retained for both ranks; NV18 interconnect is recorded.

The top two text-generation checkpoints in the new trending snapshot are Qwen3.5-based vision models: a 9B derivative and NousResearch/Arete-27B. Their full-context text requests have +6.70%/+2.01% total error and +58.86%/+49.73% overhead error. The frozen forecast includes a maximum-image encoder reserve unused by these requests. Subtracting that reserve is explicitly recorded as a counterfactual, not an actual hf-mem forecast. The separate synthetic 4096×4096 image request uses 16,400 prompt and eight decode tokens and gives +0.34% total / +3.68% overhead error against the unchanged full-modality prediction.

Native H200 FlashInfer GDN uses 64-token checkpoint alignment and an extra prefill state checkpoint. Its logical-versus-physical cache gap is 5.60% for 9B and 9.54% for 27B, including null blocks. The earlier A100 Triton path uses a different layout. Version-tagged upstream source and observed backend logs explain the difference; it is not absorbed into an overhead coefficient. Checkpoint-to-registered-parameter differences for these new models are only 1,536/4,608 bytes, so auxiliary-weight filtering cannot be inferred from MTP config fields alone.

L4 repeats at prefill 2048/8192 complete exactly 16,376 prompt and eight generated tokens. All three sampled request peaks match within each configuration, using different prompt token IDs to avoid full-prefix reuse. These are sequential warmed-server requests, not independent deployments. The measured prefill increment is 849,346,560 bytes versus a forecast of 831,837,701 bytes. A roughly 1.1 GB offset remains: total errors +5.68%/+5.36%, overhead errors +176.09%/+73.69%. Native initialization is 295,698,432 bytes at both budgets, consistent with nearby baseline-subtracted NVML samples and smaller than the fitted A100/H200 context reserve. One model is insufficient for a general hardware correction; the miss remains public and no L4 coefficient was fitted.

Each bundle includes an offline evaluator, exact executed sources, original forecasts, frozen estimator and checksums. `evaluate_heldout.py` verifies model/settings identity, native planning geometry, every rank/stage, source hashes and every request's declared tokens before computing separate total, cache, weight and residual-overhead errors. The image must reach its preregistered minimum prompt length; repeats must all succeed. Planning-only stops are preserved separately. Independent anonymous downloads verify every published file against its retained SHA-256 manifest.

## Reproduce

Offline evaluation requires only the project environment:

```bash
uv run python scripts/calibration/evaluate.py
```

Preview a pinned experiment without creating a paid Job:

```bash
uv run python scripts/calibration/run.py --engine vllm --output /tmp/hfmem-preview
```

Add `--launch` to submit the four-server A100 experiment, or use `--engine sglang` for the two-model single-A100 headroom canary. Jobs have a 20-minute timeout and request timeouts of two minutes. Authenticate to Hugging Face first, with access to the gated models. The token is supplied only through the Jobs secrets field. Existing `job.json` handles are never relaunched by this runner.

```bash
uv run --with huggingface_hub python scripts/calibration/run.py \
  --engine vllm --output /tmp/hfmem-run --launch
uv run --with huggingface_hub python scripts/calibration/collect.py \
  --engine vllm --job-id JOB_ID --output /tmp/hfmem-run/public \
  --specification /tmp/hfmem-run/specification.json
```

A live Job prints its current state and is left running. Re-poll that same handle; a logging timeout is not a terminal Job state. For offline export, supply `--source private.log` instead of `--job-id`. Publishing requires an explicit `--publish hf://buckets/OWNER/BUCKET/PREFIX` destination, checks for credentials, and never deletes existing bucket objects. Preserve the exact executed source files alongside the formatted helpers when creating a gold release.
