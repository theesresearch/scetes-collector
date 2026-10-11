# SCETES Measurement Specification 1.0

Status: draft for review · Version 1.0.0 · 6 October 2026
Applies to: `sce_infer.py` and `sce_upload.py` 1.0.0 to 1.0.4 (see `SHA256SUMS`; 1.0.1 to 1.0.4 change no measurement, 1.0.2 adds the idle-share calculation in the upload tool).
Licence of this document: to be confirmed by the owner (proposed: CC BY 4.0).

This document states exactly what a SCETES measurement is, how each number is produced and where its limits are. It describes the released code; where the code and this text disagree, the code is authoritative and this text must be corrected.

## 1. Scope

A SCETES measurement describes **one language model, one serving configuration and one NVIDIA GPU** under a defined synthetic workload. It is valid for that setup only.

Covered: autoregressive text generation with Hugging Face Transformers (`generate`, greedy decoding, FP16 weights) on one CUDA-visible NVIDIA GPU with NVML support.

Not covered: other serving engines (for example vLLM), multi-GPU or multi-node serving, CPU or non-NVIDIA accelerators, whole-server or facility energy, output quality, service-level objectives, carbon emissions.

## 2. Definitions

| Term | Definition |
|---|---|
| Output token | A token produced by the model during generation. Input (prompt) tokens are not counted as output. |
| Batch | `B` identical requests processed together in one `generate` call. |
| Generated tokens per run | `B × N`, where `N` is the fixed number of output tokens per request. |
| Run (repeat) | One timed generation of `B × N` output tokens, preceded by its own prefill and first-token measurements. |
| Warm-up | Untimed generations executed before the first run (default 2). |
| Busy sample | A telemetry sample taken during any measured phase other than `idle`. |

Output length is **fixed**: `max_new_tokens = min_new_tokens = N`, so every request generates exactly `N` tokens regardless of end-of-sequence tokens. This makes runs comparable but differs from natural production output lengths.

## 3. Measurement boundary

Energy is the **total energy of the selected GPU device** observed during the measured interval, as reported by NVML. It is not isolated to the measuring process: anything else running on the same GPU contributes. The GPU must therefore be otherwise idle during the measurement.

Not included: CPU, memory, storage, network, power supply losses, cooling or facility overhead (PUE). Idle power is recorded (2-second idle phase) but **not subtracted**: reported energy is gross device energy, not net energy above idle.

## 4. Procedure

1. Map the CUDA-visible GPU index to its NVML handle by CUDA UUID, then PCI bus ID, then index (`gpu_mapping_method` records which was used). Only that handle is sampled.
2. Load the model in FP16 on the selected GPU; tokenize `B` copies of the prompt with left padding.
3. Run the warm-up generations.
4. Measure a 2-second idle phase.
5. For each run `r = 1 … R` measure, in this order, with GPU synchronisation before and after each phase:
   - **prefill**: one forward pass over the prompt batch;
   - **first token**: `generate` with one output token;
   - **full generation**: `generate` with `N` output tokens.
6. Write the result archive (section 8).

The prompt and generated text stay in process memory and are never written to the archive.

## 5. Telemetry and energy

During every phase a background thread samples NVML every `sample_period_s` (default 0.02 s): power, GPU and memory utilisation, SM clock, maximum SM clock, temperature, clock-reduction reasons (bit mask) and device memory in use. One sample is also taken immediately before and after the phase.

Energy of a phase is taken from the NVML total-energy counter (difference of the readings before and after, in millijoules). Independently, sampled power is integrated over time (trapezoidal rule). The counter value is used when it is available and positive; otherwise the integral is used (`energy_source`). A phase without any valid energy reading stops the measurement; zero or negative energy is never reported.

**Energy check:** when both values exist, their relative difference `|counter − integral| / counter` is recorded. A run passes when it is at most **10 %**. The check is a consistency indicator, not a calibration.

## 6. Metrics

| Metric | Definition |
|---|---|
| Throughput (tokens/s) | `B × N / t_full`, where `t_full` is the wall time of the full-generation phase. |
| GPU energy per output token (J/token) | `E_full / (B × N)`. Includes the prefill work contained in the full generation. |
| Request latency | `t_full` of each run (batch-level: all `B` requests finish together). |
| Latency percentiles | Nearest-rank percentiles over the runs. p50 always; **p95 only with ≥ 20 runs; p99 only with ≥ 100 runs**, otherwise omitted. |
| Time to first token (TTFT) | Wall time of the separate one-token `generate` call. |
| Decode inter-token latency | Estimate: `(t_full − t_prefill) / (N − 1)`. Not a per-token trace. |
| Decode energy per token | Estimate: `(E_full − E_prefill) / (B × N − B)`. |
| Coefficient of variation | Population standard deviation of request latency divided by its mean. |

Reported summary values are **medians over runs** for throughput and energy per token.

## 7. Throttling, memory and idle time

**Throttling flag (pass/fail rule, unchanged from the preregistered E0 experiment):** a configuration is flagged when, over its busy samples, the lowest SM clock is more than 10 % below the highest observed SM clock, or the temperature exceeds 80 °C. A configuration without recorded clocks is treated as flagged.

**Throttling loss (reported values):**
- share of busy samples with an SM clock more than 10 % below the reference clock;
- mean SM clock as a percentage of the reference clock;
- reasons reported by NVML in at least 1 % of busy samples: `power_cap`, `hardware_slowdown`, `sync_boost`, `thermal_slowdown`, `hardware_thermal_slowdown`, `power_brake`, `display_clock_setting`. The idle and application-clock bits are normal settings and are not reported.

The reference clock is the maximum SM clock reported by NVML; if the driver does not report it, the highest observed clock is used.

**GPU memory headroom:** peak device memory in use during the run and total device memory. Device memory in use includes all processes on the GPU. Free memory indicates room to test a larger batch; it does not show that a larger batch will help.

**GPU idle share:** 100 % minus the mean GPU utilisation reported by the NVIDIA driver (NVML) over all samples of the measured phases (the 2-second idle phase is excluded). NVML utilisation is the share of each sample period in which at least one kernel was executing, so this is the share of measured time in which, according to the driver, no GPU kernel was running, for example while the CPU prepared the next step. Driver-reported utilisation only shows whether a kernel was running, not how efficiently the GPU was used; a low idle share does not mean the GPU is used efficiently. It describes the test run and not production traffic, and it is not a saving. Computed by the upload tool from 1.0.2.

## 8. Result archive and integrity

One ZIP per configuration: `manifest.json` (configuration, versions, privacy flags), `system.json` (Python, PyTorch, Transformers, CUDA, GPU name, compute capability, memory, resolved model revision when available), `metrics.json` (summary), `benchmark.csv` (one row per run), `telemetry.csv` (all samples) and `checksum.txt` (SHA-256 of every other file). The upload tool refuses an archive whose checksums do not match.

The hostname is stored as a SHA-256 hash by default; the operator may include or omit it.

## 9. Comparing configurations

A comparison is made only between configurations measured on the same GPU, model, precision and output length, changing one variable (for example batch size). The upload tool warns when runs differ in model, precision or output length.

- **Verified comparison:** both configurations passed the energy check in every run and are not flagged for throttling.
- **Indicative comparison:** throttling is the only reason a verified comparison is missing, the energy check passed, and the alternative is at least **20 %** cheaper. Measured values already include the slowdown. The 20 % margin is provisional: two sessions of the same configuration on an NVIDIA T4 differed by about 11 % in throughput (6 October 2026, two samples). It must be reviewed when more data exists.
- Otherwise no comparison is reported, and the reason is stated.

## 10. Cost calculation

Costs are calculated only from measured values and prices supplied by the customer. No market prices are assumed.

- Energy cost per 1 million output tokens = `J/token × 10^6 / 3.6 × 10^6 × price per kWh`.
- GPU time cost per 1 million output tokens = `price per GPU hour × 10^6 / (tokens/s × 3600)`.
- The two are never added: a rental price per GPU hour already includes electricity.
- A monthly figure is shown only when the customer states a monthly output volume. It is an estimate of potential for the measured setup, not a guaranteed saving.

## 11. Known limitations

- Single GPU, single model per archive, Transformers only, FP16 only.
- Synthetic workload: one prompt repeated `B` times, fixed output length, greedy decoding.
- Device-level energy without idle subtraction; other GPU workloads contribute.
- Batch-level latency, not a per-request production distribution.
- Run-to-run and session-to-session variation on shared cloud GPUs can be large; on a Google Colab T4 the GPU reduced its clock in all runs (reason: power cap).
- No measurement of output quality or service-level objectives.

## 12. Related work

This specification is narrower than and independent of existing benchmarks such as MLPerf Power (MLCommons) and the ML.ENERGY Benchmark, which measure energy across systems and models on their own hardware. SCETES measures a customer's own configuration on the customer's own hardware.

## 13. Change log

- 1.0.0 (6 October 2026): first written specification, matching collector 1.0.0.
- 1.0.0, applies to collector 1.0.1 (7 October 2026): no change to any measurement; the upload tool gained a separate, optional statistics consent.
- 1.0.0, applies to collector 1.0.2 (7 October 2026): section 7 defines GPU idle time, calculated from utilisation samples that collector 1.0.0 already records.
