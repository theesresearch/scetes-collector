# SCETES collector

Measure LLM inference on your own NVIDIA GPU: output tokens per second, batch-level latency and GPU energy per output token. The collector runs locally, writes a ZIP archive with SHA-256 checksums and **uploads nothing**. Sending results to SCETES is a separate, optional step with its own consent.

Licence: [Apache License 2.0](LICENSE). Copyright 2026 Jana Kolářová. Website: https://scetes.com

## What it measures

- Output-token throughput (tokens per second) for one model, one configuration and one selected NVIDIA GPU.
- Batch-level latency percentiles: p50 always, p95 from 20 repeats, p99 from 100 repeats. They describe repeated batch runs, not a production per-request distribution.
- GPU energy per output token: total energy of the selected GPU during the measured interval (NVML energy counter where available, otherwise integrated power samples; both are recorded when available), divided by the generated output tokens.
- Prompt prefill time, time to first token, an estimate of decode energy (by subtraction, not token-by-token tracing), GPU utilisation, clocks, temperature, memory and the reasons the driver reports for a reduced clock.

The exact definitions, the measurement boundary and the limits are in [SPECIFICATION.md](SPECIFICATION.md).

## What it does not claim

- Results describe the measured model, configuration, software and GPU only. Do not generalise them to other models, GPUs or serving systems.
- Energy is the energy of the GPU device, not of the whole server, a wall-socket meter or the data centre. It is not isolated to the measuring process: other work on the same GPU is included, so use an otherwise idle GPU.
- The collector reports no cost, CO2e, output quality, SLO goodput or multi-GPU scaling. A cost per million tokens can be calculated from your own prices (for example with the calculator on scetes.com).
- A result is a measurement record, not a certification and not a guarantee of savings.

## Verified and not verified

Verified on real result archives on 6 October 2026: one NVIDIA Tesla T4 on Google Colab, `Qwen/Qwen2.5-1.5B-Instruct`, batch sizes 1, 8 and 32, 20 repeats each, energy from the NVML counter (energy check passed in every run). The T4 reduced its clock speed in all runs (driver reason: power cap); the collector reports this openly. The measurement code of releases 1.0.0 to 1.0.4 is identical to the version used for that verification.

Not verified: other GPUs, other models, other serving systems, multi-GPU setups and non-NVIDIA hardware. The vLLM variant (`sce_infer_vllm.py`) is experimental and not verified.

## Run

Tested with Python 3.13, PyTorch 2.11 and Transformers 5.18 on an NVIDIA GPU with NVML (NVIDIA Tesla T4 on Google Colab, with accelerate 1.15 and nvidia-ml-py 13.615, 10 October 2026). Other versions are not verified. Install `torch`, `transformers`, `accelerate` and `nvidia-ml-py` builds that match your CUDA setup, then run:

```bash
python sce_infer.py --model Qwen/Qwen2.5-1.5B-Instruct --prompt "Explain how a refrigerator works." \
  --output-tokens 128 --batch-size 1 --repeats 20 --warmup 3 --gpu 0
```

The collector puts the selected GPU under full load for the duration of the measurement. Run it on hardware where this is acceptable, ideally outside production, and only with the permission of the hardware owner.

Use a prompt you are allowed to process. It is passed on the local command line, held in memory and not written to the archive; model outputs are not saved. The hostname is stored as a SHA-256 hash by default (`--hostname omit` or `include` to change this). Loading a model may contact its model repository unless you give a local path.

Archive contents: `manifest.json`, `system.json`, `metrics.json`, `benchmark.csv`, `telemetry.csv` and `checksum.txt`.

## End of a run: summary and optional web report (1.0.5.dev0, not released)

After the archive is written, `sce_infer.py` prints a short summary (tokens per second, p95 batch latency, GPU energy per output token, throttling and the path of the local report). It then offers to send the summary numbers to SCETES for a private link to a web report: it lists every field that would be sent, links to https://scetes.com/en/privacy/ and asks `[y/N]` (default no). The fields are exactly those of `sce_upload.py` below.

- Without a terminal (CI, pipes, notebooks without a TTY) nothing is ever sent, not even with `--upload`; send later with `sce_upload.py`.
- `--upload` sends without the question (the field list is still shown); `--no-upload` skips the offer.
- `--dry-run` prints exactly what would be sent and sends nothing.

## Optional upload (`sce_upload.py`)

`sce_upload.py` reads one or more archives (one per configuration), verifies their checksums, prints exactly what would be sent and sends it only after you agree (`[y/N]`, default no). Use `--dry-run` to see the content without sending or being asked anything. Prompts, model outputs, the hostname and the GPU UUID are never read or sent. Prices and an e-mail address are optional; an e-mail needs its own consent, and a separate optional question asks whether anonymised, aggregated statistics may use the measurements (default no). `--delete LINK` removes a result you sent. Standard library only, no GPU needed. How SCETES handles the data: https://scetes.com/en/privacy/

## Integrity and tests

`SHA256SUMS` lists the checksum of each script: `sha256sum -c SHA256SUMS`. Tests: `python3 -m unittest discover -s tests` (the contract test against the SCETES API is skipped here, because the server is not part of this repository).

## Scope of this repository

This repository contains the local collector and the optional upload tool only. The SCETES website, API and server-side analysis are not part of it.
