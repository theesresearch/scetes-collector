#!/usr/bin/env python3
"""SCETES collector for vLLM (EXPERIMENTAL, not yet verified on real hardware).

Measures one model served by vLLM's offline engine (`LLM.generate`) on one NVIDIA GPU and writes a result archive in the
same format as sce_infer.py, so sce_upload.py can read it. GPU sampling, energy and integrity checks are imported from
sce_infer.py unchanged. Differences from the Transformers collector are stated in the archive manifest:
- vLLM has no forward-only call, so the prefill phase is the one-token generation (prefill + first token);
- the GPU is synchronised by waiting for `generate` to return, not by torch.cuda.synchronize().
"""
from __future__ import annotations

import argparse, atexit, csv, hashlib, json, os, platform, shutil, socket, statistics, time, uuid, zipfile
from datetime import datetime, timezone
from pathlib import Path

import sce_infer as base

VERSION = "1.1.0.dev0+vllm"


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face model id or local path")
    ap.add_argument("--prompt", required=True, help="Prompt text (kept in memory; never written to report)")
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--gpu", type=int, default=0, help="CUDA-visible GPU index")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="passed to vLLM")
    ap.add_argument("--max-model-len", type=int, default=None, help="passed to vLLM (smaller values fit small GPUs)")
    ap.add_argument("--sample-period", type=float, default=0.02)
    ap.add_argument("--hostname", choices=["include", "hash", "omit"], default="hash")
    ap.add_argument("--output", default="sce_inference_vllm.zip")
    a = ap.parse_args(argv)
    if min(a.output_tokens, a.batch_size, a.repeats, a.sample_period) <= 0 or a.warmup < 0 or not 0 < a.gpu_memory_utilization <= 1:
        ap.error("Invalid measurement parameter")
    return ap, a


def main(argv=None, vllm_module=None, gpu_class=None):
    ap, a = parse(argv)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)  # vLLM uses the first visible device
        visible = 0
    else:
        visible = a.gpu
    if vllm_module is None:
        import vllm as vllm_module  # noqa: PLC0415 (imported after CUDA_VISIBLE_DEVICES is set)
    import torch
    if not torch.cuda.is_available():
        ap.error("PyTorch CUDA is not available")
    props = torch.cuda.get_device_properties(visible)

    llm_kwargs = {"model": a.model, "dtype": "float16", "gpu_memory_utilization": a.gpu_memory_utilization, "seed": 0}
    if a.max_model_len:
        llm_kwargs["max_model_len"] = a.max_model_len
    llm = vllm_module.LLM(**llm_kwargs)
    prompts = [a.prompt] * a.batch_size
    full_params = vllm_module.SamplingParams(max_tokens=a.output_tokens, min_tokens=a.output_tokens, temperature=0.0, ignore_eos=True)
    one_params = vllm_module.SamplingParams(max_tokens=1, min_tokens=1, temperature=0.0, ignore_eos=True)

    counted = {}
    def run(params, key):
        outs = llm.generate(prompts, params, use_tqdm=False)
        counted[key] = sum(len(o.outputs[0].token_ids) for o in outs)
        counted[key + "_input"] = max(len(o.prompt_token_ids) for o in outs)
    sync = lambda: None  # generate() returns only after all requests are finished

    for _ in range(a.warmup):
        run(full_params, "warmup")

    host = socket.gethostname()
    host_record = host if a.hostname == "include" else hashlib.sha256(host.encode()).hexdigest() if a.hostname == "hash" else None
    rows, telemetry = [], []
    gpu = (gpu_class or base.GPU)(visible, getattr(props, "uuid", None), getattr(props, "pci_bus_id", None))
    try:
        idle = gpu.measure(lambda: time.sleep(2), sync, a.sample_period)
        for s in idle["samples"]:
            telemetry.append({"repeat": 0, "phase": "idle", "gpu_index": a.gpu, "nvml_gpu_index": gpu.index, "gpu_uuid": gpu.uuid, **s})
        for rep in range(1, a.repeats + 1):
            first = gpu.measure(lambda: run(one_params, "first"), sync, a.sample_period)
            full = gpu.measure(lambda: run(full_params, "full"), sync, a.sample_period)
            generated = counted["full"]
            if generated != a.batch_size * a.output_tokens:
                raise RuntimeError(f"vLLM generated {generated} tokens, expected {a.batch_size * a.output_tokens}; refusing to report")
            decode_energy = max(0.0, full["energy_j"] - first["energy_j"])
            decode_time = max(0.0, full["elapsed_s"] - first["elapsed_s"])
            row = {"repeat": rep, "gpu_index": a.gpu, "nvml_gpu_index": gpu.index, "gpu_uuid": gpu.uuid, "model": a.model,
                   "precision": "float16", "batch_size": a.batch_size, "input_tokens_per_request": counted["full_input"],
                   "output_tokens_per_request": a.output_tokens, "prefill_latency_s": first["elapsed_s"], "ttft_s": first["elapsed_s"],
                   "request_latency_s": full["elapsed_s"], "energy_window_s": full["energy_window_s"],
                   "decode_itl_estimate_s": decode_time / max(1, a.output_tokens - 1),
                   "throughput_tokens_s": generated / full["elapsed_s"], "prefill_energy_j": first["energy_j"],
                   "total_energy_j": full["energy_j"], "decode_energy_incremental_j": decode_energy, "idle_power_w": idle["avg_power_w"],
                   "j_per_output_token_total": full["energy_j"] / generated,
                   "j_per_output_token_decode_incremental": decode_energy / max(1, generated - a.batch_size),
                   "energy_source": full["energy_source"], "energy_counter_j": full["energy_counter_j"],
                   "energy_integral_j": full["energy_integral_j"], "energy_counter_integral_rel_diff": full["energy_counter_integral_rel_diff"],
                   "energy_check_passed": full["energy_check_passed"], "avg_power_w": full["avg_power_w"],
                   "avg_util_gpu_pct": full["avg_util_gpu_pct"], "max_temp_c": full["max_temp_c"]}
            rows.append(row)
            for phase, result in (("first_token", first), ("full_generation", full)):
                for s in result["samples"]:
                    telemetry.append({"repeat": rep, "phase": phase, "gpu_index": a.gpu, "nvml_gpu_index": gpu.index, "gpu_uuid": gpu.uuid, **s})
            print(f"repeat {rep}/{a.repeats}: {row['throughput_tokens_s']:.2f} tok/s, {row['j_per_output_token_total']:.4f} J/output-token")
    finally:
        gpu.close()

    lat = [r["request_latency_s"] for r in rows]
    checks = [r["energy_check_passed"] for r in rows if r["energy_check_passed"] is not None]
    metric = {"runs": len(rows), "throughput_tokens_s_median": statistics.median(r["throughput_tokens_s"] for r in rows),
              "j_per_output_token_median": statistics.median(r["j_per_output_token_total"] for r in rows),
              "request_latency_p50_s": base.percentile(lat, .50), "request_latency_p95_s": base.percentile(lat, .95, 20),
              "request_latency_p99_s": base.percentile(lat, .99, 100), "ttft_p50_s": base.percentile([r["ttft_s"] for r in rows], .50),
              "decode_itl_p50_s": base.percentile([r["decode_itl_estimate_s"] for r in rows], .50),
              "coefficient_of_variation_latency": statistics.pstdev(lat) / statistics.mean(lat) if len(lat) > 1 and statistics.mean(lat) else 0,
              "energy_sources": sorted(set(r["energy_source"] for r in rows)), "energy_check_threshold_rel_diff": 0.10,
              "latency_percentile_minimum_repeats": {"p95": 20, "p99": 100},
              "energy_check_pass_fraction": statistics.mean(checks) if checks else None,
              "note": "vLLM offline engine. Prefill is not measured separately: TTFT is a one-token generate call and decode values are estimates from full minus that call."}
    outdir = Path(".sce-" + uuid.uuid4().hex); outdir.mkdir()
    atexit.register(shutil.rmtree, outdir, ignore_errors=True)
    sysinfo = {"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
               "vllm": getattr(vllm_module, "__version__", None), "cuda": getattr(torch.version, "cuda", None),
               "cuda_visible_gpu_index": a.gpu, "nvml_gpu_index": gpu.index, "gpu_uuid": gpu.uuid, "gpu_mapping_method": gpu.mapping_method,
               "gpu_name": gpu.name, "gpu_compute_capability": f"{props.major}.{props.minor}", "gpu_vram_bytes": props.total_memory,
               "hostname_policy": a.hostname, "hostname": host_record, "collector_version": VERSION}
    manifest = {"experiment_id": outdir.name, "collector_version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
                "model": a.model, "resolved_model_revision": None, "precision": "float16", "engine": "vllm",
                "workload": "vLLM offline LLM.generate, greedy (temperature 0), ignore_eos, fixed output length",
                "parameters": {"batch_size": a.batch_size, "output_tokens": a.output_tokens, "repeats": a.repeats, "warmup_runs": a.warmup,
                               "cuda_visible_gpu_index": a.gpu, "nvml_gpu_index": gpu.index, "gpu_uuid": gpu.uuid,
                               "gpu_mapping_method": gpu.mapping_method, "sample_period_s": a.sample_period,
                               "gpu_memory_utilization": a.gpu_memory_utilization, "max_model_len": a.max_model_len},
                "privacy": {"prompt_text_saved": False, "model_responses_saved": False, "network_upload": False, "hostname_policy": a.hostname}}
    for name, obj in (("manifest.json", manifest), ("system.json", sysinfo), ("metrics.json", metric)):
        (outdir / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, data in (("benchmark.csv", rows), ("telemetry.csv", telemetry)):
        with (outdir / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in data for k in r})); w.writeheader(); w.writerows(data)
    files = list(outdir.iterdir())
    (outdir / "checksum.txt").write_text("\n".join(f"{base.sha256(p)}  {p.name}" for p in files) + "\n", encoding="utf-8")
    with zipfile.ZipFile(a.output, "w", zipfile.ZIP_DEFLATED) as z:
        for p in outdir.iterdir():
            z.write(p, p.name)
    shutil.rmtree(outdir, ignore_errors=True)
    print("Created", Path(a.output).resolve())
    return metric


if __name__ == "__main__":
    main()
