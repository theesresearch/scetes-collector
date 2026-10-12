#!/usr/bin/env python3
"""SCE inference audit v1.0: local Hugging Face causal LLM measurement."""
from __future__ import annotations

import argparse, atexit, csv, hashlib, json, math, os, platform, shutil, socket, statistics, sys, threading, time, uuid, zipfile
from datetime import datetime, timezone
from pathlib import Path

VERSION = "1.0.5.dev0"

def percentile(xs, q, minimum_n=1):
    ys = sorted(xs)
    if len(ys) < minimum_n: return None
    return ys[max(0, math.ceil(q * len(ys)) - 1)]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()

class GPU:
    def __init__(self, visible_index, cuda_uuid=None, pci_bus_id=None):
        import pynvml
        self.nv = pynvml
        pynvml.nvmlInit()
        self.visible_index = visible_index
        self.index = None
        self.mapping_method = None
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        # Prefer the identity reported by CUDA for the selected logical device.
        # CUDA ordinals and NVML ordinals are not guaranteed to share ordering.
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        def norm_uuid(value):
            if isinstance(value, bytes): value = value.decode(errors="replace")
            return str(value).strip().lower().removeprefix("gpu-").removeprefix("mig-")
        def norm_bus(value):
            if isinstance(value, bytes): value = value.decode(errors="replace")
            value = str(value).strip().lower()
            return value.split(":", 1)[-1] if ":" in value else value
        matched = None
        if cuda_uuid:
            uuid_text = cuda_uuid.decode(errors="replace") if isinstance(cuda_uuid, bytes) else str(cuda_uuid)
            # MIG UUIDs identify an instance, while PCI bus IDs identify its
            # parent card. Never silently charge a parent card's energy to a MIG workload.
            if uuid_text.upper().startswith("MIG-"):
                try:
                    self.h = pynvml.nvmlDeviceGetHandleByUUID(uuid_text)
                    self.mapping_method = "torch_mig_uuid"
                except Exception as exc:
                    pynvml.nvmlShutdown()
                    raise ValueError("Could not map selected CUDA MIG UUID to NVML") from exc
                matched = self.h
            else:
                expected = norm_uuid(uuid_text)
                for h in handles:
                    if norm_uuid(pynvml.nvmlDeviceGetUUID(h)) == expected:
                        matched = h
                        self.mapping_method = "torch_cuda_uuid"
                        break
        if matched is None and pci_bus_id:
            expected = norm_bus(pci_bus_id)
            for h in handles:
                try: actual = pynvml.nvmlDeviceGetPciInfo(h).busId
                except Exception: continue
                if norm_bus(actual) == expected:
                    matched = h
                    self.mapping_method = "torch_pci_bus_id"
                    break
        if matched is not None:
            self.h = matched
            self.index = next((i for i, h in enumerate(handles) if h == matched), None)
        elif cvd is None:
            nvml_index = visible_index
            if nvml_index < 0 or nvml_index >= pynvml.nvmlDeviceGetCount():
                pynvml.nvmlShutdown()
                raise ValueError("Selected CUDA/NVML GPU index is unavailable")
            self.h = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)
            self.index = nvml_index
            self.mapping_method = "same_ordinal_fallback"
        else:
            visible = [item.strip() for item in cvd.split(",") if item.strip()]
            if visible_index < 0 or visible_index >= len(visible):
                pynvml.nvmlShutdown()
                raise ValueError("--gpu is outside CUDA_VISIBLE_DEVICES")
            token = visible[visible_index]
            if token.isdigit():
                nvml_index = int(token)
                if nvml_index >= pynvml.nvmlDeviceGetCount():
                    pynvml.nvmlShutdown()
                    raise ValueError("CUDA_VISIBLE_DEVICES refers to a missing NVML device")
                self.h = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)
                self.index = nvml_index
                self.mapping_method = "cuda_visible_devices_ordinal_fallback"
            elif token.startswith(("GPU-", "MIG-")):
                try:
                    self.h = pynvml.nvmlDeviceGetHandleByUUID(token)
                except Exception:
                    try: self.h = pynvml.nvmlDeviceGetHandleByUUID(token.encode())
                    except Exception as exc:
                        pynvml.nvmlShutdown()
                        raise ValueError("Could not map CUDA_VISIBLE_DEVICES UUID to NVML") from exc
                nvml_index = None
                self.mapping_method = "cuda_visible_devices_uuid"
            else:
                pynvml.nvmlShutdown()
                raise ValueError("Unsupported CUDA_VISIBLE_DEVICES entry; use an index or GPU UUID")
        self.uuid = pynvml.nvmlDeviceGetUUID(self.h)
        if isinstance(self.uuid, bytes): self.uuid = self.uuid.decode(errors="replace")
        self.name = pynvml.nvmlDeviceGetName(self.h)
        if isinstance(self.name, bytes): self.name = self.name.decode(errors="replace")
        try: pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h); self.counter = True
        except Exception: self.counter = False
        # Maximum SM clock: the reference for "how much speed did throttling cost". None when the driver does not report it.
        try: self.max_sm_mhz = pynvml.nvmlDeviceGetMaxClockInfo(self.h, pynvml.NVML_CLOCK_SM)
        except Exception: self.max_sm_mhz = None

    def sample(self):
        n, h = self.nv, self.h
        d = {"t": time.perf_counter(), "power_w": None, "util_gpu_pct": None,
             "util_mem_pct": None, "clock_sm_mhz": None, "temp_c": None,
             "max_clock_sm_mhz": self.max_sm_mhz, "throttle_reasons": None, "mem_used_mib": None, "mem_total_mib": None}
        try: d["power_w"] = n.nvmlDeviceGetPowerUsage(h) / 1000
        except Exception: pass
        try:
            u = n.nvmlDeviceGetUtilizationRates(h); d["util_gpu_pct"], d["util_mem_pct"] = u.gpu, u.memory
        except Exception: pass
        try: d["clock_sm_mhz"] = n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM)
        except Exception: pass
        try: d["temp_c"] = n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU)
        except Exception: pass
        # Bit mask of the reasons NVML reports for a reduced clock (the function was renamed in newer drivers).
        for name in ("nvmlDeviceGetCurrentClocksEventReasons", "nvmlDeviceGetCurrentClocksThrottleReasons"):
            try: d["throttle_reasons"] = int(getattr(n, name)(h)); break
            except Exception: pass
        try:
            mem = n.nvmlDeviceGetMemoryInfo(h); d["mem_used_mib"], d["mem_total_mib"] = round(mem.used / 2**20), round(mem.total / 2**20)
        except Exception: pass
        return d

    def measure(self, fn, sync, period):
        samples, stop = [], threading.Event()
        def poll():
            while not stop.is_set():
                try: samples.append(self.sample())
                except Exception: pass
                stop.wait(period)
        sync()
        try: e0 = self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h) if self.counter else None
        except Exception: e0 = None
        samples.append(self.sample())
        th = threading.Thread(target=poll, daemon=True); th.start(); t0 = time.perf_counter()
        try: fn(); sync()
        finally:
            t1 = time.perf_counter()
            try: e1 = self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h) if self.counter else None
            except Exception: e1 = None
            stop.set(); th.join()
        samples.append(self.sample())
        integ = sum((b["t"]-a["t"]) * (a["power_w"]+b["power_w"])/2
                    for a,b in zip(samples,samples[1:]) if a["power_w"] is not None and b["power_w"] is not None)
        has_integral = any(a["power_w"] is not None and b["power_w"] is not None
                           for a,b in zip(samples,samples[1:]))
        if not has_integral: integ = None
        counter_j = (e1-e0)/1000 if e0 is not None and e1 is not None else None
        if counter_j is not None and counter_j <= 0: counter_j = None
        used = counter_j if counter_j is not None else integ
        if used is None or used <= 0:
            raise RuntimeError("No valid GPU energy reading; refusing to report zero or invalid energy.")
        agreement = abs(counter_j - integ) / counter_j if counter_j is not None and integ is not None and counter_j > 0 else None
        return {"elapsed_s": t1-t0, "energy_window_s": samples[-1]["t"]-samples[0]["t"],
                "energy_j": used, "energy_counter_j": counter_j,
                "energy_integral_j": integ, "energy_counter_integral_rel_diff": agreement,
                "energy_check_passed": agreement <= 0.10 if agreement is not None else None,
                "energy_source": "nvml_counter" if counter_j is not None else "sampled_power_integral",
                "samples": samples,
                "avg_power_w": statistics.mean(x["power_w"] for x in samples if x["power_w"] is not None) if any(x["power_w"] is not None for x in samples) else None,
                "avg_util_gpu_pct": statistics.mean(x["util_gpu_pct"] for x in samples if x["util_gpu_pct"] is not None) if any(x["util_gpu_pct"] is not None for x in samples) else None,
                "max_temp_c": max((x["temp_c"] for x in samples if x["temp_c"] is not None), default=None)}

    def close(self):
        try: self.nv.nvmlShutdown()
        except Exception: pass

PRIVACY_URL = "https://scetes.com/en/privacy/"


def _field_paths(obj, prefix=""):
    """Every field name in the upload payload, e.g. runs[].metrics.tokens_per_s, so the question lists exactly what is sent."""
    if isinstance(obj, dict):
        return [q for k, v in obj.items() for q in _field_paths(v, f"{prefix}.{k}" if prefix else k)]
    if isinstance(obj, list):
        return sorted({q for v in obj for q in _field_paths(v, prefix + "[]")}) if obj else [prefix + "[]"]
    return [prefix]


def after_run(archive_path, upload=None, dry_run=False, endpoint=None, stdin=None, out=None, interactive=None, post=None):
    """End of a run: a short summary, then the optional upload of the summary numbers.

    Never sends without a yes: the default answer is no, --no-upload skips the question, --dry-run prints the exact JSON and
    sends nothing, and without a terminal (CI, pipes, notebooks without a TTY) nothing is ever sent, not even with --upload.
    The payload is built by sce_upload.build_payload, so the fields are exactly those of the separate upload tool."""
    import sce_upload as up  # standard library only; ships next to this file
    out = out or sys.stdout
    say = lambda *x: print(*x, file=out)
    arch = up.read_archive(str(archive_path))
    m = arch["metrics"]
    flagged, checked = up.throttling_flagged(arch["telemetry"])
    detail = up.throttling_detail(arch["telemetry"])
    p95 = m.get("request_latency_p95_s")
    say("\nSummary")
    say(f"  Tokens per second (median):     {m['throughput_tokens_s_median']:.2f}")
    say(f"  p95 batch latency:              {'%.3f s' % p95 if p95 is not None else 'not reported (needs at least 20 repeats)'}")
    say(f"  GPU energy per output token:    {m['j_per_output_token_median']:.4f} J")
    if not checked:
        say("  Throttling:                     clocks not recorded")
    elif detail:
        say(f"  Throttling:                     {'yes' if flagged else 'no'}, {detail['reduced_clock_time_pct']} % of the time with a reduced clock")
    else:
        say(f"  Throttling:                     {'yes' if flagged else 'no'}")
    say(f"  Local report:                   {os.path.abspath(archive_path)}")
    if upload is False:
        say("\nUpload: skipped (--no-upload). Nothing was sent.")
        return 0
    payload = up.build_payload([arch])
    if dry_run:
        say("\nThis is exactly what would be sent:\n" + json.dumps(payload, indent=2, ensure_ascii=False))
        say("\nDry run: nothing was sent.")
        return 0
    if interactive is None:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if not interactive:
        say("\nNo terminal: nothing was sent. To send the summary numbers later, run: python sce_upload.py " + str(archive_path))
        return 0
    say("\nOptional: send the summary numbers to SCETES and get a private link to your web report.")
    say("These fields would be sent (prompts, model outputs, your computer's name and the GPU UUID never are):")
    for f in _field_paths(payload):
        say("  " + f)
    say(f"How SCETES handles the data: {PRIVACY_URL}")
    if upload is not True:
        out.write("Send these measurements? [y/N] "); out.flush()
        if (stdin or sys.stdin).readline().strip().lower() not in ("y", "yes"):
            say("Nothing was sent.")
            return 0
    endpoint = up.check_endpoint(endpoint or up.DEFAULT_ENDPOINT)
    status, body = (post or up.http)("POST", endpoint, payload)
    if status != 201:
        say(f"Not accepted ({status}): {(body or {}).get('error', 'unknown error')}. Your local report is unchanged.")
        return 1
    say("\nSent. Your web report (keep this link private; it is the only way to read or delete the result):")
    say(f"  {up.page_link(endpoint, body['token'])}\n  stored until {body['expires_at']}")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Hugging Face model id or local path")
    ap.add_argument("--prompt", required=True, help="Prompt text (kept in memory; never written to report)")
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--gpu", type=int, default=0, help="CUDA-visible GPU index; NVML mapping prefers CUDA UUID/PCI identity")
    ap.add_argument("--sample-period", type=float, default=0.02)
    ap.add_argument("--hostname", choices=["include", "hash", "omit"], default="hash")
    ap.add_argument("--output", default="sce_inference_v0_2.zip")
    up_group = ap.add_mutually_exclusive_group()
    up_group.add_argument("--upload", dest="upload", action="store_true", default=None,
                          help="after the run, send the summary numbers without asking (still never without a terminal)")
    up_group.add_argument("--no-upload", dest="upload", action="store_false", help="after the run, do not offer the upload")
    ap.add_argument("--dry-run", action="store_true", help="after the run, print exactly what the upload would send; send nothing")
    ap.add_argument("--endpoint", default=None, help=argparse.SUPPRESS)
    return ap


def main():
    ap = build_parser()
    a = ap.parse_args()
    if min(a.output_tokens, a.batch_size, a.repeats, a.sample_period) <= 0 or a.warmup < 0: ap.error("Invalid measurement parameter")

    import torch, transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available(): ap.error("PyTorch CUDA is not available")
    if a.gpu < 0 or a.gpu >= torch.cuda.device_count(): ap.error("Selected CUDA-visible GPU index is unavailable")
    props = torch.cuda.get_device_properties(a.gpu)
    dev = f"cuda:{a.gpu}"
    torch.cuda.set_device(a.gpu)
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    precision = "float16"
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16, device_map={"": a.gpu}).eval()
    prompts = [a.prompt] * a.batch_size
    enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
    sync = lambda: torch.cuda.synchronize(a.gpu)

    def forward():
        with torch.inference_mode(): model(**enc, use_cache=True)
    def decode():
        with torch.inference_mode():
            model.generate(**enc, max_new_tokens=a.output_tokens, min_new_tokens=a.output_tokens,
                           do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
    def first_token():
        with torch.inference_mode():
            model.generate(**enc, max_new_tokens=1, min_new_tokens=1,
                           do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
    for _ in range(a.warmup): decode()

    # Capture public host identity only in the explicitly selected form.
    host = socket.gethostname()
    host_record = host if a.hostname == "include" else hashlib.sha256(host.encode()).hexdigest() if a.hostname == "hash" else None
    rows, telemetry = [], []
    gpu = GPU(a.gpu, getattr(props, "uuid", None), getattr(props, "pci_bus_id", None))
    try:
        idle = gpu.measure(lambda: time.sleep(2), sync, a.sample_period)
        for s in idle["samples"]:
            telemetry.append({"repeat": 0, "phase": "idle", "gpu_index": a.gpu,
                              "nvml_gpu_index": gpu.index, "gpu_uuid": gpu.uuid, **s})
        for rep in range(1, a.repeats + 1):
            # Prefill is a separate prompt forward pass; full generation measures prefill + decode.
            pf = gpu.measure(forward, sync, a.sample_period)
            first = gpu.measure(first_token, sync, a.sample_period)
            full = gpu.measure(decode, sync, a.sample_period)
            generated = a.batch_size * a.output_tokens
            decode_energy = max(0.0, full["energy_j"] - pf["energy_j"])
            decode_time = max(0.0, full["elapsed_s"] - pf["elapsed_s"])
            row = {"repeat": rep, "gpu_index": a.gpu, "nvml_gpu_index": gpu.index,
                   "gpu_uuid": gpu.uuid, "model": a.model, "precision": precision, "batch_size": a.batch_size,
                   "input_tokens_per_request": int(enc["attention_mask"].sum(dim=1).max().item()),
                   "output_tokens_per_request": a.output_tokens, "prefill_latency_s": pf["elapsed_s"],
                   "ttft_s": first["elapsed_s"], "request_latency_s": full["elapsed_s"],
                   "energy_window_s": full["energy_window_s"],
                   "decode_itl_estimate_s": decode_time / max(1, a.output_tokens - 1),
                   "throughput_tokens_s": generated / full["elapsed_s"],
                   "prefill_energy_j": pf["energy_j"], "total_energy_j": full["energy_j"],
                   "decode_energy_incremental_j": decode_energy, "idle_power_w": idle["avg_power_w"],
                   "j_per_output_token_total": full["energy_j"] / generated,
                   "j_per_output_token_decode_incremental": decode_energy / max(1, generated-a.batch_size),
                   "energy_source": full["energy_source"], "energy_counter_j": full["energy_counter_j"],
                   "energy_integral_j": full["energy_integral_j"],
                   "energy_counter_integral_rel_diff": full["energy_counter_integral_rel_diff"],
                   "energy_check_passed": full["energy_check_passed"], "avg_power_w": full["avg_power_w"],
                   "avg_util_gpu_pct": full["avg_util_gpu_pct"], "max_temp_c": full["max_temp_c"]}
            rows.append(row)
            for phase, result in (("prefill", pf), ("first_token", first), ("full_generation", full)):
                for s in result["samples"]:
                    telemetry.append({"repeat": rep, "phase": phase, "gpu_index": a.gpu,
                                      "nvml_gpu_index": gpu.index, "gpu_uuid": gpu.uuid, **s})
            print(f"repeat {rep}/{a.repeats}: {row['throughput_tokens_s']:.2f} tok/s, {row['j_per_output_token_total']:.4f} J/output-token")
    finally: gpu.close()

    lat = [r["request_latency_s"] for r in rows]
    metric = {"runs": len(rows), "throughput_tokens_s_median": statistics.median(r["throughput_tokens_s"] for r in rows),
              "j_per_output_token_median": statistics.median(r["j_per_output_token_total"] for r in rows),
              "request_latency_p50_s": percentile(lat,.50), "request_latency_p95_s": percentile(lat,.95,20),
              "request_latency_p99_s": percentile(lat,.99,100), "ttft_p50_s": percentile([r["ttft_s"] for r in rows],.50),
              "decode_itl_p50_s": percentile([r["decode_itl_estimate_s"] for r in rows],.50),
              "coefficient_of_variation_latency": statistics.pstdev(lat)/statistics.mean(lat) if len(lat)>1 and statistics.mean(lat) else 0,
              "energy_sources": sorted(set(r["energy_source"] for r in rows)),
              "energy_check_threshold_rel_diff": 0.10,
              "latency_percentile_minimum_repeats": {"p95": 20, "p99": 100},
              "energy_check_pass_fraction": statistics.mean(r["energy_check_passed"] for r in rows
                                                            if r["energy_check_passed"] is not None)
                                                            if any(r["energy_check_passed"] is not None for r in rows) else None,
              "note": "p95 is omitted below 20 repeats and p99 below 100 repeats; reported percentiles are over repeated batch-level request latencies. TTFT is the separately timed one-token generate call, and decode ITL is an estimate from full minus prefill timing."}
    outdir = Path(".sce-" + uuid.uuid4().hex); outdir.mkdir()
    # The archive is the deliverable; do not leave an unadvertised second copy
    # of benchmark rows and telemetry in a hidden staging directory.
    atexit.register(shutil.rmtree, outdir, ignore_errors=True)
    model_revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    sysinfo = {"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
               "transformers": transformers.__version__, "cuda": torch.version.cuda,
               "cuda_visible_gpu_index": a.gpu, "nvml_gpu_index": gpu.index, "gpu_uuid": gpu.uuid,
               "gpu_mapping_method": gpu.mapping_method,
               "gpu_name": gpu.name, "gpu_compute_capability": f"{props.major}.{props.minor}",
               "gpu_vram_bytes": props.total_memory, "hostname_policy": a.hostname,
               "hostname": host_record, "collector_version": VERSION}
    manifest = {"experiment_id": outdir.name, "collector_version": VERSION,
                "created_utc": datetime.now(timezone.utc).isoformat(), "model": a.model,
                "resolved_model_revision": model_revision, "precision": precision,
                "workload": "Hugging Face Transformers causal language model generate, greedy decoding",
                "parameters": {"batch_size": a.batch_size, "output_tokens": a.output_tokens, "repeats": a.repeats,
                               "warmup_runs": a.warmup, "cuda_visible_gpu_index": a.gpu,
                               "nvml_gpu_index": gpu.index, "gpu_uuid": gpu.uuid,
                               "gpu_mapping_method": gpu.mapping_method,
                               "sample_period_s": a.sample_period},
                "privacy": {"prompt_text_saved": False, "model_responses_saved": False, "network_upload": False,
                            "hostname_policy": a.hostname}}
    artifacts = {"manifest.json": manifest, "system.json": sysinfo, "metrics.json": metric}
    for name, obj in artifacts.items(): (outdir/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, data in (("benchmark.csv", rows), ("telemetry.csv", telemetry)):
        with (outdir/name).open("w", newline="", encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=sorted({k for r in data for k in r})); w.writeheader(); w.writerows(data)
    files=list(outdir.iterdir()); (outdir/"checksum.txt").write_text("\n".join(f"{sha256(p)}  {p.name}" for p in files)+"\n",encoding="utf-8")
    with zipfile.ZipFile(a.output,"w",zipfile.ZIP_DEFLATED) as z:
        for p in outdir.iterdir(): z.write(p,p.name)
    shutil.rmtree(outdir, ignore_errors=True)
    print("Created", Path(a.output).resolve())
    print(json.dumps(metric, indent=2))
    return after_run(a.output, upload=a.upload, dry_run=a.dry_run, endpoint=a.endpoint)

if __name__ == "__main__": sys.exit(main())
