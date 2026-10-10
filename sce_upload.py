#!/usr/bin/env python3
"""SCETES upload tool 1.0.3: send collector results to SCETES, only with your explicit consent.

sce_infer.py measures and writes a local ZIP; it never uploads anything. This separate, optional tool reads one or more of
those ZIP files, shows you EXACTLY what would be sent, and sends it only after you agree. Prompts, model outputs and your
computer's name are not part of the data (they are not read at all). Standard library only; Python 3.9 or newer.

Typical use (measure 2-3 batch sizes with sce_infer.py first, then):
  python sce_upload.py run_b1.zip run_b8.zip run_b32.zip --baseline run_b1.zip --dry-run     # look, send nothing
  python sce_upload.py run_b1.zip run_b8.zip run_b32.zip --baseline run_b1.zip               # asks for your consent
  python sce_upload.py --delete <result_url_or_token>                                        # delete what you sent
Payload format: schema_version 1, see api/API.md.
"""
from __future__ import annotations

import argparse, csv, hashlib, io, json, re, sys, zipfile
from typing import Optional
from urllib import error, parse, request

TOOL_VERSION = "1.0.3"
SCHEMA_VERSION = 1
CONSENT_VERSION = "2026-10-08"
DEFAULT_ENDPOINT = "https://scetes.com/api/check"
CONSENT_TEXT = (
    "To calculate your estimate, SCETES sends the measurements listed below to scetes.com: GPU model and memory, the model "
    "name you ran, batch size, output length, speed, latency, GPU energy and utilisation, how much the GPU slowed down (clock "
    "speed, reasons reported by the driver) and how much GPU memory was used, and the prices you entered. Your "
    "prompts, the model's outputs and your computer's name are never sent. Results are stored for 12 months; you can delete "
    "them at any time with the link you receive. Details: https://scetes.com/en/privacy/"
)
# Separate and optional; the default is no and the result is the same either way (Privacy: "Statistics and benchmarks").
STATISTICS_TEXT = (
    "Optional: may SCETES also use these measurements, without your e-mail address, in anonymised, aggregated statistics "
    "and benchmarks (for example the median energy per token for a GPU type)? Only values combined from at least five "
    "different customers are published; if that cannot be established, nothing is published, and never your single result. Your result is the same either way, and you can withdraw at any "
    "time by deleting the result or by writing to other@scetes.com."
)
PRECISIONS = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16", "int8": "int8", "int4": "int4"}
# Same rule as the preregistered E0 experiment: SM clock drop above 10 % or temperature above 80 C marks a run as throttled.
CLOCK_DROP = 0.90
TEMP_LIMIT_C = 80.0


class UploadError(Exception):
    pass


def _num(value) -> Optional[float]:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def read_archive(path: str) -> dict:
    """Read the files of one sce_infer.py archive and verify their SHA-256 checksums."""
    try:
        z = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise UploadError(f"{path}: cannot open as a ZIP archive ({exc})")
    with z:
        names = set(z.namelist())
        missing = {"manifest.json", "system.json", "metrics.json", "benchmark.csv", "telemetry.csv", "checksum.txt"} - names
        if missing:
            raise UploadError(f"{path}: not a complete collector archive, missing {', '.join(sorted(missing))}")
        for line in z.read("checksum.txt").decode("utf-8").splitlines():
            if not line.strip():
                continue
            digest, _, name = line.partition("  ")
            if name not in names or hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise UploadError(f"{path}: checksum mismatch for {name}; the archive was changed after it was written")
        return {
            "path": path,
            "manifest": json.loads(z.read("manifest.json")),
            "system": json.loads(z.read("system.json")),
            "metrics": json.loads(z.read("metrics.json")),
            "benchmark": list(csv.DictReader(io.StringIO(z.read("benchmark.csv").decode("utf-8")))),
            "telemetry": list(csv.DictReader(io.StringIO(z.read("telemetry.csv").decode("utf-8")))),
        }


def mean_of(rows, key) -> Optional[float]:
    values = [v for v in (_num(r.get(key)) for r in rows) if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def throttling_flagged(telemetry) -> tuple:
    """(flagged, checked). Unknown counts as flagged: a run whose clocks were not recorded is never reported as clean."""
    busy = [r for r in telemetry if r.get("phase") != "idle"]
    clocks = [v for v in (_num(r.get("clock_sm_mhz")) for r in busy) if v is not None]
    temps = [v for v in (_num(r.get("temp_c")) for r in busy) if v is not None]
    if not clocks:
        return True, False
    return (min(clocks) < CLOCK_DROP * max(clocks)) or (bool(temps) and max(temps) > TEMP_LIMIT_C), True


# Reasons reported by NVML (bit masks). Idle and "applications clocks" are normal settings, not throttling, so they are never reported.
REASON_BITS = {0x4: "power_cap", 0x8: "hardware_slowdown", 0x10: "sync_boost", 0x20: "thermal_slowdown", 0x40: "hardware_thermal_slowdown",
               0x80: "power_brake", 0x100: "display_clock_setting"}
REASON_MIN_SHARE = 0.01


def throttling_detail(telemetry) -> Optional[dict]:
    """How much speed throttling cost, as measured: share of busy time with a clock >10 % below the maximum, the mean clock
    as a percentage of the maximum, and the reasons NVML reported. None when clocks were not recorded."""
    busy = [r for r in telemetry if r.get("phase") != "idle"]
    rows = [(_num(r.get("clock_sm_mhz")), _num(r.get("max_clock_sm_mhz")), _num(r.get("throttle_reasons"))) for r in busy]
    clocks = [c for c, _, _ in rows if c is not None]
    if not clocks:
        return None
    maxima = [m for _, m, _ in rows if m]
    reference = max(maxima) if maxima else max(clocks)  # without a reported maximum, the highest observed clock is used
    reasons_seen = [int(x) for _, _, x in rows if x is not None]
    detail = {
        "reduced_clock_time_pct": round(100 * sum(1 for c in clocks if c < CLOCK_DROP * reference) / len(clocks), 1),
        "mean_clock_pct_of_max": round(100 * (sum(clocks) / len(clocks)) / reference, 1),
        "reasons_reported": bool(reasons_seen),
        "reasons": [],
    }
    if reasons_seen:
        for bit, name in REASON_BITS.items():
            if sum(1 for x in reasons_seen if x & bit) / len(reasons_seen) >= REASON_MIN_SHARE:
                detail["reasons"].append(name)
    return detail


def memory_detail(telemetry) -> Optional[dict]:
    """Peak GPU memory in use during the run and the total, in GB. None when memory was not recorded."""
    used = [v for v in (_num(r.get("mem_used_mib")) for r in telemetry) if v is not None]
    total = [v for v in (_num(r.get("mem_total_mib")) for r in telemetry) if v is not None]
    if not used or not total:
        return None
    return {"peak_used_gb": round(max(used) / 1024, 2), "total_gb": round(max(total) / 1024, 2)}


def idle_detail(telemetry) -> Optional[dict]:
    """Share of the measured time in which, according to the driver, no GPU kernel was running: 100 % minus the mean NVML utilisation over the measured
    phases (NVML utilisation is the share of time in which at least one kernel was executing). None when not recorded."""
    util = [v for v in (_num(r.get("util_gpu_pct")) for r in telemetry if r.get("phase") != "idle") if v is not None]
    if not util:
        return None
    return {"idle_time_pct": round(max(0.0, 100.0 - sum(util) / len(util)), 1)}


def run_entry(archive: dict, is_baseline: bool, model_label: Optional[str]) -> dict:
    m, mf, bench = archive["metrics"], archive["manifest"], archive["benchmark"]
    flagged, _ = throttling_flagged(archive["telemetry"])
    seconds_to_ms = lambda v: None if v is None else round(v * 1000, 3)
    metrics = {
        "tokens_per_s": round(m["throughput_tokens_s_median"], 6),
        "j_per_output_token": round(m["j_per_output_token_median"], 6),
        "latency_ms": {"p50": seconds_to_ms(m["request_latency_p50_s"]), "p95": seconds_to_ms(m.get("request_latency_p95_s")),
                       "p99": seconds_to_ms(m.get("request_latency_p99_s"))},
    }
    for key, value in (("gpu_util_pct_mean", mean_of(bench, "avg_util_gpu_pct")), ("gpu_power_w_mean", mean_of(bench, "avg_power_w"))):
        if value is not None:
            metrics[key] = value
    params = mf["parameters"]
    entry = {
        "model_id": model_label or str(mf["model"])[:200],
        "precision": PRECISIONS.get(str(mf.get("precision")), "other"),
        "batch_size": int(params["batch_size"]),
        "output_tokens": int(params["output_tokens"]),
        "repeats": int(params["repeats"]),
        "metrics": metrics,
        # All repeats must have passed the counter/integral comparison; "not checked" does not count as passed.
        "quality": {"energy_check_ok": (m.get("energy_check_pass_fraction") or 0) >= 1.0, "throttling_flagged": flagged},
    }
    for key, value in (("throttling", throttling_detail(archive["telemetry"])), ("memory", memory_detail(archive["telemetry"])), ("idle", idle_detail(archive["telemetry"]))):
        if value is not None:
            entry[key] = value
    if is_baseline:
        entry["is_baseline"] = True
    return entry


def build_payload(archives, baseline_path=None, prices=None, contact=None, model_label=None, instance_type=None, statistics=False) -> dict:
    """Pure function: archives -> upload payload. Only allow-listed values are read from the archives."""
    if not archives:
        raise UploadError("give at least one result ZIP")
    if baseline_path is not None and baseline_path not in [a["path"] for a in archives]:
        raise UploadError("--baseline must be one of the given ZIP files")
    first = archives[0]
    gpus = {a["system"]["gpu_name"] for a in archives}
    if len(gpus) > 1:
        raise UploadError("all archives must come from the same GPU model")
    system = {"gpu_model": str(first["system"]["gpu_name"])[:80], "gpu_vram_gb": round(first["system"]["gpu_vram_bytes"] / 1024 ** 3, 2), "gpu_count": 1}
    if instance_type:
        system["instance_type"] = instance_type
    payload = {
        "schema_version": SCHEMA_VERSION,
        "collector_version": str(first["manifest"]["collector_version"]),
        "consent": {"version": CONSENT_VERSION, "accepted": True, "statistics": statistics is True},
        "system": system,
        "runs": [run_entry(a, a["path"] == baseline_path, model_label) for a in archives],
    }
    if prices:
        payload["prices"] = prices
    if contact:
        payload["contact"] = contact
    return payload


def comparison_warnings(archives) -> list:
    """Comparisons are only meaningful when a single variable changes; say so instead of silently allowing noise."""
    notes = []
    for label, get in (("model", lambda a: a["manifest"]["model"]), ("precision", lambda a: a["manifest"].get("precision")),
                       ("output length", lambda a: a["manifest"]["parameters"]["output_tokens"])):
        if len({get(a) for a in archives}) > 1:
            notes.append(f"the runs differ in {label}; comparisons between them are less meaningful")
    return notes


def print_measured(m) -> None:
    """The result never ends empty: measured values as they were, what throttling cost and how much GPU memory was free."""
    if not m:
        return
    tps, jpt = m["throughput_tokens_s"], m["j_per_output_token"]
    print(f"Measured as it was: {tps['min']} to {tps['max']} tokens/s, {jpt['min']} to {jpt['max']} J per output token.")
    t = m.get("throttling")
    if t:
        why = ", ".join(t["reasons"]) if t["reasons"] else ("no reason reported by the driver" if t["reasons_reported"] else "reasons not available")
        print(f"Throttling: up to {t['max_reduced_clock_time_pct']} % of the time with a reduced clock; mean clock down to {t['min_mean_clock_pct_of_max']} % of the maximum ({why}).")
    mem = m.get("memory")
    if mem:
        print(f"GPU memory: up to {mem['peak_used_gb']} of {mem['total_gb']} GB in use, {mem['free_gb']} GB free (room for a larger batch).")
    idle = m.get("idle")
    if idle:
        print(f"GPU idle share: {idle['min_idle_time_pct']} to {idle['max_idle_time_pct']} % of the measured time no GPU kernel was running "
              "(according to the driver; measured during this run, not in production; not a savings claim). Driver-reported utilisation only "
              "shows whether a kernel was running, not how efficiently the GPU was used: a low idle share does not mean efficient use.")
    print("Next step: SCETES can repeat the measurement under controlled conditions and look for the cause (Diagnostic).")


def check_endpoint(url: str) -> str:
    p = parse.urlparse(url)
    local = p.hostname in ("127.0.0.1", "localhost")
    if p.scheme != "https" and not (p.scheme == "http" and local):
        raise UploadError("the endpoint must use https")
    return url.rstrip("/")


def http(method: str, url: str, body: Optional[dict] = None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = request.Request(url, data=data, method=method, headers={"Content-Type": "application/json", "User-Agent": f"sce-upload/{TOOL_VERSION}"})
    try:
        with request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else None)
    except error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, {"error": raw.decode("utf-8", "replace")[:200]}
    except error.URLError as exc:
        raise UploadError(f"cannot reach {url}: {exc.reason}")


def token_of(value: str) -> str:
    """Accepts the result page link (.../check/#TOKEN), the API link (.../api/check/TOKEN) or the bare token."""
    found = re.search(r"(?:^|[/#])([A-Za-z0-9_-]{43})/?$", value.strip())
    if not found:
        raise UploadError("that is not a valid result link or token")
    return found.group(1)


def page_link(endpoint: str, token: str) -> str:
    """Human-readable result page. The token sits after '#', which browsers never send to a server."""
    parts = parse.urlsplit(endpoint)
    return f"{parts.scheme}://{parts.netloc}/en/check/#{token}"


def ask_consent(stdin=None, out=None) -> bool:
    out = out or sys.stdout
    print("\n" + CONSENT_TEXT + "\n", file=out)
    out.write("Send these measurements? [y/N] ")
    out.flush()
    answer = (stdin or sys.stdin).readline().strip().lower() if stdin is not None or sys.stdin.isatty() else ""
    return answer in ("y", "yes")


def _line(prompt, stdin, out):
    out.write(prompt); out.flush()
    return (stdin or sys.stdin).readline().strip()


def ask_prices(stdin=None, out=None):
    """Optional, interactive: the customer's own prices make the result a money figure. Every question can be skipped with Enter."""
    out = out or sys.stdout
    print("\nOptional: enter your own prices to see the result in money (press Enter to skip each question).", file=out)
    currency = _line("Currency, for example EUR: ", stdin, out).upper()
    if not re.fullmatch(r"[A-Z]{3}", currency):
        return None
    prices = {"currency": currency}
    for key, label in (("gpu_hour_price", "Price of one GPU hour (a rental price already includes electricity): "), ("kwh_price", "Price of one kWh: "),
                       ("monthly_output_tokens", "Output tokens you generate per month: ")):
        raw = _line(label, stdin, out).replace(",", ".")
        try:
            value = float(raw)
        except ValueError:
            continue
        if value > 0:
            prices[key] = value
    return prices if len(prices) > 1 else None


def ask_contact(stdin=None, out=None):
    """Optional, interactive and separate from the main consent: an e-mail address for a follow-up offer. Skipping it never limits the result."""
    out = out or sys.stdout
    print("\nOptional: if you would like SCETES to send you an offer based on this result (for example a controlled-conditions Diagnostic), "
          "enter an e-mail address. You get your result either way. Press Enter to skip.", file=out)
    email = _line("E-mail address: ", stdin, out)
    if not email:
        return None
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
        print("That does not look like an e-mail address; skipping.", file=out)
        return None
    answer = _line("I agree that SCETES may contact me at this address about this result. You can withdraw at any time. [y/N] ", stdin, out).lower()
    return {"email": email, "follow_up_consent": True} if answer in ("y", "yes") else None


def ask_statistics(stdin=None, out=None) -> bool:
    """Optional, interactive and separate from the main consent; only an explicit yes counts."""
    out = out or sys.stdout
    print("\n" + STATISTICS_TEXT, file=out)
    return _line("Allow use in anonymised statistics? [y/N] ", stdin, out).lower() in ("y", "yes")


def main(argv=None, stdin=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archives", nargs="*", help="result ZIP files written by sce_infer.py (one per configuration)")
    ap.add_argument("--baseline", help="the ZIP that represents your CURRENT configuration (others are compared with it)")
    ap.add_argument("--model-label", help="send this text instead of the real model name (for private models)")
    ap.add_argument("--instance-type", help="optional, for example g4dn.xlarge")
    ap.add_argument("--currency", help="ISO code of your prices, for example EUR")
    ap.add_argument("--gpu-hour-price", type=float, help="price of one GPU hour (rental already includes electricity)")
    ap.add_argument("--kwh-price", type=float, help="price of one kWh")
    ap.add_argument("--monthly-output-tokens", type=float, help="your monthly output tokens, to express the result in money per month")
    ap.add_argument("--email", help="optional: SCETES may contact you about this result (needs --follow-up-consent)")
    ap.add_argument("--follow-up-consent", action="store_true", help="I agree that SCETES contacts me at the e-mail above")
    ap.add_argument("--statistics-consent", action="store_true", help="optional: SCETES may also use these measurements in anonymised, aggregated statistics")
    ap.add_argument("--dry-run", action="store_true", help="show exactly what would be sent, send nothing, ask nothing")
    ap.add_argument("--consent", action="store_true", help="I have read the consent text and agree (skips the question; for scripts)")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--delete", metavar="LINK_OR_TOKEN", help="delete a result you sent earlier")
    a = ap.parse_args(argv)
    try:
        endpoint = check_endpoint(a.endpoint)
        if a.delete:
            status, body = http("DELETE", f"{endpoint}/{token_of(a.delete)}")
            print("Deleted." if status == 204 else f"Not deleted ({status}): {(body or {}).get('error', 'unknown error')}")
            return 0 if status == 204 else 1
        if not a.archives:
            ap.error("give at least one result ZIP, or use --delete")
        if a.email and not a.follow_up_consent:
            ap.error("--email needs --follow-up-consent")
        prices = None
        if any(v is not None for v in (a.gpu_hour_price, a.kwh_price, a.monthly_output_tokens)):
            if not a.currency:
                ap.error("--currency is needed with prices")
            prices = {"currency": a.currency.upper()}
            for key, value in (("gpu_hour_price", a.gpu_hour_price), ("kwh_price", a.kwh_price), ("monthly_output_tokens", a.monthly_output_tokens)):
                if value is not None:
                    prices[key] = value
        archives = [read_archive(p) for p in a.archives]
        contact = {"email": a.email, "follow_up_consent": True} if a.email else None
        interactive = not a.dry_run and not a.consent and (stdin is not None or sys.stdin.isatty())
        if interactive and prices is None:
            prices = ask_prices(stdin)
        if interactive and contact is None:
            contact = ask_contact(stdin)
        statistics = a.statistics_consent or (interactive and ask_statistics(stdin))
        payload = build_payload(archives, a.baseline, prices, contact, a.model_label, a.instance_type, statistics)
        for note in comparison_warnings(archives):
            print(f"Note: {note}.", file=sys.stderr)
        if any(r["quality"]["throttling_flagged"] for r in payload["runs"]):
            print("Note: at least one run is marked as throttled (the GPU lowered its clock speed or ran above 80 C) or its clocks were not recorded. "
                  "Such runs are shown but not used for the comparison, so the result may contain no estimate. "
                  "Repeat the measurement under controlled conditions if you need one.", file=sys.stderr)
        print("This is exactly what would be sent:\n" + json.dumps(payload, indent=2, ensure_ascii=False))
        if a.dry_run:
            print("\nDry run: nothing was sent.")
            return 0
        if not (a.consent or ask_consent(stdin)):
            print("No consent: nothing was sent.")
            return 0
        status, body = http("POST", endpoint, payload)
        if status != 201:
            detail = "; ".join(f"{d['path']}: {d['message']}" for d in (body or {}).get("details", [])[:5])
            print(f"Not accepted ({status}): {(body or {}).get('error', 'unknown error')} {detail}".strip())
            return 1
        head = body["result"]["headline"]
        print("\nSent. Keep this link private; it is the only way to read or delete your result:")
        print(f"  {page_link(endpoint, body['token'])}\n  stored until {body['expires_at']}")
        print(f"  (the same result as data: {parse.urljoin(endpoint, body['result_url'])})")
        if head.get("available"):
            if head.get("confidence") == "indicative":
                print("INDICATIVE ONLY: the GPU was throttling while you measured, so treat this as a pointer, not a confirmed result. "
                      "SCETES can confirm it under controlled conditions (Diagnostic).")
            print(f"Best measured alternative: run {head['best_run']}, cost ratio {head['cost_ratio']} of the baseline"
                  + (f", p95 latency ratio {head['p95_latency_ratio']}" if head.get("p95_latency_ratio") else "") + ".")
            if head.get("potential_monthly"):
                pm = head["potential_monthly"]
                print(f"Estimated potential at {pm['at_monthly_output_tokens']:,.0f} output tokens per month: {pm['amount']} {pm['currency']}.")
        else:
            print(f"No comparison available: {head.get('reason')}.")
        print_measured(body["result"].get("measured"))
        print(body["result"]["basis_text"])
        return 0
    except UploadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
