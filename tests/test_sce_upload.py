import hashlib, io, json, os, shutil, subprocess, sys, tempfile, unittest, zipfile
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sce_upload as up  # noqa: E402

HERE = os.path.dirname(__file__)


def make_archive(path, batch=1, tps=50.0, j=4.0, p50=2.5, p95=2.6, clocks=(1590, 1590, 1585), temp=60, energy_ok=1.0, hostname="secret-box", detail=None):
    manifest = {"collector_version": "0.2.0", "model": "Qwen/Qwen2.5-1.5B-Instruct", "precision": "float16",
                "parameters": {"batch_size": batch, "output_tokens": 128, "repeats": 20},
                "privacy": {"prompt_text_saved": False}}
    system = {"gpu_name": "Tesla T4", "gpu_vram_bytes": 15843721216, "hostname": hostname, "gpu_uuid": "GPU-secret-uuid"}
    metrics = {"throughput_tokens_s_median": tps, "j_per_output_token_median": j, "request_latency_p50_s": p50,
               "request_latency_p95_s": p95, "request_latency_p99_s": None, "energy_check_pass_fraction": energy_ok}
    bench = "repeat,avg_util_gpu_pct,avg_power_w\n1,90,60\n2,92,62\n"
    extra = "" if detail is None else ",max_clock_sm_mhz,throttle_reasons,mem_used_mib,mem_total_mib"
    rows = ["repeat,phase,clock_sm_mhz,temp_c" + extra, "0,idle,300,40" + ("" if detail is None else ",1590,1,1000,15360")]
    rows += [f"1,full_generation,{c},{temp}" + ("" if detail is None else f",1590,{detail['reasons']},{detail['used']},15360") for c in clocks]
    files = {"manifest.json": json.dumps(manifest), "system.json": json.dumps(system), "metrics.json": json.dumps(metrics),
             "benchmark.csv": bench, "telemetry.csv": "\n".join(rows) + "\n"}
    files["checksum.txt"] = "".join(f"{hashlib.sha256(v.encode()).hexdigest()}  {k}\n" for k, v in files.items())
    with zipfile.ZipFile(path, "w") as z:
        for k, v in files.items():
            z.writestr(k, v)
    return path


class BuildPayload(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        self.b1 = make_archive(os.path.join(self.dir, "b1.zip"))
        self.b8 = make_archive(os.path.join(self.dir, "b8.zip"), batch=8, tps=300.0, j=0.8, p50=3.0, p95=3.12)

    def payload(self, **kw):
        return up.build_payload([up.read_archive(self.b1), up.read_archive(self.b8)], baseline_path=self.b1, **kw)

    def test_mapping(self):
        p = self.payload()
        self.assertEqual(p["schema_version"], 1)
        self.assertEqual(p["system"], {"gpu_model": "Tesla T4", "gpu_vram_gb": 14.76, "gpu_count": 1})
        r0, r1 = p["runs"]
        self.assertTrue(r0["is_baseline"]); self.assertNotIn("is_baseline", r1)
        self.assertEqual((r0["precision"], r0["batch_size"], r1["batch_size"]), ("fp16", 1, 8))
        self.assertEqual(r0["metrics"]["latency_ms"], {"p50": 2500.0, "p95": 2600.0, "p99": None})
        self.assertEqual(r0["metrics"]["gpu_util_pct_mean"], 91.0)
        self.assertEqual(r0["quality"], {"energy_check_ok": True, "throttling_flagged": False})

    def test_nothing_private_is_sent(self):
        text = json.dumps(self.payload())
        for secret in ("secret-box", "GPU-secret-uuid"):
            self.assertNotIn(secret, text)

    def test_model_label_hides_the_model_name(self):
        text = json.dumps(self.payload(model_label="private-model"))
        self.assertIn("private-model", text); self.assertNotIn("Qwen", text)

    def test_throttling_rule_matches_the_preregistered_e0_rule(self):
        self.assertEqual(up.throttling_flagged([{"phase": "full_generation", "clock_sm_mhz": "1000", "temp_c": "60"},
                                                {"phase": "full_generation", "clock_sm_mhz": "899", "temp_c": "60"}]), (True, True))
        self.assertEqual(up.throttling_flagged([{"phase": "full_generation", "clock_sm_mhz": "1000", "temp_c": "60"},
                                                {"phase": "full_generation", "clock_sm_mhz": "900", "temp_c": "60"}]), (False, True))
        self.assertEqual(up.throttling_flagged([{"phase": "full_generation", "clock_sm_mhz": "1000", "temp_c": "81"}]), (True, True))
        # idle clocks are ignored, and unknown clocks are never reported as clean
        self.assertEqual(up.throttling_flagged([{"phase": "idle", "clock_sm_mhz": "300", "temp_c": "40"}, {"phase": "full_generation", "clock_sm_mhz": "1500", "temp_c": "60"}]), (False, True))
        self.assertEqual(up.throttling_flagged([{"phase": "full_generation", "clock_sm_mhz": "", "temp_c": "60"}]), (True, False))

    def test_energy_check_must_have_passed_in_every_repeat(self):
        for fraction, expected in ((1.0, True), (0.95, False), (None, False)):
            path = make_archive(os.path.join(self.dir, f"e{fraction}.zip"), energy_ok=fraction)
            self.assertEqual(up.build_payload([up.read_archive(path)])["runs"][0]["quality"]["energy_check_ok"], expected)

    def test_a_modified_archive_is_refused(self):
        bad = os.path.join(self.dir, "bad.zip")
        with zipfile.ZipFile(self.b1) as src, zipfile.ZipFile(bad, "w") as dst:
            for n in src.namelist():
                dst.writestr(n, src.read(n).replace(b"2.5", b"1.1") if n == "metrics.json" else src.read(n))
        with self.assertRaises(up.UploadError):
            up.read_archive(bad)

    def test_baseline_must_be_one_of_the_archives_and_gpus_must_match(self):
        with self.assertRaises(up.UploadError):
            up.build_payload([up.read_archive(self.b1)], baseline_path="other.zip")
        other = up.read_archive(self.b8); other["system"]["gpu_name"] = "A100"
        with self.assertRaises(up.UploadError):
            up.build_payload([up.read_archive(self.b1), other])

    def test_endpoint_must_be_https_except_localhost(self):
        up.check_endpoint("https://scetes.com/api/check"); up.check_endpoint("http://127.0.0.1:8080/api/check")
        with self.assertRaises(up.UploadError):
            up.check_endpoint("http://scetes.com/api/check")


class Tokens(unittest.TestCase):
    def test_token_is_found_in_every_link_form(self):
        tok = "A" * 21 + "-_" + "b" * 20
        for value in (tok, f"https://scetes.com/en/check/#{tok}", f"https://scetes.com/api/check/{tok}", f"https://scetes.com/api/check/{tok}/"):
            self.assertEqual(up.token_of(value), tok)
        for bad in ("", "short", "x" * 44, "https://scetes.com/en/check/"):
            with self.assertRaises(up.UploadError):
                up.token_of(bad)


class MeasuredDetail(unittest.TestCase):
    def tele(self, rows):
        return [{"phase": "full_generation", "clock_sm_mhz": str(c), "max_clock_sm_mhz": "1590", "throttle_reasons": str(r),
                 "mem_used_mib": str(u), "mem_total_mib": "15360"} for c, r, u in rows]

    def test_throttling_loss_is_share_of_time_mean_clock_and_reasons(self):
        # 4 samples: two at full speed, two at 795 MHz (50 %) with the power cap (0x4); idle bit 0x1 must never be reported
        d = up.throttling_detail(self.tele([(1590, 1, 0), (1590, 1, 0), (795, 5, 0), (795, 5, 0)]))
        self.assertEqual(d["reduced_clock_time_pct"], 50.0)
        self.assertEqual(d["mean_clock_pct_of_max"], 75.0)
        self.assertEqual(d["reasons"], ["power_cap"]); self.assertTrue(d["reasons_reported"])

    def test_old_archives_without_the_new_columns_still_work(self):
        d = up.throttling_detail([{"phase": "full_generation", "clock_sm_mhz": "1000"}, {"phase": "full_generation", "clock_sm_mhz": "500"}])
        self.assertEqual((d["reduced_clock_time_pct"], d["reasons_reported"], d["reasons"]), (50.0, False, []))
        self.assertIsNone(up.throttling_detail([{"phase": "full_generation", "clock_sm_mhz": ""}]))
        self.assertIsNone(up.memory_detail([{"phase": "full_generation"}]))

    def test_idle_share_is_100_minus_mean_utilisation_over_measured_phases_only(self):
        rows = [{"phase": "idle", "util_gpu_pct": "0"}, {"phase": "full_generation", "util_gpu_pct": "40"},
                {"phase": "full_generation", "util_gpu_pct": "60"}, {"phase": "prefill", "util_gpu_pct": ""}]
        self.assertEqual(up.idle_detail(rows), {"idle_time_pct": 50.0})
        self.assertIsNone(up.idle_detail([{"phase": "full_generation", "util_gpu_pct": ""}]))

    def test_memory_headroom_inputs_are_peak_used_and_total_in_gb(self):
        self.assertEqual(up.memory_detail(self.tele([(1590, 0, 4096), (1590, 0, 6144), (1590, 0, 5000)])), {"peak_used_gb": 6.0, "total_gb": 15.0})

    def test_only_aggregates_enter_the_payload(self):
        d = tempfile.mkdtemp(); self.addCleanup(shutil.rmtree, d)
        path = make_archive(os.path.join(d, "a.zip"), clocks=(1590, 800, 800), detail={"reasons": 5, "used": 5120})
        payload = up.build_payload([up.read_archive(path)], path)
        run = payload["runs"][0]
        self.assertEqual(run["memory"], {"peak_used_gb": 5.0, "total_gb": 15.0})
        self.assertEqual(run["throttling"]["reasons"], ["power_cap"])
        self.assertNotIn("secret-box", json.dumps(payload))


class InteractivePrompts(unittest.TestCase):
    def test_prices_and_a_separate_follow_up_consent_are_asked_and_can_each_be_skipped(self):
        d = tempfile.mkdtemp(); self.addCleanup(shutil.rmtree, d)
        b1 = make_archive(os.path.join(d, "b1.zip")); b8 = make_archive(os.path.join(d, "b8.zip"), batch=8, tps=300.0, j=0.8)
        # currency, GPU hour, kWh, monthly tokens, e-mail, e-mail consent, main consent: all answered, send declined at the end
        out = io.StringIO()
        with redirect_stdout(out):
            up.main([b1, b8, "--baseline", b1, "--endpoint", "http://127.0.0.1:9/api/check"], stdin=io.StringIO("eur\n1\n0.2\n100000000\nada@example.com\ny\nn\n"))
        text = out.getvalue()
        self.assertIn('"gpu_hour_price": 1.0', text); self.assertIn('"email": "ada@example.com"', text); self.assertIn('"follow_up_consent": true', text)
        self.assertIn("No consent: nothing was sent.", text)

    def test_skipping_everything_sends_no_prices_and_no_address(self):
        d = tempfile.mkdtemp(); self.addCleanup(shutil.rmtree, d)
        b1 = make_archive(os.path.join(d, "b1.zip"))
        out = io.StringIO()
        with redirect_stdout(out):
            up.main([b1, "--endpoint", "http://127.0.0.1:9/api/check"], stdin=io.StringIO("\n\nn\n"))
        text = out.getvalue()
        self.assertNotIn('"prices"', text); self.assertNotIn('"contact"', text)

    def test_an_address_without_a_yes_is_dropped_and_dry_run_and_consent_flag_never_prompt(self):
        d = tempfile.mkdtemp(); self.addCleanup(shutil.rmtree, d)
        b1 = make_archive(os.path.join(d, "b1.zip"))
        out = io.StringIO()
        with redirect_stdout(out):
            up.main([b1, "--endpoint", "http://127.0.0.1:9/api/check"], stdin=io.StringIO("\nada@example.com\nn\nn\n"))
        self.assertNotIn("ada@example.com", out.getvalue().split("This is exactly what would be sent")[1].split("Send these")[0])
        for flag in ("--dry-run", "--consent"):
            out = io.StringIO()
            with redirect_stdout(out):
                up.main([b1, flag, "--endpoint", "http://127.0.0.1:9/api/check"], stdin=io.StringIO(""))
            self.assertNotIn("Optional:", out.getvalue())


class StatisticsConsent(unittest.TestCase):
    def setUp(self):
        d = tempfile.mkdtemp(); self.addCleanup(shutil.rmtree, d)
        self.b1 = make_archive(os.path.join(d, "b1.zip"))

    def sent(self, args, stdin_text):
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()), mock.patch.object(up, "http") as http:
            http.return_value = (201, {"result": {"headline": {"available": False, "reason": "x"}, "basis_text": "b"}, "token": "a" * 43, "result_url": "/api/check/" + "a" * 43, "expires_at": "2027"})
            up.main([self.b1] + args, stdin=io.StringIO(stdin_text) if stdin_text is not None else None)
        return http.call_args[0][2]["consent"] if http.called else None, out.getvalue()

    def test_default_is_no_and_only_an_explicit_yes_counts(self):
        for answer, expected in (("\n", False), ("n\n", False), ("maybe\n", False), ("y\n", True)):
            consent, out = self.sent([], "\n\n" + answer + "y\n")
            self.assertEqual(consent, {"version": up.CONSENT_VERSION, "accepted": True, "statistics": expected}, answer)
            self.assertIn("at least five different customers", out); self.assertIn("nothing is published", out)

    def test_scripts_need_the_flag_and_the_main_consent_never_implies_it(self):
        self.assertFalse(self.sent(["--consent"], None)[0]["statistics"])
        self.assertTrue(self.sent(["--consent", "--statistics-consent"], None)[0]["statistics"])

    def test_the_answer_is_shown_in_the_payload_before_the_main_consent(self):
        _, out = self.sent([], "\n\ny\nn\n")
        self.assertIn('"statistics": true', out.split("This is exactly what would be sent")[1].split("Send these")[0])


class Release(unittest.TestCase):
    ROOT = os.path.join(HERE, "..")

    def test_checksums_match_the_scripts_so_a_change_without_a_new_release_fails(self):
        lines = [l.split("  ") for l in open(os.path.join(self.ROOT, "SHA256SUMS"), encoding="utf-8").read().splitlines() if l.strip()]
        self.assertEqual(sorted(n for _, n in lines), ["sce_infer.py", "sce_upload.py"])
        for digest, name in lines:
            with open(os.path.join(self.ROOT, name), "rb") as f:
                self.assertEqual(hashlib.sha256(f.read()).hexdigest(), digest, f"{name} changed: update SHA256SUMS and the version")

    def test_both_tools_report_the_same_release(self):
        import re
        infer = open(os.path.join(self.ROOT, "sce_infer.py"), encoding="utf-8").read()
        self.assertEqual(re.search(r'^VERSION = "([^"]+)"', infer, re.M).group(1), up.TOOL_VERSION)
        self.assertEqual(up.TOOL_VERSION, "1.0.5.dev0")


class Cli(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        self.b1 = make_archive(os.path.join(self.dir, "b1.zip"))
        self.b8 = make_archive(os.path.join(self.dir, "b8.zip"), batch=8, tps=300.0, j=0.8, p50=3.0, p95=3.12)

    def run_cli(self, args, stdin_text=None):
        out, err = io.StringIO(), io.StringIO()
        stdin = io.StringIO(stdin_text) if stdin_text is not None else None
        with redirect_stdout(out), redirect_stderr(err), mock.patch.object(up, "http") as http:
            http.return_value = (201, {"result": {"headline": {"available": False, "reason": "x"}, "basis_text": "b"}, "token": "a" * 43, "result_url": "/api/check/" + "a" * 43, "expires_at": "2027"})
            code = up.main(args, stdin=stdin)
        return code, out.getvalue(), http

    def test_dry_run_sends_nothing_and_asks_nothing(self):
        code, out, http = self.run_cli([self.b1, self.b8, "--baseline", self.b1, "--dry-run"])
        self.assertEqual(code, 0); http.assert_not_called(); self.assertIn("exactly what would be sent", out)

    def test_without_consent_nothing_is_sent(self):
        for answer in ("", "n\n", "maybe\n"):
            code, out, http = self.run_cli([self.b1], stdin_text=answer)
            self.assertEqual(code, 0); http.assert_not_called(); self.assertIn("No consent", out)

    def test_non_interactive_without_the_consent_flag_sends_nothing(self):
        code, out, http = self.run_cli([self.b1])
        http.assert_not_called(); self.assertIn("No consent", out)

    def test_yes_or_the_consent_flag_sends(self):
        # interactive order: prices (skipped with Enter), e-mail (skipped), statistics (skipped), then the main consent question
        for kwargs in (dict(stdin_text="\n\n\ny\n"), dict(stdin_text=None)):
            args = [self.b1] + (["--consent"] if kwargs["stdin_text"] is None else [])
            code, out, http = self.run_cli(args, **kwargs)
            self.assertEqual(code, 0); http.assert_called_once(); self.assertEqual(http.call_args[0][0], "POST")

    def test_e_mail_needs_its_own_consent(self):
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                self.run_cli([self.b1, "--email", "a@b.cz", "--dry-run"])

    def test_prices_need_a_currency(self):
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                self.run_cli([self.b1, "--gpu-hour-price", "1", "--dry-run"])


@unittest.skipUnless(shutil.which("node") and os.path.exists(os.path.join(HERE, "mock_server.mjs")),
                     "node and the SCETES API (mock_server.mjs, not in the public collector repository) are needed")
class AgainstTheRealApi(unittest.TestCase):
    """Contract test: the payload built here must pass the real validator in api/check.js and produce a result."""

    def start(self, enabled="1"):
        proc = subprocess.Popen(["node", "--no-warnings", os.path.join(HERE, "mock_server.mjs")], stdout=subprocess.PIPE, text=True,
                                env={**os.environ, "CHECK_ENABLED": enabled})
        self.addCleanup(lambda: (proc.kill(), proc.wait(), proc.stdout.close()))
        line = proc.stdout.readline().strip()
        self.assertTrue(line.startswith("READY"), line)
        return f"http://127.0.0.1:{line.split()[1]}/api/check"

    def test_upload_read_back_and_delete(self):
        d = tempfile.mkdtemp(); self.addCleanup(shutil.rmtree, d)
        b1 = make_archive(os.path.join(d, "b1.zip"))
        b8 = make_archive(os.path.join(d, "b8.zip"), batch=8, tps=300.0, j=0.8, p50=3.0, p95=3.12)
        endpoint = self.start()
        out = io.StringIO()
        with redirect_stdout(out):
            code = up.main([b1, b8, "--baseline", b1, "--currency", "EUR", "--gpu-hour-price", "1", "--kwh-price", "0.2",
                            "--monthly-output-tokens", "100000000", "--consent", "--endpoint", endpoint])
        text = out.getvalue()
        self.assertEqual(code, 0, text)
        self.assertIn("462.96 EUR", text)  # same figure as the API unit test: (5.5556 - 0.9259) EUR per 1M tokens * 100 M
        page = [l.strip() for l in text.splitlines() if "/en/check/#" in l][0]
        self.assertRegex(page, r"^http://127\.0\.0\.1:\d+/en/check/#[A-Za-z0-9_-]{43}$")
        link = [l.strip() for l in text.splitlines() if "as data: " in l][0].split("as data: ")[1].rstrip(")")
        self.assertEqual(up.token_of(page), up.token_of(link))
        status, body = up.http("GET", link)
        self.assertEqual(status, 200); self.assertEqual(len(body["input"]["runs"]), 2)
        self.assertNotIn("secret-box", json.dumps(body))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(up.main(["--delete", page, "--endpoint", endpoint]), 0)
        self.assertEqual(up.http("GET", link)[0], 404)

    def test_a_throttled_check_still_shows_measured_values_throttling_memory_and_next_step(self):
        d = tempfile.mkdtemp(); self.addCleanup(shutil.rmtree, d)
        detail = {"reasons": 5, "used": 6144}
        b1 = make_archive(os.path.join(d, "b1.zip"), clocks=(1590, 700, 700), detail=detail)
        b8 = make_archive(os.path.join(d, "b8.zip"), batch=8, tps=300.0, j=0.8, p50=3.0, p95=3.12, clocks=(1590, 700, 700), detail=detail)
        out = io.StringIO()
        with redirect_stdout(out):
            code = up.main([b1, b8, "--baseline", b1, "--consent", "--endpoint", self.start()])
        text = out.getvalue()
        self.assertEqual(code, 0, text)
        self.assertIn("INDICATIVE - GPU THROTTLED", text)
        self.assertIn("Best measured alternative", text)
        self.assertIn("Measured as it was: 50 to 300 tokens/s", text)
        self.assertIn("Throttling: up to 66.7 %", text)
        self.assertIn("power_cap", text)
        self.assertIn("9 GB free", text)
        self.assertIn("Diagnostic", text)

    def test_a_disabled_service_gives_a_clear_message(self):
        d = tempfile.mkdtemp(); self.addCleanup(shutil.rmtree, d)
        b1 = make_archive(os.path.join(d, "b1.zip"))
        out = io.StringIO()
        with redirect_stdout(out):
            code = up.main([b1, "--consent", "--endpoint", self.start("0")])
        self.assertEqual(code, 1); self.assertIn("not enabled", out.getvalue())


if __name__ == "__main__":
    unittest.main()
