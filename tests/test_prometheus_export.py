"""Prometheus textfile export (PR D): format, metric set, labels, privacy. No GPU needed.

The format check is a strict reading of the Prometheus text exposition format 0.0.4 (what the node_exporter textfile
collector reads). When prometheus_client is installed, its parser checks the output as well."""
import json, os, re, shutil, sys, tempfile, unittest, zipfile, hashlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import sce_infer as inf  # noqa: E402
from test_sce_upload import make_archive  # noqa: E402

NAME = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*$")
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\\n]|\\[\\"n])*)"')
SAMPLE = re.compile(r'([a-zA-Z_:][a-zA-Z0-9_:]*)(\{(.*)\})? (\S+)( -?\d+)?$')
TYPES = {"counter", "gauge", "histogram", "summary", "untyped"}
LABELS = ["model", "batch_size", "precision", "gpu_name", "collector_version"]


def parse_strict(text):
    """Returns {family: {"type", "help", "samples": [(name, labels, value)]}}; raises AssertionError on any format error."""
    assert text.endswith("\n"), "the file must end with a newline"
    fams, seen = {}, set()
    for n, line in enumerate(text.splitlines(), 1):
        assert line == line.strip() and line, f"line {n}: empty line or stray whitespace"
        if line.startswith("# HELP "):
            name, _, help_ = line[7:].partition(" ")
            assert NAME.match(name) and help_, f"line {n}: bad HELP"
            assert "help" not in fams.setdefault(name, {"samples": []}), f"line {n}: second HELP for {name}"
            fams[name]["help"] = help_
        elif line.startswith("# TYPE "):
            name, _, kind = line[7:].partition(" ")
            assert NAME.match(name) and kind in TYPES, f"line {n}: bad TYPE"
            fam = fams.setdefault(name, {"samples": []})
            assert "type" not in fam and not fam["samples"], f"line {n}: TYPE must come once and before the samples of {name}"
            fam["type"] = kind
        else:
            assert not line.startswith("#"), f"line {n}: unknown comment"
            m = SAMPLE.match(line)
            assert m, f"line {n}: not a sample line: {line}"
            name, _, body, value, _ts = m.groups()
            labels = {}
            if body:
                pos = 0
                while pos < len(body):
                    lm = LABEL.match(body, pos)
                    assert lm, f"line {n}: bad label syntax at {body[pos:]!r}"
                    assert lm.group(1) not in labels, f"line {n}: duplicate label {lm.group(1)}"
                    labels[lm.group(1)] = lm.group(2)
                    pos = lm.end()
                    if pos < len(body):
                        assert body[pos] == ",", f"line {n}: labels must be separated by commas"
                        pos += 1
            float(value)  # Prometheus also accepts NaN, +Inf, -Inf, which float() reads too
            family = next((f for f in (name, re.sub(r"_(sum|count|bucket)$", "", name)) if f in fams), None)
            assert family and "type" in fams[family], f"line {n}: sample {name} has no TYPE"
            if "quantile" in labels:
                assert fams[family]["type"] == "summary" and name == family, f"line {n}: quantile label outside a summary"
            if name != family:
                assert fams[family]["type"] in ("summary", "histogram"), f"line {n}: {name} suffix on a {fams[family]['type']}"
            key = (name, tuple(sorted(labels.items())))
            assert key not in seen, f"line {n}: duplicate series"
            seen.add(key)
            fams[family]["samples"].append((name, labels, float(value)))
    return fams


class PrometheusExport(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        self.zip = make_archive(os.path.join(self.dir, "run.zip"), detail={"reasons": 4, "used": 4000})

    def test_format_and_metric_set(self):
        fams = parse_strict(inf.prometheus_text(self.zip))
        for name in ("scetes_tokens_per_second", "scetes_energy_joules_per_output_token", "scetes_latency_seconds",
                     "scetes_throttle_loss_ratio", "scetes_gpu_memory_headroom_bytes"):
            self.assertIn(name, fams)
            self.assertEqual(fams[name]["type"], "summary" if name == "scetes_latency_seconds" else "gauge")
        self.assertEqual(fams["scetes_tokens_per_second"]["samples"][0][2], 50.0)
        self.assertEqual(fams["scetes_energy_joules_per_output_token"]["samples"][0][2], 4.0)
        quantiles = {s[1]["quantile"]: s[2] for s in fams["scetes_latency_seconds"]["samples"] if "quantile" in s[1]}
        self.assertEqual(quantiles, {"0.5": 2.5, "0.95": 2.6})  # p99 is not reported with 20 repeats
        self.assertEqual(fams["scetes_gpu_memory_headroom_bytes"]["samples"][0][2], (15360 - 4000) * 1024 * 1024)

    def test_labels_exact_and_private(self):
        text = inf.prometheus_text(self.zip)
        for _, fam in parse_strict(text).items():
            for name, labels, _ in fam["samples"]:
                self.assertEqual([k for k in labels if k != "quantile"], LABELS, name)
        for secret in ("secret-box", "GPU-secret-uuid", "gpu_uuid", "hostname"):
            self.assertNotIn(secret, text)

    def test_idle_ratio_and_summary_counts_with_full_rows(self):
        path = os.path.join(self.dir, "full.zip")
        make_archive(path, detail={"reasons": 0, "used": 2000})
        with zipfile.ZipFile(path) as z:
            files = {n: z.read(n).decode() for n in z.namelist() if n != "checksum.txt"}
        files["benchmark.csv"] = "repeat,avg_util_gpu_pct,avg_power_w,request_latency_s\n1,90,60,2.4\n2,92,62,2.6\n"
        files["telemetry.csv"] = files["telemetry.csv"].replace("phase,clock_sm_mhz", "phase,clock_sm_mhz").splitlines()
        tele = [files["telemetry.csv"][0] + ",util_gpu_pct"] + [r + ",80" for r in files["telemetry.csv"][1:]]
        files["telemetry.csv"] = "\n".join(tele) + "\n"
        files["checksum.txt"] = "".join(f"{hashlib.sha256(v.encode()).hexdigest()}  {k}\n" for k, v in files.items())
        with zipfile.ZipFile(path, "w") as z:
            for k, v in files.items():
                z.writestr(k, v)
        fams = parse_strict(inf.prometheus_text(path))
        self.assertAlmostEqual(fams["scetes_gpu_idle_ratio"]["samples"][0][2], 0.2)
        by_name = {s[0]: s[2] for s in fams["scetes_latency_seconds"]["samples"]}
        self.assertAlmostEqual(by_name["scetes_latency_seconds_sum"], 5.0)
        self.assertEqual(by_name["scetes_latency_seconds_count"], 2)

    def test_label_escaping(self):
        self.assertEqual(inf._prom_label('a"b\\c\nd'), 'a\\"b\\\\c\\nd')
        line = 'x{model="' + inf._prom_label('a"b\\c\nd') + '"} 1'
        self.assertTrue(LABEL.search(line))

    def test_atomic_write(self):
        target = os.path.join(self.dir, "scetes.prom")
        written = inf.write_prometheus(self.zip, target)
        self.assertEqual(str(written), os.path.realpath(target))
        self.assertFalse(os.path.exists(target + ".tmp"))
        parse_strict(open(target, encoding="utf-8").read())

    def test_prometheus_client_parser(self):
        try:
            from prometheus_client.parser import text_string_to_metric_families
        except ImportError:
            self.skipTest("prometheus_client is not installed (pip install prometheus_client to run this check)")
        names = {f.name for f in text_string_to_metric_families(inf.prometheus_text(self.zip))}
        self.assertIn("scetes_latency_seconds", names)

    def test_flags(self):
        a = inf.build_parser().parse_args(["--model", "m", "--prompt", "p", "--export", "prometheus", "--export-file", "/tmp/x.prom"])
        self.assertEqual((a.export, a.export_file), ("prometheus", "/tmp/x.prom"))


if __name__ == "__main__":
    unittest.main()
