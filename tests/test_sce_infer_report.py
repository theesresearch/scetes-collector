"""End of a collector run: summary and the optional upload (PR C). No GPU, no network: the POST is a fake."""
import io, json, os, shutil, sys, tempfile, unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import sce_infer as inf  # noqa: E402
import sce_upload as up  # noqa: E402
from test_sce_upload import make_archive  # noqa: E402

TOKEN = "A" * 43


class AfterRun(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        self.zip = make_archive(os.path.join(self.dir, "run.zip"), detail={"reasons": 4, "used": 4000})
        self.calls = []
        # Any real network access fails the test.
        patcher = mock.patch.object(up, "http", side_effect=AssertionError("network used"))
        patcher.start(); self.addCleanup(patcher.stop)

    def fake_post(self, method, url, payload):
        self.calls.append((method, url, payload))
        return 201, {"token": TOKEN, "expires_at": "2027-10-12T00:00:00Z", "result_url": "/api/check/" + TOKEN, "result": {}}

    def run_(self, answer="", **kw):
        out = io.StringIO()
        code = inf.after_run(self.zip, stdin=io.StringIO(answer), out=out, post=self.fake_post, **kw)
        return code, out.getvalue()

    def test_summary(self):
        code, text = self.run_(upload=False)
        self.assertEqual(code, 0)
        for part in ("Tokens per second (median):     50.00", "p95 batch latency:              2.600 s", "GPU energy per output token:    4.0000 J",
                     "Throttling:", "Local report:", os.path.abspath(self.zip)):
            self.assertIn(part, text)

    def test_tty_yes_sends_and_prints_report_link(self):
        code, text = self.run_("y\n", interactive=True)
        self.assertEqual(code, 0)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][1], up.DEFAULT_ENDPOINT)
        self.assertIn("[y/N]", text)
        self.assertIn(inf.PRIVACY_URL, text)
        self.assertIn("runs[].metrics.tokens_per_s", text)
        self.assertIn(f"https://scetes.com/en/check/#{TOKEN}", text)

    def test_tty_default_is_no(self):
        for answer in ("\n", "n\n", "", "maybe\n"):
            code, text = self.run_(answer, interactive=True)
            self.assertEqual(code, 0)
            self.assertIn("Nothing was sent.", text)
        self.assertEqual(self.calls, [])

    def test_no_tty_never_sends(self):
        for upload in (None, True):
            code, text = self.run_("y\n", interactive=False, upload=upload)
            self.assertIn("No terminal: nothing was sent.", text)
            self.assertNotIn("[y/N]", text)
        self.assertEqual(self.calls, [])

    def test_no_tty_is_detected(self):
        with mock.patch.object(sys.stdin, "isatty", return_value=False):
            code, text = self.run_("y\n")
        self.assertIn("No terminal", text)
        self.assertEqual(self.calls, [])

    def test_upload_flag_skips_question(self):
        code, text = self.run_("", interactive=True, upload=True)
        self.assertEqual(len(self.calls), 1)
        self.assertNotIn("[y/N]", text)
        self.assertIn("runs[].metrics.tokens_per_s", text)  # the field list is still shown

    def test_no_upload_flag(self):
        code, text = self.run_("y\n", interactive=True, upload=False)
        self.assertIn("skipped (--no-upload)", text)
        self.assertNotIn("[y/N]", text)
        self.assertEqual(self.calls, [])

    def test_dry_run_prints_exact_json_and_sends_nothing(self):
        code, text = self.run_("y\n", interactive=True, dry_run=True)
        self.assertEqual(self.calls, [])
        self.assertIn("Dry run: nothing was sent.", text)
        body = text.split("This is exactly what would be sent:\n", 1)[1].split("\n\nDry run", 1)[0]
        self.assertEqual(json.loads(body), up.build_payload([up.read_archive(self.zip)]))

    def test_payload_never_contains_identifiers(self):
        self.run_("y\n", interactive=True)
        sent = json.dumps(self.calls[0][2])
        for secret in ("secret-box", "GPU-secret-uuid", "prompt"):
            self.assertNotIn(secret, sent)

    def test_rejected_upload_keeps_local_report(self):
        out = io.StringIO()
        code = inf.after_run(self.zip, stdin=io.StringIO("y\n"), out=out, interactive=True, post=lambda *a: (400, {"error": "bad"}))
        self.assertEqual(code, 1)
        self.assertIn("Your local report is unchanged.", out.getvalue())
        self.assertTrue(os.path.exists(self.zip))

    def test_flags(self):
        p = inf.build_parser()
        base = ["--model", "m", "--prompt", "p"]
        self.assertIsNone(p.parse_args(base).upload)
        self.assertTrue(p.parse_args(base + ["--upload"]).upload)
        self.assertFalse(p.parse_args(base + ["--no-upload"]).upload)
        self.assertTrue(p.parse_args(base + ["--dry-run"]).dry_run)
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", io.StringIO()):
            p.parse_args(base + ["--upload", "--no-upload"])


if __name__ == "__main__":
    unittest.main()
