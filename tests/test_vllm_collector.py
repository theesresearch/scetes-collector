"""The vLLM collector is tested with stand-ins for vLLM, PyTorch and the GPU (no GPU here). What this proves: the archive
format, integrity checks and payload are compatible with sce_upload.py, and token counts are enforced. It does not prove that
real vLLM measurements are correct; that needs a run on real hardware."""
import io, os, shutil, sys, tempfile, time, types, unittest
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sce_upload as up  # noqa: E402
import sce_infer_vllm as sv  # noqa: E402


def fake_vllm(tokens_out=None):
    class SamplingParams:
        def __init__(self, **kw): self.kw = kw
    class LLM:
        def __init__(self, **kw): self.kw = kw
        def generate(self, prompts, params, use_tqdm=False):
            n = params.kw["max_tokens"] if tokens_out is None else tokens_out
            out = types.SimpleNamespace(token_ids=list(range(n)))
            return [types.SimpleNamespace(outputs=[out], prompt_token_ids=[1, 2, 3, 4, 5]) for _ in prompts]
    return types.SimpleNamespace(LLM=LLM, SamplingParams=SamplingParams, __version__="0.0-test")


class FakeGPU:
    def __init__(self, *a): self.index, self.uuid, self.name, self.mapping_method = 0, "GPU-secret-uuid", "Tesla T4", "torch_cuda_uuid"
    def measure(self, fn, sync, period):
        t0 = time.perf_counter(); fn(); sync(); dt = max(time.perf_counter() - t0, 1e-3)
        s = {"t": t0, "power_w": 60.0, "util_gpu_pct": 90, "util_mem_pct": 50, "clock_sm_mhz": 1590, "temp_c": 60,
             "max_clock_sm_mhz": 1590, "throttle_reasons": 0, "mem_used_mib": 4000, "mem_total_mib": 15360}
        e = 60.0 * dt
        return {"elapsed_s": dt, "energy_window_s": dt, "energy_j": e, "energy_counter_j": e, "energy_integral_j": e,
                "energy_counter_integral_rel_diff": 0.0, "energy_check_passed": True, "energy_source": "nvml_counter",
                "samples": [s, {**s, "t": t0 + dt}], "avg_power_w": 60.0, "avg_util_gpu_pct": 90, "max_temp_c": 60}
    def close(self): pass


def fake_torch():
    props = types.SimpleNamespace(uuid="u", pci_bus_id="0", major=7, minor=5, total_memory=15 * 2**30)
    cuda = types.SimpleNamespace(is_available=lambda: True, get_device_properties=lambda i: props)
    return types.SimpleNamespace(cuda=cuda, __version__="0-test", version=types.SimpleNamespace(cuda="13.0"))


class VllmCollector(unittest.TestCase):
    def run_collector(self, *extra, tokens_out=None):
        d = tempfile.mkdtemp(); self.addCleanup(shutil.rmtree, d)
        out = os.path.join(d, "v.zip")
        env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
        with mock.patch.dict(sys.modules, {"torch": fake_torch()}), mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(sv.time, "sleep", lambda s: None), redirect_stdout(io.StringIO()):
            sv.main(["--model", "m", "--prompt", "secret prompt", "--batch-size", "4", "--output-tokens", "16", "--repeats", "20",
                     "--warmup", "1", "--output", out, *extra], vllm_module=fake_vllm(tokens_out), gpu_class=FakeGPU)
        return out

    def test_archive_is_accepted_by_the_upload_tool_and_keeps_privacy(self):
        out = self.run_collector()
        a = up.read_archive(out)  # verifies all checksums
        self.assertEqual(a["manifest"]["engine"], "vllm")
        self.assertEqual(len(a["benchmark"]), 20)
        payload = up.build_payload([a], out)
        run = payload["runs"][0]
        self.assertEqual((run["batch_size"], run["output_tokens"], run["quality"]["energy_check_ok"]), (4, 16, True))
        self.assertFalse(run["quality"]["throttling_flagged"])
        self.assertEqual(run["memory"]["total_gb"], 15.0)
        blob = open(out, "rb").read()
        self.assertNotIn(b"secret prompt", blob)
        self.assertIsNotNone(a["metrics"]["request_latency_p95_s"])

    def test_wrong_token_count_is_refused(self):
        with self.assertRaises(RuntimeError):
            self.run_collector(tokens_out=10)

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", io.StringIO()):
            sv.parse(["--model", "m", "--prompt", "p", "--gpu-memory-utilization", "1.5"])


if __name__ == "__main__":
    unittest.main()
