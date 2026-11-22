import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "strix-metrics.py"
spec = importlib.util.spec_from_file_location("strix_metrics", SCRIPT)
sm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sm)

L = 'engine="0",model_name="qwen3.8-flash-next"'


def snap(requests_stop, requests_length, prompt, gen, hits, queries, e2e_le1, e2e_inf, e2e_sum, accepted, drafts, ndrafts):
    return "\n".join([
        "# HELP vllm:request_success_total x",
        f'vllm:request_success_total{{{L},finished_reason="stop"}} {requests_stop}',
        f'vllm:request_success_total{{{L},finished_reason="length"}} {requests_length}',
        f"vllm:prompt_tokens_total{{{L}}} {prompt}",
        f"vllm:generation_tokens_total{{{L}}} {gen}",
        f"vllm:prefix_cache_hits_total{{{L}}} {hits}",
        f"vllm:prefix_cache_queries_total{{{L}}} {queries}",
        f'vllm:e2e_request_latency_seconds_bucket{{{L},le="1.0"}} {e2e_le1}',
        f'vllm:e2e_request_latency_seconds_bucket{{{L},le="10.0"}} {e2e_inf}',
        f'vllm:e2e_request_latency_seconds_bucket{{{L},le="+Inf"}} {e2e_inf}',
        f"vllm:e2e_request_latency_seconds_count{{{L}}} {e2e_inf}",
        f"vllm:e2e_request_latency_seconds_sum{{{L}}} {e2e_sum}",
        f"vllm:spec_decode_num_accepted_tokens_total{{{L}}} {accepted}",
        f"vllm:spec_decode_num_draft_tokens_total{{{L}}} {drafts}",
        f"vllm:spec_decode_num_drafts_total{{{L}}} {ndrafts}",
        f"vllm:kv_cache_usage_perc{{{L}}} 0.25",
    ]) + "\n"


BEFORE = snap(10, 1, 1000, 100, 500, 1000, 5, 11, 30.0, 40, 100, 25)
AFTER = snap(30, 3, 101000, 2100, 90500, 101000, 5, 33, 250.0, 140, 300, 75)


class Parse(unittest.TestCase):
    def test_drops_engine_labels_and_comments(self):
        s = sm.parse(BEFORE)
        self.assertEqual(s[("vllm:prompt_tokens_total", ())], 1000)
        self.assertEqual(s[("vllm:request_success_total", (("finished_reason", "stop"),))], 10)

    def test_delta_and_totals(self):
        d = sm.delta(sm.parse(BEFORE), sm.parse(AFTER))
        self.assertEqual(sm.total(d, "vllm:prompt_tokens_total"), 100000)
        self.assertEqual(sm.by_label(d, "vllm:request_success_total", "finished_reason"), {"stop": 20, "length": 2})


class Quantile(unittest.TestCase):
    def test_interpolates_inside_a_bucket(self):
        value, bound = sm.quantile([(1.0, 0), (10.0, 10), (float("inf"), 10)], 0.5)
        self.assertEqual(bound, "")
        self.assertAlmostEqual(value, 5.5)

    def test_first_bucket_reports_an_upper_bound(self):
        self.assertEqual(sm.quantile([(0.3, 10), (1.0, 10), (float("inf"), 10)], 0.5), (0.3, "<="))

    def test_overflow_reports_a_lower_bound(self):
        self.assertEqual(sm.quantile([(1.0, 0), (float("inf"), 4)], 0.95), (1.0, ">="))

    def test_empty(self):
        self.assertEqual(sm.quantile([], 0.5), (None, ""))


class Report(unittest.TestCase):
    def test_report_lines(self):
        out = sm.report(sm.parse(BEFORE), sm.parse(AFTER), wall=200)
        self.assertIn("finished 22: length=2, stop=20", out)
        self.assertIn("2 hit max_tokens", out)
        self.assertIn("prefix cache: 90,000 of 100,000 prompt tokens hit (90.0%)", out)
        self.assertIn("MTP: 50.0% of draft tokens accepted, 3.00 tokens per verify step", out)
        self.assertIn("end to end", out)
        self.assertIn("KV cache in use at the end: 25.0%", out)
        self.assertNotIn("WARNING", out)

    def test_restart_between_snapshots_is_flagged(self):
        self.assertIn("restarted", sm.report(sm.parse(AFTER), sm.parse(BEFORE), wall=0))

    def test_cli_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            b, a = Path(tmp) / "b.prom", Path(tmp) / "a.prom"
            b.write_text(BEFORE)
            a.write_text(AFTER)
            r = subprocess.run([sys.executable, str(SCRIPT), "diff", str(b), str(a), "--wall", "100"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("== wall clock", r.stdout)


if __name__ == "__main__":
    unittest.main()
