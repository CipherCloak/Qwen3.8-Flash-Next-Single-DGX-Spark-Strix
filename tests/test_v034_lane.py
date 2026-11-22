import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parent.parent
FILES = REPO / "files"
START = REPO / "start.sh"
WRAPPER = REPO / "start-v034.sh"
SHIM = FILES / "v034_config_shim.py"
LANE_LOADER = REPO / "scripts" / "launch-lane.sh"
SUPERVISE = REPO / "scripts" / "supervise.sh"


def load_shim():
    spec = importlib.util.spec_from_file_location("v034_config_shim", SHIM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MIA_CONFIG = {
    "model_type": "qwen3_8_flash_next",
    "architectures": ["Qwen3_8FlashNextForConditionalGeneration"],
    "text_config": {
        "model_type": "qwen3_8_flash_next_text",
        "max_position_embeddings": 262144,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"],
        "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10],
                            "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
        "ple_embedding_dtype": "nvfp4",
    },
    "vision_config": {"model_type": "qwen3_5_vision", "depth": 27},
    "quantization_config": {"quant_algo": "MIXED_PRECISION", "quantized_layers": {"a": {"quant_algo": "MXFP8"}}},
}


class ShimConvert(unittest.TestCase):
    def setUp(self):
        self.shim = load_shim()

    def test_renames_to_upstream_names(self):
        out = self.shim.convert(MIA_CONFIG)
        self.assertEqual(out["model_type"], "qwen4_exp")
        self.assertEqual(out["architectures"], ["Qwen4ExpForConditionalGeneration"])
        self.assertEqual(out["text_config"]["model_type"], "qwen4_exp_text")
        self.assertEqual(out["vision_config"]["model_type"], "qwen4_exp")

    def test_adds_fields_upstream_reads_from_text_config(self):
        text = self.shim.convert(MIA_CONFIG)["text_config"]
        self.assertEqual(text["partial_rotary_factor"], 0.25)
        self.assertEqual(text["full_attention_interval"], 4)

    def test_weights_side_untouched(self):
        out = self.shim.convert(MIA_CONFIG)
        self.assertEqual(out["quantization_config"], MIA_CONFIG["quantization_config"])
        self.assertEqual(out["text_config"]["ple_embedding_dtype"], "nvfp4")
        self.assertEqual(MIA_CONFIG["model_type"], "qwen3_8_flash_next", "input must not be mutated")

    def test_yarn_scales_max_position_embeddings(self):
        text = self.shim.convert(MIA_CONFIG, (2.0, 262144))["text_config"]
        self.assertEqual(text["max_position_embeddings"], 524288)
        rope = text["rope_parameters"]
        self.assertEqual((rope["rope_type"], rope["factor"], rope["original_max_position_embeddings"]),
                         ("yarn", 2.0, 262144))
        self.assertEqual(rope["mrope_section"], [11, 11, 10])

    def test_no_yarn_keeps_native_length(self):
        text = self.shim.convert(MIA_CONFIG)["text_config"]
        self.assertEqual(text["max_position_embeddings"], 262144)
        self.assertEqual(text["rope_parameters"]["rope_type"], "default")


class ShimFiles(unittest.TestCase):
    def run_shim(self, snap, out, *extra):
        return subprocess.run([sys.executable, str(SHIM), str(snap), str(out), "/ctr/snap", *extra],
                              capture_output=True, text=True)

    def make_snapshot(self, root, config):
        snap = Path(root) / "snap"
        snap.mkdir()
        (snap / "config.json").write_text(json.dumps(config))
        (snap / "model-00001-of-00002.safetensors").write_text("x")
        (snap / "tokenizer.json").write_text("{}")
        return snap

    def test_links_point_inside_the_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap, out = self.make_snapshot(tmp, MIA_CONFIG), Path(tmp) / "out"
            r = self.run_shim(snap, out)
            self.assertEqual((r.returncode, r.stdout.strip()), (0, "shim"), r.stderr)
            self.assertEqual(os.readlink(out / "tokenizer.json"), "/ctr/snap/tokenizer.json")
            self.assertFalse((out / "config.json").is_symlink())
            self.assertEqual(json.loads((out / "config.json").read_text())["model_type"], "qwen4_exp")

    def test_rerun_is_idempotent_and_follows_yarn(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap, out = self.make_snapshot(tmp, MIA_CONFIG), Path(tmp) / "out"
            self.assertEqual(self.run_shim(snap, out).returncode, 0)
            r = self.run_shim(snap, out, "2.0", "262144")
            self.assertEqual(r.returncode, 0, r.stderr)
            cfg = json.loads((out / "config.json").read_text())
            self.assertEqual(cfg["text_config"]["max_position_embeddings"], 524288)
            r = self.run_shim(snap, out)
            self.assertEqual(json.loads((out / "config.json").read_text())["text_config"]["max_position_embeddings"], 262144)

    def test_refuses_to_replace_a_foreign_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap, out = self.make_snapshot(tmp, MIA_CONFIG), Path(tmp) / "out"
            out.mkdir()
            (out / "tokenizer.json").write_text("mine")
            r = self.run_shim(snap, out)
            self.assertNotEqual(r.returncode, 0)
            self.assertEqual((out / "tokenizer.json").read_text(), "mine")

    def test_native_config_without_yarn_needs_no_shim(self):
        native = {**MIA_CONFIG, "model_type": "qwen4_exp"}
        with tempfile.TemporaryDirectory() as tmp:
            snap, out = self.make_snapshot(tmp, native), Path(tmp) / "out"
            r = self.run_shim(snap, out)
            self.assertEqual((r.returncode, r.stdout.strip()), (0, "native"))
            self.assertFalse(out.exists())


class DraftVocabGenerator(unittest.TestCase):
    ANCHORS = ("import torch\nfrom torch import nn\n"
               "from vllm.distributed import get_pp_group\n"
               "\nclass Qwen4ExpMultiTokenPredictor(nn.Module):\n    pass\n"
               "\nclass M:\n    def compute_logits(self, hidden_states):\n"
               "        return self.logits_processor(self.lm_head, hidden_states)\n"
               "\n    def load_weights(self, weights):\n"
               "        return loader.load_weights(remap_weight_names(), mapper=mapper)\n")

    def test_explicit_paths_for_the_034_mtp(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig, out = Path(tmp) / "mtp_v034_patched.py.orig", Path(tmp) / "mtp_v034_patched.py"
            orig.write_text(self.ANCHORS)
            r = subprocess.run([sys.executable, str(FILES / "patch_mtp_draft_vocab_v030.py"), str(orig), str(out)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("patched mtp_v034_patched.py", r.stdout)
            src = out.read_text()
            self.assertIn("_attach_draft_vocab(self)", src)
            self.assertIn("def get_top_tokens", src)

    def test_moved_anchor_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig, out = Path(tmp) / "a.orig", Path(tmp) / "a.py"
            orig.write_text(self.ANCHORS.replace("from vllm.distributed import get_pp_group\n", ""))
            r = subprocess.run([sys.executable, str(FILES / "patch_mtp_draft_vocab_v030.py"), str(orig), str(out)],
                               capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertFalse(out.exists())


class StartScript(unittest.TestCase):
    def test_scripts_parse(self):
        r = subprocess.run(["bash", "-n", str(START), str(WRAPPER)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_wrapper_only_sets_the_lane(self):
        src = WRAPPER.read_text()
        self.assertIn("export V034=true", src)
        self.assertIn('exec "$SCRIPT_DIR/start.sh" "$@"', src)
        # Knobs belong in .env; an export here would beat .env (snapshot rule).
        self.assertNotRegex(src, r"export (V034_KV_GIB|IMAGE|KV_CACHE_DTYPE)=")

    def test_new_knobs_beat_dotenv(self):
        snapshot = re.search(r"_ENV_SNAPSHOT_VARS=\((.*?)\)", START.read_text(), re.S).group(1).split()
        for name in ("V034", "V034_KV_GIB", "V034_IMAGE", "TOOL_CALL_PARSER", "TOOL_STRICT_LEVEL"):
            self.assertIn(name, snapshot)

    def test_lane_rejects_unsupported_knobs(self):
        src = START.read_text()
        for needle in ("V034: V030 and V034 are separate lanes", "V034: MTP_K_SCHEDULE",
                       "V034: the determinism knobs", "V034: set V034_KV_GIB instead of KV_CACHE_MEMORY",
                       "TOOL_STRICT_LEVEL needs the vLLM 0.34 lane"):
            self.assertIn(needle, src)

    def test_lane_uses_native_features_not_day0_overlays(self):
        src = START.read_text()
        self.assertIn('[[ "$V034" == "true" ]] && VLLM_ARGS+=("--engram-config" "\'{\\"disk_offload\\":true}\'" '
                      '"--kv-cache-memory-bytes" "${V034_KV_GIB}G" "--enable-prefix-caching")', src)
        self.assertIn('if [[ "$V030" != "true" && "$V034" != "true" ]]; then\nPLE_CACHE_ID=', src)
        self.assertIn('if [[ -n "$YARN_FACTOR" && "$V034" != "true" ]]; then', src)

    def test_api_key_never_on_the_command_line(self):
        template = re.search(r'<<LAUNCH_EOF\n(.*?)^LAUNCH_EOF\n', START.read_text(), re.S | re.M).group(1)
        self.assertIn("${API_KEY:+-e VLLM_API_KEY}", template)
        self.assertNotIn("--api-key", template)
        self.assertIn('export VLLM_API_KEY="$API_KEY"', START.read_text())

    def test_lane_command_has_no_day0_mounts(self):
        src = START.read_text()
        block = re.search(r'^if \[\[ "\$V030" == "true" \]\]; then\n    PLE_ENV=.*?^fi\n', src, re.S | re.M).group(0)
        script = ('V030=false\nV034=true\nV034_MOUNTS="@V034_MOUNTS@"\nPLE_CACHE_CTR=x\nPATCHED_PLE=x\n'
                  + block + 'printf "%s|%s|%s|%s" "$PLE_ENV" "$OVERLAY_MOUNTS" "$BLOCK_DROP_MOUNTS" "$OFFLOAD_MOUNTS"\n')
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        ple_env, overlays, block_drop, offload = r.stdout.split("|")
        self.assertIn("flashinfer_autotune_v034", ple_env)
        self.assertNotIn("VLLM_PLE_", ple_env)
        self.assertEqual((overlays, block_drop, offload), ("@V034_MOUNTS@", "", ""))


class LaunchLane(unittest.TestCase):
    def decide(self, record):
        with tempfile.TemporaryDirectory() as repo:
            (Path(repo) / "logs").mkdir()
            (Path(repo) / "logs" / "launch-lane").write_text(record)
            script = (f'REPO_DIR="{repo}"\nsource "{LANE_LOADER}"\nload_launch_lane\n'
                      'echo "${START_SCRIPT#$REPO_DIR/} ${LANE_MEMWATCH_MIN_GIB:-none}"\n')
            r = subprocess.run(["bash", "-uc", script], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            return r.stdout.split()

    def test_v034_record_relaunches_its_wrapper(self):
        self.assertEqual(self.decide("V030=false\nMEMWATCH_MIN_GIB=6\nMEMWATCH_MIN_FREE_GIB=2\nV034=true\n"),
                         ["start-v034.sh", "6"])

    def test_day0_record_still_means_start_sh(self):
        self.assertEqual(self.decide("V030=false\nMEMWATCH_MIN_GIB=6\nMEMWATCH_MIN_FREE_GIB=2\nV034=false\n"),
                         ["start.sh", "none"])

    def test_start_appends_the_v034_line(self):
        src = START.read_text()
        first = src.index('> "$SCRIPT_DIR/logs/launch-lane"')
        second = src.index("printf 'V034=%s\\n' \"$V034\" >> \"$SCRIPT_DIR/logs/launch-lane\"")
        self.assertLess(first, second)


if __name__ == "__main__":
    unittest.main()
