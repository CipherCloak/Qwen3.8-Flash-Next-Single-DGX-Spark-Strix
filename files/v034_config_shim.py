#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Present a Mia-layout snapshot to vLLM 0.34 under the upstream Qwen4Exp names.

The Mia and Keys ablit checkpoints still declare the pre-rename model type
(qwen3_8_flash_next / Qwen3_8FlashNextForConditionalGeneration). vLLM 0.34 and
transformers 5.22 only know qwen4_exp, so the config fails to parse before any
--hf-overrides can apply. The weights themselves load unchanged: the tensor
names, the 128 NVFP4 PLE shards and the ModelOpt mixed-precision map are what
upstream reads from local-inference-lab/Qwen3.8-Flash-Next-NVFP4, the
checkpoint Mia was spliced from (its config moved to qwen4_exp in October).

YaRN goes into the same config, in the transformers convention 0.34 follows:
max_position_embeddings is the scaled length and the factor is not applied to
it again (vllm/config/model.py, "YaRN variants leave max_position_embeddings
already scaled"). The pinned image multiplied instead, which is why start.sh
passes only rope_parameters there; on 0.34 that left the derived
max_model_len at 262144 and a 524288 request was rejected.

Writes <out>/config.json with the renamed fields (and the YaRN rope, when
asked) and links every other snapshot entry to its path as the container sees
it, so nothing in the Hugging Face cache is touched and the shim costs a few KiB.

Usage: v034_config_shim.py <snapshot_dir> <out_dir> <container_snapshot_dir> [<yarn_factor> <original_max>]
Prints "shim" when it wrote a shim, "native" when the config already uses the
upstream names and no YaRN was asked for (start.sh then serves the snapshot
directly).
"""
import json
import math
import os
import sys

RENAMES = {
    "qwen3_8_flash_next": "qwen4_exp",
    "qwen3_8_flash_next_text": "qwen4_exp_text",
}


def convert(cfg: dict, yarn: tuple[float, int] | None = None) -> dict:
    cfg = dict(cfg)
    cfg["model_type"] = RENAMES.get(cfg.get("model_type"), cfg.get("model_type"))
    cfg["architectures"] = ["Qwen4ExpForConditionalGeneration"]
    text = dict(cfg.get("text_config") or {})
    text["model_type"] = RENAMES.get(text.get("model_type"), text.get("model_type"))
    # Upstream reads these two from text_config; the Mia config keeps the
    # rotary factor inside rope_parameters and implies the interval through
    # layer_types.
    rope = text.get("rope_parameters") or {}
    if "partial_rotary_factor" not in text and "partial_rotary_factor" in rope:
        text["partial_rotary_factor"] = rope["partial_rotary_factor"]
    if "full_attention_interval" not in text and text.get("layer_types"):
        full = [i for i, t in enumerate(text["layer_types"]) if t == "full_attention"]
        if len(full) > 1:
            text["full_attention_interval"] = full[1] - full[0]
    if yarn:
        # Same keys start.sh passes as --hf-overrides on the pinned image;
        # mrope_section / rope_theta / partial_rotary_factor are kept.
        text["rope_parameters"] = {**rope, "rope_type": "yarn", "factor": yarn[0],
                                   "original_max_position_embeddings": yarn[1]}
        text["max_position_embeddings"] = math.ceil(yarn[0] * yarn[1])
    cfg["text_config"] = text
    if isinstance(cfg.get("vision_config"), dict):
        vision = dict(cfg["vision_config"])
        vision["model_type"] = "qwen4_exp"
        cfg["vision_config"] = vision
    return cfg


def main() -> None:
    if len(sys.argv) not in (4, 6):
        sys.exit(__doc__)
    snapshot, out, container_snapshot = sys.argv[1:4]
    yarn = (float(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) == 6 else None
    cfg = json.load(open(os.path.join(snapshot, "config.json")))
    if cfg.get("model_type") not in RENAMES and yarn is None:
        print("native")
        return
    os.makedirs(out, exist_ok=True)
    for name in sorted(os.listdir(snapshot)):
        if name == "config.json":
            continue
        link = os.path.join(out, name)
        target = os.path.join(container_snapshot, name)
        if os.path.islink(link) and os.readlink(link) == target:
            continue
        if os.path.lexists(link):
            sys.exit(f"v034_config_shim: {link} exists and is not the expected link; refusing to replace it")
        os.symlink(target, link)
    tmp = os.path.join(out, ".config.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(convert(cfg, yarn), fh, indent=2)
        fh.write("\n")
    os.replace(tmp, os.path.join(out, "config.json"))
    print("shim")


if __name__ == "__main__":
    main()
