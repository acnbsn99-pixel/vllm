# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline SpecSteer example (greedy-only).

Usage:
  python examples/offline_inference/specsteer.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --draft-model meta-llama/Llama-3.2-1B-Instruct
"""

import argparse

from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        speculative_config={
            "method": "specsteer",
            "model": args.draft_model,
            "base_model": args.base_model,
            "num_speculative_tokens": 4,
            "gamma": 0.6,
            "eps": 1e-10,
            "fusion_method": "costeer",
            "specsteer_enable_bonus_token": False,
        },
    )

    outputs = llm.generate(
        [
            {
                "prompt": "Describe speculative decoding in one paragraph.",
                "draft_prompt": (
                    "Give a concise, high-level, approximate draft for "
                    "speculative decoding."
                ),
            }
        ],
        sampling_params=SamplingParams(temperature=0, max_tokens=args.max_tokens),
    )

    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
