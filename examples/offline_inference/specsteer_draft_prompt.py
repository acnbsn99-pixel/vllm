# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline SpecSteer example with per-request draft prompts (greedy only).

SpecSteer deploy checklist (rapid rollout):
- `speculative_config` required fields: `method="specsteer"`, `model`,
  `num_speculative_tokens`.
- Greedy-only path: use `SamplingParams(temperature=0, ...)`.
- `base_model`: omit/`None` to use the target `--model`; set a distinct model
  only when draft scoring should be anchored to another base model.
- `draft_prompt`: per-request steering text; if omitted, defaults to that
  request's `prompt`.
- Current support scope: offline `LLM.generate` examples only (not server API
  path yet).

Usage:
  python examples/offline_inference/specsteer_draft_prompt.py \
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
        },
    )

    requests = [
        {
            "prompt": "Describe speculative decoding in one paragraph.",
            "draft_prompt": (
                "Give a concise, high-level, approximate draft for "
                "speculative decoding."
            ),
        },
        {
            "prompt": "List three practical latency tips for LLM serving.",
            "draft_prompt": "Give short bullet points focused on latency.",
        },
    ]

    # SpecSteer currently supports greedy generation only.
    outputs = llm.generate(
        requests,
        sampling_params=SamplingParams(temperature=0, max_tokens=args.max_tokens),
    )

    for i, output in enumerate(outputs):
        print(f"=== Request {i} ===")
        print(f"Prompt: {output.prompt!r}")
        print(f"Text: {output.outputs[0].text!r}")


if __name__ == "__main__":
    main()
