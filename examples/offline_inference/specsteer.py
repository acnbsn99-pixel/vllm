# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline speculative steering example.

This example demonstrates the shape of a SpecSteer request in the offline
``LLM.generate`` API.

Required ``speculative_config`` keys for SpecSteer:
* ``method``: must be ``"specsteer"``.
* ``model``: draft/assistant model used by the speculative proposer.
* ``num_speculative_tokens``: number of draft tokens to propose each step.

Per-request ``draft_prompt`` behavior:
* ``draft_prompt`` is passed inside each request object alongside ``prompt``.
* ``draft_prompt`` is currently only demonstrated for the offline Python API.
* OpenAI-compatible server endpoints do not currently accept a request-level
  ``draft_prompt`` field.
"""

from vllm import LLM, SamplingParams

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DRAFT_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


def main() -> None:
    llm = LLM(
        model=MODEL_NAME,
        speculative_config={
            "method": "specsteer",
            "model": DRAFT_MODEL_NAME,
            "num_speculative_tokens": 4,
        },
    )

    requests = [
        {
            "prompt": "Explain speculative decoding in one short paragraph.",
            "draft_prompt": "Write a concise explanation focused on latency.",
        },
        {
            "prompt": "List two practical benefits of request-level steering.",
            "draft_prompt": "Prefer bullet points and mention controllability.",
        },
    ]

    # Greedy decoding only.
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        max_tokens=64,
    )

    outputs = llm.generate(requests, sampling_params=sampling_params)

    for i, output in enumerate(outputs):
        print(f"\n=== Request {i} ===")
        print(f"Prompt: {requests[i]['prompt']}")
        print(f"Draft prompt: {requests[i]['draft_prompt']}")
        print(f"Output: {output.outputs[0].text}")


if __name__ == "__main__":
    main()
