# SpecSteer

SpecSteer is available first in **offline** usage (Python `LLM(...)` + `generate(...)`).
It supports per-request `draft_prompt` input and currently requires **greedy**
decoding (`temperature=0`).

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    speculative_config={
        "method": "specsteer",
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "num_speculative_tokens": 4,
        # optional
        "base_model": "meta-llama/Llama-3.2-1B-Instruct",
    },
)

outputs = llm.generate(
    [
        {
            "prompt": "Describe speculative decoding in one paragraph.",
            "draft_prompt": "Give a short rough draft first.",
        }
    ],
    sampling_params=SamplingParams(temperature=0, max_tokens=64),
)
```

## Supported mode

- **Offline first**: use SpecSteer through the Python API (`LLM.generate`).
- For a runnable script, see
  [`examples/offline_inference/specsteer_draft_prompt.py`](../../../examples/offline_inference/specsteer_draft_prompt.py).

## Required config

When `speculative_config["method"] == "specsteer"`, configure:

- required: `model`
- required: `num_speculative_tokens`
- optional: `base_model` (if omitted, the same model id can be reused)

## `draft_prompt` in server mode

`draft_prompt` is currently an offline input feature. In OpenAI-compatible server
requests, the public chat/completions schemas do not expose a `draft_prompt`
field, so this behavior is currently unsupported/partial for server clients.

## Expected output semantics

SpecSteer follows speculative decoding output composition semantics:

- accepted draft tokens,
- plus optional recovery tokens (after first rejection),
- plus an optional bonus token when bonus-token behavior is enabled.

In short: output tokens are composed from accepted tokens plus optional
recovery and optional bonus behavior.
