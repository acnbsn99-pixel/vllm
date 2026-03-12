# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.engine.input_processor import InputProcessor


class _DummyModelConfig:
    max_model_len = 64
    runner_type = "generate"

    def try_get_generation_config(self):
        return {}

    def get_vocab_size(self):
        return 1024


class _DummyRenderer:
    def get_eos_token_id(self):
        return 0


def _make_input_processor() -> InputProcessor:
    proc = InputProcessor.__new__(InputProcessor)
    proc.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_size_local=1,
            local_engines_only=False,
        )
    )
    proc.model_config = _DummyModelConfig()
    proc.cache_config = None
    proc.lora_config = None
    proc.scheduler_config = None
    proc.speculative_config = None
    proc.structured_outputs_config = None
    proc.observability_config = None
    proc.generation_config_fields = {}
    proc.renderer = _DummyRenderer()
    proc.supports_mm_inputs = False
    proc.mm_encoder_cache_size = 0
    proc.skip_prompt_length_check = False
    proc.input_preprocessor = None
    return proc


def test_process_inputs_passes_through_draft_prompt_token_ids(monkeypatch):
    proc = _make_input_processor()

    proc._validate_params = lambda *args, **kwargs: None
    proc._validate_lora = lambda *args, **kwargs: None
    proc._validate_model_inputs = lambda *args, **kwargs: None

    monkeypatch.setattr(
        "vllm.v1.engine.input_processor.current_platform.validate_request",
        lambda *args, **kwargs: None,
    )

    params = SamplingParams(max_tokens=4)
    req = proc.process_inputs(
        request_id="req-1",
        prompt={
            "type": "token",
            "prompt_token_ids": [1, 2, 3],
            "draft_prompt_token_ids": [9, 8],
        },
        params=params,
        supported_tasks=("generate",),
        arrival_time=1.23,
    )

    assert req.prompt_token_ids == [1, 2, 3]
    assert req.draft_prompt_token_ids == [9, 8]


def test_process_inputs_uses_none_for_embeds_draft_prompt_token_ids(monkeypatch):
    proc = _make_input_processor()

    proc._validate_params = lambda *args, **kwargs: None
    proc._validate_lora = lambda *args, **kwargs: None
    proc._validate_model_inputs = lambda *args, **kwargs: None

    monkeypatch.setattr(
        "vllm.v1.engine.input_processor.current_platform.validate_request",
        lambda *args, **kwargs: None,
    )

    params = SamplingParams(max_tokens=4)
    req = proc.process_inputs(
        request_id="req-embeds",
        prompt={
            "type": "embeds",
            "prompt_embeds": torch.zeros((1, 1)),
        },
        params=params,
        supported_tasks=("generate",),
        arrival_time=2.34,
    )

    assert req.prompt_token_ids is None
    assert req.draft_prompt_token_ids is None
