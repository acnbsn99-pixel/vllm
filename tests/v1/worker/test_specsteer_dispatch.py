# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def test_sample_dispatches_to_specsteer_sampler_when_configured():
    runner = GPUModelRunner.__new__(GPUModelRunner)

    sampling_metadata = object()
    input_batch = Mock()
    input_batch.sampling_metadata = sampling_metadata
    runner.input_batch = input_batch

    runner.speculative_config = SimpleNamespace(method="specsteer")
    runner._specsteer_base_logits = "base_logits"
    runner._specsteer_steer_logits = "steer_logits"

    runner.specsteer_sampler = Mock(return_value="specsteer_output")
    runner.sampler = Mock(return_value="sampler_output")
    runner.spec_decode_sampler = Mock(return_value="spec_decode_output")

    spec_decode_metadata = object()
    logits = torch.randn(2, 4)

    output = GPUModelRunner._sample(
        runner,
        logits=logits,
        spec_decode_metadata=spec_decode_metadata,
    )

    assert output == "specsteer_output"
    runner.specsteer_sampler.assert_called_once_with(
        metadata=spec_decode_metadata,
        logits=logits,
        base_logits="base_logits",
        steer_logits="steer_logits",
        sampling_metadata=sampling_metadata,
    )
    runner.sampler.assert_not_called()
    runner.spec_decode_sampler.assert_not_called()
    input_batch.update_async_output_token_ids.assert_called_once_with()
