# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import pytest
import torch

from vllm.v1.sample.specsteer_sampler import SpecSteerSampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata, SpecSteerMetadata


@dataclass
class _DummySamplerOutput:
    sampled_token_ids: torch.Tensor


class _DummySampler:
    def __call__(self, logits, sampling_metadata, predict_bonus_token=False):
        del sampling_metadata, predict_bonus_token
        return _DummySamplerOutput(
            sampled_token_ids=logits.argmax(dim=-1, keepdim=True)
        )


class _DummySamplingMetadata:
    def __init__(self, all_greedy: bool = True):
        self.all_greedy = all_greedy


def _metadata(device: torch.device) -> SpecDecodeMetadata:
    draft_token_ids = torch.tensor([1, 2], dtype=torch.int32, device=device)
    num_draft_tokens = [2]
    cu_num_draft_tokens = torch.tensor([2], dtype=torch.int32, device=device)
    cu_num_sampled_tokens = torch.tensor([3], dtype=torch.int32, device=device)
    target_logits_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    bonus_logits_indices = torch.tensor([2], dtype=torch.int32, device=device)
    logits_indices = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    return SpecDecodeMetadata(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=num_draft_tokens,
        cu_num_draft_tokens=cu_num_draft_tokens,
        cu_num_sampled_tokens=cu_num_sampled_tokens,
        target_logits_indices=target_logits_indices,
        bonus_logits_indices=bonus_logits_indices,
        logits_indices=logits_indices,
    )


def test_specsteer_sampler_accept_then_recover_without_bonus():
    device = torch.device("cpu")
    metadata = _metadata(device)
    sampler = SpecSteerSampler(_DummySampler(), gamma=0.9, enable_bonus_token=False)

    logits = torch.tensor(
        [
            [0.1, 3.0, 0.1],  # target step 0, accept token 1
            [0.1, 0.1, 3.0],  # target step 1, reject token 2 under aux probs
            [0.1, 0.1, 5.0],  # bonus logits (unused when rejection occurs)
        ],
        dtype=torch.float32,
    )
    base_logits = torch.tensor(
        [
            [0.1, 0.2, 0.1],
            [0.1, 0.1, 5.0],
        ],
        dtype=torch.float32,
    )
    steer_logits = base_logits.clone()

    out = sampler(
        metadata=metadata,
        logits=logits,
        base_logits=base_logits,
        steer_logits=steer_logits,
        sampling_metadata=_DummySamplingMetadata(all_greedy=True),
    )

    # token 1 accepted, then recovery at reject position chooses argmax(llm)=2.
    assert out.sampled_token_ids.tolist() == [[1, 2, -1]]


def test_specsteer_sampler_uses_specsteer_metadata_logits_when_args_missing():
    device = torch.device("cpu")
    metadata = _metadata(device)
    metadata.specsteer = SpecSteerMetadata(
        draft_token_ids=metadata.draft_token_ids,
        num_draft_tokens=metadata.num_draft_tokens,
        cu_num_draft_tokens=metadata.cu_num_draft_tokens,
        target_logits_indices=metadata.target_logits_indices,
        base_verifier_logits=torch.zeros((2, 3), dtype=torch.float32, device=device),
        augmented_drafter_logits=torch.zeros(
            (2, 3), dtype=torch.float32, device=device
        ),
        augmented_drafter_logits_indices=metadata.target_logits_indices,
    )
    sampler = SpecSteerSampler(_DummySampler(), enable_bonus_token=False)

    logits = torch.zeros((3, 3), dtype=torch.float32, device=device)
    out = sampler(
        metadata=metadata,
        logits=logits,
        base_logits=None,
        steer_logits=None,
        sampling_metadata=_DummySamplingMetadata(all_greedy=True),
    )
    assert out.sampled_token_ids.shape == (1, 3)


def test_specsteer_sampler_enforces_greedy_only():
    device = torch.device("cpu")
    metadata = _metadata(device)
    sampler = SpecSteerSampler(_DummySampler())
    logits = torch.zeros((3, 3), dtype=torch.float32, device=device)

    with pytest.raises(ValueError, match="greedy-only"):
        sampler(
            metadata=metadata,
            logits=logits,
            base_logits=logits[:2],
            steer_logits=logits[:2],
            sampling_metadata=_DummySamplingMetadata(all_greedy=False),
        )


def test_specsteer_sampler_requires_aux_logits():
    device = torch.device("cpu")
    metadata = _metadata(device)
    sampler = SpecSteerSampler(_DummySampler())
    logits = torch.zeros((3, 3), dtype=torch.float32, device=device)

    with pytest.raises(ValueError, match="requires both base and augmented"):
        sampler(
            metadata=metadata,
            logits=logits,
            base_logits=None,
            steer_logits=None,
            sampling_metadata=_DummySamplingMetadata(all_greedy=True),
        )
