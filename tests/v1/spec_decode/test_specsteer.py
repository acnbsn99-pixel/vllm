# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID, RejectionSampler
from vllm.v1.specsteer import (
    CoSteerFusion,
    LinearFusion,
    PadTruncateVocabAligner,
    VerificationPolicy,
    ensure_greedy_only,
    resolve_draft_prompt,
)


def test_verification_policy_acceptance_rule_gamma_eps_first_rejection():
    policy = VerificationPolicy(gamma=0.6, eps=1e-6)

    p_llm_val = torch.tensor([[0.70, 0.40, 0.05]], dtype=torch.float32)
    p_base_val = torch.tensor([[0.90, 0.40, 0.01]], dtype=torch.float32)

    accept_mask, num_matches = policy.verify(p_llm_val, p_base_val)

    expected = p_llm_val > (0.6 * (p_base_val + 1e-6))
    assert torch.equal(accept_mask, expected)
    assert num_matches == 1


def test_linear_fusion_with_synthetic_logits():
    args = SimpleNamespace(fusion_coeff=0.5)
    fusion = LinearFusion(args)

    llm_logits = torch.tensor([[1.0, 2.0, -1.0]], dtype=torch.float32)
    slm_wo_logits = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    slm_with_logits = torch.tensor([[0.0, 2.0, -2.0]], dtype=torch.float32)

    fused = fusion.fuse(llm_logits, slm_wo_logits, slm_with_logits)
    expected = torch.log_softmax(llm_logits, dim=-1) + 0.5 * (
        torch.log_softmax(slm_with_logits, dim=-1)
        - torch.log_softmax(slm_wo_logits, dim=-1)
    )

    assert torch.allclose(fused, expected, atol=1e-6)


def test_costeer_fusion_with_synthetic_logits_is_normalized_and_finite():
    args = SimpleNamespace(T=3, alpha=1.5, beta=1.2, player_lambda=2.0, eta=10.0)
    fusion = CoSteerFusion(args)

    llm_logits = torch.tensor([[3.0, 1.0, -1.0]], dtype=torch.float32)
    slm_wo_logits = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    slm_with_logits = torch.tensor([[-2.0, 2.5, -1.0]], dtype=torch.float32)

    fused = fusion.fuse(llm_logits, slm_wo_logits, slm_with_logits)

    assert torch.isfinite(fused).all()
    assert torch.allclose(
        torch.logsumexp(fused, dim=-1),
        torch.zeros(1, dtype=fused.dtype),
        atol=1e-5,
    )


def test_pad_truncate_vocab_aligner_behavior():
    aligner = PadTruncateVocabAligner()

    ref_logits = torch.zeros((1, 5), dtype=torch.float32)
    short_aux = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    long_aux = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=torch.float32)

    aligned_short, aligned_long = aligner.align_to_ref(ref_logits, short_aux, long_aux)

    assert aligned_short.shape[-1] == 5
    assert torch.equal(aligned_short[0, :3], short_aux[0])
    assert torch.isneginf(aligned_short[0, 3:]).all()

    assert aligned_long.shape[-1] == 5
    assert torch.equal(aligned_long, long_aux[:, :5])


def test_parse_output_filters_placeholder_and_oov_tokens():
    output_token_ids = torch.tensor(
        [
            [1, 2, PLACEHOLDER_TOKEN_ID, 4],
            [PLACEHOLDER_TOKEN_ID, 6, 99, 3],
        ],
        dtype=torch.int64,
    )

    parsed, logprobs = RejectionSampler.parse_output(
        output_token_ids=output_token_ids,
        vocab_size=10,
    )

    assert parsed == [[1, 2, 4], [6, 3]]
    assert logprobs is None


def test_greedy_only_enforcement():
    ensure_greedy_only(SimpleNamespace(do_sample=False, temperature=0.0))

    with pytest.raises(ValueError, match="greedy"):
        ensure_greedy_only(SimpleNamespace(do_sample=True, temperature=0.0))

    with pytest.raises(ValueError, match="temperature"):
        ensure_greedy_only(SimpleNamespace(do_sample=False, temperature=0.7))


def test_draft_prompt_defaults_to_prompt_when_missing():
    prompt = "Explain speculative decoding in one sentence."

    assert resolve_draft_prompt(prompt, None) == prompt
    assert resolve_draft_prompt(prompt, "short draft") == "short draft"


def test_specsteer_sampler_output_layout_compatible_with_parse_output_shape():
    batch = 2
    max_spec_len = 3
    output_token_ids = torch.tensor(
        [
            [7, 8, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID],
            [4, 42, 9, PLACEHOLDER_TOKEN_ID],
        ],
        dtype=torch.int64,
    )

    assert output_token_ids.shape == (batch, max_spec_len + 1)

    parsed, _ = RejectionSampler.parse_output(
        output_token_ids=output_token_ids,
        vocab_size=10,
    )
    assert parsed == [[7, 8], [4, 9]]


def test_spec_decode_smoke_e2e_no_crash_and_non_empty_output():
    llm = LLM(
        model="hmellor/tiny-random-LlamaForCausalLM",
        enforce_eager=True,
        speculative_config={
            "method": "specsteer",
            "model": "hmellor/tiny-random-LlamaForCausalLM",
            "base_model": "hmellor/tiny-random-LlamaForCausalLM",
            "num_speculative_tokens": 2,
        },
    )

    outputs = llm.generate(
        ["Write exactly five random words."],
        SamplingParams(temperature=0, max_tokens=8),
    )

    assert outputs
    assert outputs[0].outputs
    assert len(outputs[0].outputs[0].token_ids) > 0


def test_spec_decode_smoke_e2e_defaults_draft_prompt_in_request_flow():
    llm = LLM(
        model="hmellor/tiny-random-LlamaForCausalLM",
        enforce_eager=True,
        speculative_config={
            "method": "specsteer",
            "model": "hmellor/tiny-random-LlamaForCausalLM",
            "base_model": "hmellor/tiny-random-LlamaForCausalLM",
            "num_speculative_tokens": 2,
        },
    )

    outputs = llm.generate(
        [{"prompt": "Give one short adjective."}],
        SamplingParams(temperature=0, max_tokens=4),
    )

    assert outputs
    assert outputs[0].outputs
    assert len(outputs[0].outputs[0].token_ids) > 0


def test_spec_decode_smoke_e2e_mixed_draft_prompt_requests_greedy():
    llm = LLM(
        model="hmellor/tiny-random-LlamaForCausalLM",
        enforce_eager=True,
        speculative_config={
            "method": "specsteer",
            "model": "hmellor/tiny-random-LlamaForCausalLM",
            "base_model": "hmellor/tiny-random-LlamaForCausalLM",
            "num_speculative_tokens": 2,
        },
    )

    outputs = llm.generate(
        [
            {
                "prompt": "Give one noun.",
                "draft_prompt": "Give one noun",
            },
            {
                "prompt": "Give one verb.",
            },
        ],
        SamplingParams(temperature=0, max_tokens=4),
    )

    assert len(outputs) == 2
    assert all(output.outputs for output in outputs)
    assert all(len(output.outputs[0].token_ids) > 0 for output in outputs)


def test_spec_decode_smoke_e2e_rejects_non_greedy_sampling():
    llm = LLM(
        model="hmellor/tiny-random-LlamaForCausalLM",
        enforce_eager=True,
        speculative_config={
            "method": "specsteer",
            "model": "hmellor/tiny-random-LlamaForCausalLM",
            "base_model": "hmellor/tiny-random-LlamaForCausalLM",
            "num_speculative_tokens": 2,
        },
    )

    with pytest.raises(ValueError, match="greedy"):
        llm.generate(
            ["Write exactly five random words."],
            SamplingParams(temperature=0.7, max_tokens=8),
        )
