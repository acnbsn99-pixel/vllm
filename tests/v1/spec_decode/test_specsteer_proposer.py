# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.spec_decode.specsteer_proposer import SpecSteerProposer
from vllm.v1.worker.gpu_input_batch import CachedRequestState


def _make_req_state(
    prompt_token_ids: list[int],
    draft_prompt_token_ids: list[int],
    output_token_ids: list[int],
) -> CachedRequestState:
    return CachedRequestState(
        req_id="req-1",
        prompt_token_ids=prompt_token_ids,
        draft_prompt_token_ids=draft_prompt_token_ids,
        prompt_embeds=None,
        mm_features=[],
        sampling_params=None,
        generator=None,
        block_ids=([],),
        num_computed_tokens=0,
        output_token_ids=output_token_ids,
    )


def test_specsteer_drafter_prefix_differs_from_verifier_prefix() -> None:
    req_state = _make_req_state(
        prompt_token_ids=[11, 12, 13],
        draft_prompt_token_ids=[101, 102, 103, 104],
        output_token_ids=[201, 202],
    )
    proposer = object.__new__(SpecSteerProposer)
    proposer._logical_streams = {}

    # Verifier stream uses the original prompt prefix.
    assert req_state.get_token_id(0) == 11
    assert req_state.get_token_id(3) == 201

    # Drafter stream uses draft_prompt_token_ids + generated output.
    assert proposer._get_drafter_token_id(req_state, 0) == 101
    assert proposer._get_drafter_token_id(req_state, 3) == 104
    assert proposer._get_drafter_token_id(req_state, 4) == 201


def test_specsteer_on_request_removed_cleans_logical_stream_state() -> None:
    proposer = object.__new__(SpecSteerProposer)
    proposer._logical_streams = {"req-1": [1, 2], "req-2": [3]}
    proposer._accepted_prefix_lens = {"req-1": 2, "req-2": 1}

    proposer.on_request_removed("req-1")

    assert "req-1" not in proposer._logical_streams
    assert "req-1" not in proposer._accepted_prefix_lens
    assert proposer._logical_streams["req-2"] == [3]
    assert proposer._accepted_prefix_lens["req-2"] == 1


def test_update_accepted_draft_token_counts_trims_speculative_tail() -> None:
    proposer = object.__new__(SpecSteerProposer)
    proposer._logical_streams = {"req-1": [10, 11, 20, 21, 31, 32]}
    proposer._accepted_prefix_lens = {"req-1": 4}

    req_state = _make_req_state(
        prompt_token_ids=[1],
        draft_prompt_token_ids=[10, 11],
        output_token_ids=[20, 21],
    )
    requests = {"req-1": req_state}

    proposer.update_accepted_draft_token_counts(
        requests,
        ["req-1"],
        torch.tensor([1], dtype=torch.int32),
    )

    # Accepted prefix (4) + accepted draft token (1) = 5 tokens retained.
    assert proposer._logical_streams["req-1"] == [10, 11, 20, 21, 31]
    assert proposer._accepted_prefix_lens["req-1"] == 5


def test_sync_logical_stream_resets_on_prompt_change() -> None:
    proposer = object.__new__(SpecSteerProposer)
    proposer._logical_streams = {"req-1": [10, 11, 20, 21, 31]}
    proposer._accepted_prefix_lens = {"req-1": 4}

    req_state = _make_req_state(
        prompt_token_ids=[1],
        draft_prompt_token_ids=[99, 100],
        output_token_ids=[50],
    )
    requests = {"req-1": req_state}

    class _Batch:
        req_ids = ["req-1"]

    proposer._sync_logical_stream(requests, _Batch())

    # Stream is reset to draft_prompt + committed output.
    assert proposer._logical_streams["req-1"] == [99, 100, 50]
    assert proposer._accepted_prefix_lens["req-1"] == 3
