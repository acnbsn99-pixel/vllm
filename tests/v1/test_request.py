# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.request import Request, RequestStatus, StreamingUpdate


def test_request_status_fmt_str():
    """Test that the string representation of RequestStatus is correct."""
    assert f"{RequestStatus.WAITING}" == "WAITING"
    assert f"{RequestStatus.WAITING_FOR_FSM}" == "WAITING_FOR_FSM"
    assert f"{RequestStatus.WAITING_FOR_REMOTE_KVS}" == "WAITING_FOR_REMOTE_KVS"
    assert f"{RequestStatus.WAITING_FOR_STREAMING_REQ}" == "WAITING_FOR_STREAMING_REQ"
    assert f"{RequestStatus.RUNNING}" == "RUNNING"
    assert f"{RequestStatus.PREEMPTED}" == "PREEMPTED"
    assert f"{RequestStatus.FINISHED_STOPPED}" == "FINISHED_STOPPED"
    assert f"{RequestStatus.FINISHED_LENGTH_CAPPED}" == "FINISHED_LENGTH_CAPPED"
    assert f"{RequestStatus.FINISHED_ABORTED}" == "FINISHED_ABORTED"
    assert f"{RequestStatus.FINISHED_IGNORED}" == "FINISHED_IGNORED"


def test_request_from_engine_core_request_preserves_draft_prompt_token_ids():
    req = EngineCoreRequest(
        request_id="req-1",
        prompt_token_ids=[1, 2],
        draft_prompt_token_ids=[8, 9],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )

    v1_req = Request.from_engine_core_request(req, block_hasher=None)

    assert v1_req.prompt_token_ids == [1, 2]
    assert v1_req.draft_prompt_token_ids == [8, 9]


def test_streaming_update_from_request_includes_draft_prompt_token_ids():
    req = Request(
        request_id="req-2",
        prompt_token_ids=[3, 4],
        draft_prompt_token_ids=[7],
        sampling_params=SamplingParams(max_tokens=2),
        pooling_params=None,
        arrival_time=2.0,
        resumable=True,
    )

    update = StreamingUpdate.from_request(req)

    assert update is not None
    assert update.prompt_token_ids == [3, 4]
    assert update.draft_prompt_token_ids == [7]


def test_streaming_update_non_resumable_request_is_ignored():
    req = Request(
        request_id="req-3",
        prompt_token_ids=[3, 4],
        draft_prompt_token_ids=[7],
        sampling_params=SamplingParams(max_tokens=2),
        pooling_params=None,
        arrival_time=2.0,
        resumable=False,
    )

    assert StreamingUpdate.from_request(req) is None
