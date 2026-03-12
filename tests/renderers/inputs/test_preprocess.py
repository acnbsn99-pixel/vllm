# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.renderers.inputs.preprocess import parse_dec_only_prompt, prompt_to_seq


def test_empty_input():
    assert prompt_to_seq([]) == []
    assert prompt_to_seq([[]]) == [[]]
    assert prompt_to_seq([[], []]) == [[], []]


def test_text_input():
    assert prompt_to_seq("foo") == ["foo"]
    assert prompt_to_seq(["foo"]) == ["foo"]
    assert prompt_to_seq(["foo", "bar"]) == ["foo", "bar"]


def test_token_input():
    assert prompt_to_seq([1, 2]) == [[1, 2]]
    assert prompt_to_seq([[1, 2]]) == [[1, 2]]
    assert prompt_to_seq([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]


def test_text_token_input():
    assert prompt_to_seq([[1, 2], "foo"]) == [[1, 2], "foo"]
    assert prompt_to_seq(["foo", [1, 2]]) == ["foo", [1, 2]]


def test_bytes_input():
    assert prompt_to_seq(b"foo") == [b"foo"]
    assert prompt_to_seq([b"foo"]) == [b"foo"]
    assert prompt_to_seq([b"foo", b"bar"]) == [b"foo", b"bar"]


def test_dict_input():
    assert prompt_to_seq({"prompt": "foo"}) == [{"prompt": "foo"}]
    assert prompt_to_seq([{"prompt": "foo"}]) == [{"prompt": "foo"}]
    assert prompt_to_seq([{"prompt": "foo"}, {"prompt_token_ids": [1, 2]}]) == [
        {"prompt": "foo"},
        {"prompt_token_ids": [1, 2]},
    ]


def test_decoder_only_draft_prompt_dict():
    assert parse_dec_only_prompt({"prompt": "foo", "draft_prompt": "bar"}) == {
        "prompt": "foo",
        "draft_prompt": "bar",
    }


def test_decoder_only_draft_prompt_token_ids_dict():
    assert parse_dec_only_prompt({
        "prompt_token_ids": [1, 2],
        "draft_prompt_token_ids": [3, 4],
    }) == {
        "prompt_token_ids": [1, 2],
        "draft_prompt_token_ids": [3, 4],
    }


def test_decoder_only_draft_prompt_requires_base_prompt():
    with pytest.raises(TypeError, match="Draft text prompt requires prompt text"):
        parse_dec_only_prompt({"draft_prompt": "bar"})

    with pytest.raises(TypeError, match="Draft token prompt requires prompt_token_ids"):
        parse_dec_only_prompt({"draft_prompt_token_ids": [3, 4]})
