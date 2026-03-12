# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.inputs.data import token_inputs
from vllm.inputs.preprocess import InputPreprocessor


def _make_preprocessor() -> InputPreprocessor:
    return InputPreprocessor.__new__(InputPreprocessor)


def test_process_text_tokenizes_draft_prompt_and_defaults():
    pre = _make_preprocessor()
    pre._tokenize_prompt = lambda prompt, tokenization_kwargs=None: [len(prompt)]

    with_draft = pre._process_text({"prompt": "abcd", "draft_prompt": "xy"})
    assert with_draft["prompt_token_ids"] == [4]
    assert with_draft["draft_prompt_token_ids"] == [2]

    without_draft = pre._process_text({"prompt": "abcd"})
    assert without_draft["prompt_token_ids"] == [4]
    assert without_draft["draft_prompt_token_ids"] == [4]


def test_process_tokens_uses_or_defaults_draft_prompt_token_ids():
    pre = _make_preprocessor()
    pre._truncate_inputs = lambda tokens, tokenization_kwargs=None: tokens[:1]

    with_draft = pre._process_tokens(
        {"prompt_token_ids": [1, 2], "draft_prompt_token_ids": [3, 4]}
    )
    assert with_draft["prompt_token_ids"] == [1]
    assert with_draft["draft_prompt_token_ids"] == [3]

    without_draft = pre._process_tokens({"prompt_token_ids": [1, 2]})
    assert without_draft["prompt_token_ids"] == [1]
    assert without_draft["draft_prompt_token_ids"] == [1]


def test_process_text_multimodal_defaults_or_tokenizes_draft_prompt():
    pre = _make_preprocessor()
    pre._tokenize_prompt = lambda prompt, tokenization_kwargs=None: [len(prompt)]
    pre._process_multimodal = lambda *args, **kwargs: {
        "type": "multimodal",
        "prompt_token_ids": [42],
    }

    without_draft = pre._process_text({
        "prompt": "hello",
        "multi_modal_data": {"image": []},
    })
    assert without_draft["prompt_token_ids"] == [42]
    assert without_draft["draft_prompt_token_ids"] == [42]

    with_draft = pre._process_text({
        "prompt": "hello",
        "draft_prompt": "xy",
        "multi_modal_data": {"image": []},
    })
    assert with_draft["prompt_token_ids"] == [42]
    assert with_draft["draft_prompt_token_ids"] == [2]


def test_process_tokens_multimodal_defaults_or_uses_draft_prompt_token_ids():
    pre = _make_preprocessor()
    pre._truncate_inputs = lambda tokens, tokenization_kwargs=None: tokens
    pre._process_multimodal = lambda *args, **kwargs: {
        "type": "multimodal",
        "prompt_token_ids": [9, 8],
    }

    without_draft = pre._process_tokens({
        "prompt_token_ids": [1, 2],
        "multi_modal_data": {"image": []},
    })
    assert without_draft["prompt_token_ids"] == [9, 8]
    assert without_draft["draft_prompt_token_ids"] == [9, 8]
    assert without_draft["draft_prompt_token_ids"] == without_draft["prompt_token_ids"]

    with_draft = pre._process_tokens({
        "prompt_token_ids": [1, 2],
        "draft_prompt_token_ids": [3, 4],
        "multi_modal_data": {"image": []},
    })
    assert with_draft["prompt_token_ids"] == [9, 8]
    assert with_draft["draft_prompt_token_ids"] == [3, 4]


def test_token_inputs_helper_keeps_draft_field_opt_in_only():
    inputs = token_inputs([1, 2], prompt="hello")

    assert "draft_prompt_token_ids" not in inputs
