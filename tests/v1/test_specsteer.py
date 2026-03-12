from vllm.v1.specsteer import _specstep_prefix_lens


def test_specstep_prefix_lens_all_accepted():
    verified_len, emitted_len = _specstep_prefix_lens(
        context_len=10,
        draft_len=4,
        accepted_tokens=4,
    )

    assert verified_len == 14
    assert emitted_len == 14


def test_specstep_prefix_lens_first_rejection_after_prefix():
    verified_len, emitted_len = _specstep_prefix_lens(
        context_len=10,
        draft_len=4,
        accepted_tokens=2,
    )

    # Keep only context + accepted as speculative-verified KV.
    assert verified_len == 12
    # Emit the recovery token, but do not include it in verified KV.
    assert emitted_len == 13


def test_specstep_prefix_lens_first_draft_rejected():
    verified_len, emitted_len = _specstep_prefix_lens(
        context_len=10,
        draft_len=4,
        accepted_tokens=0,
    )

    assert verified_len == 10
    assert emitted_len == 11
