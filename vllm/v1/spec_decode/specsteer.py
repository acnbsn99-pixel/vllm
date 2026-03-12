# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.speculative import SpeculativeConfig
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


class SpecSteerProposer(DraftModelProposer):
    """SpecSteer proposer implementation.

    Currently this reuses the draft-model proposer execution path while allowing
    SpecSteer-specific config and dispatch.
    """


class SpecSteerSampler:
    """Sampler wrapper for SpecSteer verification/sampling."""

    def __init__(self, sampler: Sampler, spec_config: SpeculativeConfig):
        self.sampler = sampler
        self.spec_config = spec_config
        self._rejection_sampler = RejectionSampler(sampler)

    def __call__(
        self,
        spec_decode_metadata: SpecDecodeMetadata,
        draft_probs,
        target_logits,
        sampling_metadata: SamplingMetadata,
    ):
        return self._rejection_sampler(
            spec_decode_metadata,
            draft_probs,
            target_logits,
            sampling_metadata,
        )
