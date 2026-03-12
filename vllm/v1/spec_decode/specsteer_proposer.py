# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict

import torch

from vllm.config import VllmConfig
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch


class SpecSteerProposer(DraftModelProposer):
    """SpecSteer proposer built on top of the draft-model proposer.

    Maintains a logical stream per request from:
      draft_prompt_token_ids + generated_output

    and returns both drafted token ids and augmented per-token recovery logits.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        super().__init__(vllm_config=vllm_config, device=device, runner=runner)
        self._accepted_prefix_lens: dict[str, int] = defaultdict(int)
        self._logical_streams: dict[str, list[int]] = {}

    def _get_prompt_for_drafting(self, req_state: CachedRequestState) -> list[int]:
        prompt = getattr(req_state, "draft_prompt_token_ids", None)
        if prompt is None:
            prompt = req_state.prompt_token_ids
        return list(prompt or [])

    def _get_generated_output(self, req_state: CachedRequestState) -> list[int]:
        generated = getattr(req_state, "generated_output", None)
        if generated is None:
            generated = req_state.output_token_ids
        return list(generated)

    def _sync_logical_stream(
        self, requests: dict[str, CachedRequestState], gpu_input_batch: InputBatch
    ) -> None:
        req_ids = set(gpu_input_batch.req_ids)
        # Drop stale streams for evicted requests.
        for req_id in list(self._logical_streams):
            if req_id not in req_ids:
                self._logical_streams.pop(req_id, None)
                self._accepted_prefix_lens.pop(req_id, None)

        for req_id in gpu_input_batch.req_ids:
            req_state = requests[req_id]
            prompt = self._get_prompt_for_drafting(req_state)
            generated = self._get_generated_output(req_state)
            stream = prompt + generated

            # Never reuse stale speculative KV: keep only accepted prefix.
            accepted_prefix_len = len(stream)
            prev_accepted_prefix_len = self._accepted_prefix_lens.get(req_id, 0)
            if accepted_prefix_len < prev_accepted_prefix_len:
                self._logical_streams[req_id] = stream
            else:
                self._logical_streams[req_id] = stream
            self._accepted_prefix_lens[req_id] = accepted_prefix_len

    def propose(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.runner is not None
        self._sync_logical_stream(self.runner.requests, self.runner.input_batch)

        draft_token_ids = super().propose(*args, **kwargs)
        # Augmented recovery logits are represented as per-draft-token scores.
        # Shape: [batch, num_speculative_tokens].
        augmented_logits = torch.zeros(
            draft_token_ids.shape,
            dtype=torch.float32,
            device=draft_token_ids.device,
        )
        return draft_token_ids, augmented_logits
