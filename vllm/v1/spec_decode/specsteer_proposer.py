# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from typing import Literal

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

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
        stream_role: Literal["augmented_drafter", "base_verifier"] = (
            "augmented_drafter"
        ),
    ):
        super().__init__(vllm_config=vllm_config, device=device, runner=runner)
        self.stream_role: Literal["augmented_drafter", "base_verifier"] = (
            stream_role
        )
        self._accepted_prefix_lens: dict[str, int] = defaultdict(int)
        self._logical_streams: dict[str, list[int]] = {}
        self._step_logits: list[torch.Tensor] = []

    def _get_prompt_for_drafting(self, req_state: CachedRequestState) -> list[int]:
        if self.stream_role == "base_verifier":
            # Base verifier must always track the canonical request prompt.
            prompt = req_state.prompt_token_ids
        else:
            prompt = getattr(req_state, "draft_prompt_token_ids", None)
            if prompt is None:
                prompt = req_state.prompt_token_ids
        return list(prompt or [])

    def _get_generated_output(self, req_state: CachedRequestState) -> list[int]:
        if self.stream_role == "base_verifier":
            # Keep verifier logical state tied to the base stream only.
            generated = req_state.output_token_ids
        else:
            generated = getattr(req_state, "generated_output", None)
            if generated is None:
                generated = req_state.output_token_ids
        return list(generated)

    def _get_drafter_prefix(self, req_state: CachedRequestState) -> list[int]:
        req_id = req_state.req_id
        stream = self._logical_streams.get(req_id)
        if stream is None:
            return self._get_prompt_for_drafting(req_state)
        return list(stream)

    def update_accepted_draft_token_counts(
        self,
        requests: dict[str, CachedRequestState],
        req_ids: list[str],
        accepted_draft_token_counts: torch.Tensor,
    ) -> None:
        """Trim speculative tails to accepted prefixes from runner bookkeeping."""
        if not req_ids:
            return

        accepted_counts = accepted_draft_token_counts.tolist()
        for req_id, accepted_count in zip(req_ids, accepted_counts, strict=False):
            req_state = requests.get(req_id)
            if req_state is None:
                self.on_request_removed(req_id)
                continue

            stream = self._logical_streams.get(req_id)
            if stream is None:
                continue

            base_prefix_len = self._accepted_prefix_lens.get(req_id, 0)
            accepted_prefix_len = max(
                0,
                min(base_prefix_len + int(accepted_count), len(stream)),
            )
            self._logical_streams[req_id] = stream[:accepted_prefix_len]
            self._accepted_prefix_lens[req_id] = accepted_prefix_len

    def on_request_removed(self, req_id: str) -> None:
        self._logical_streams.pop(req_id, None)
        self._accepted_prefix_lens.pop(req_id, None)

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
            committed_stream = prompt + generated
            prev_stream = self._logical_streams.get(req_id)

            # Preemption/resume or request mutation may rewrite the prompt/output.
            # Reset to the committed stream in those cases.
            if (
                prev_stream is None
                or len(prev_stream) < len(committed_stream)
                or prev_stream[: len(committed_stream)] != committed_stream
            ):
                self._logical_streams[req_id] = committed_stream
            else:
                # Keep only committed output; never reuse stale speculative tail.
                self._logical_streams[req_id] = prev_stream[: len(committed_stream)]

            # Next-step drafting always starts from draft_prompt + committed_output.
            self._accepted_prefix_lens[req_id] = len(committed_stream)

    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy sample and record per-step logits for SpecSteer recovery."""
        if self.use_local_argmax_reduction:
            # local argmax path does not expose full logits; force full logits for
            # specsteer since recovery requires logit vectors.
            logits = self.model.compute_logits(hidden_states)
            next_token_ids = logits.argmax(dim=-1)
        else:
            logits = self.model.compute_logits(hidden_states)
            next_token_ids = logits.argmax(dim=-1)

        self._step_logits.append(logits)
        return next_token_ids

    def prepare_next_token_ids_cpu(self, *args, **kwargs) -> torch.Tensor:
        assert self.runner is not None
        self._sync_logical_stream(self.runner.requests, self.runner.input_batch)
        return super().prepare_next_token_ids_cpu(*args, **kwargs)

    def prepare_next_token_ids_padded(self, *args, **kwargs):
        assert self.runner is not None
        self._sync_logical_stream(self.runner.requests, self.runner.input_batch)
        return super().prepare_next_token_ids_padded(*args, **kwargs)

    def propose(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.runner is not None
        self._sync_logical_stream(self.runner.requests, self.runner.input_batch)

        self._step_logits = []
        draft_token_ids = super().propose(*args, **kwargs)

        # Flatten per-step logits to [num_tokens, vocab] so the sampler can index
        # by cu_num_draft_tokens slices.
        if self._step_logits:
            augmented_logits = torch.stack(self._step_logits, dim=1).reshape(
                -1, self._step_logits[0].shape[-1]
            )
        else:
            augmented_logits = torch.empty(
                (0, self.model.config.vocab_size),
                dtype=torch.float32,
                device=draft_token_ids.device,
            )

        # Cache speculative continuation per request for next-step token lookup.
        req_ids = self.runner.input_batch.req_ids
        for req_idx, req_id in enumerate(req_ids):
            base_stream = self._logical_streams.get(req_id)
            if base_stream is None:
                continue
            row = draft_token_ids[req_idx]
            drafted = row[row >= 0].tolist()
            if drafted:
                self._logical_streams[req_id] = base_stream + drafted
        return draft_token_ids, augmented_logits
