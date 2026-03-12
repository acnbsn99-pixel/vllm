# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import torch
import torch.nn as nn

from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


class SpecSteerSampler(nn.Module):
    def __init__(
        self,
        sampler: Sampler,
        *,
        gamma: float = 0.6,
        eps: float = 1e-10,
        fusion_method: str = "costeer",
        linear_coeff: float | None = None,
        costeer_T: int = 20,
        costeer_alpha: float = 2.0,
        costeer_beta: float = 1.5,
        costeer_player_lambda: float = 2.0,
        costeer_eta: float = 10.0,
        enable_bonus_token: bool = False,
    ):
        super().__init__()
        self.sampler = sampler
        self.gamma = gamma
        self.eps = eps
        self.fusion_method = fusion_method.lower()
        self.linear_coeff = (
            costeer_beta if linear_coeff is None else linear_coeff
        )
        self.costeer_T = costeer_T
        self.costeer_alpha = costeer_alpha
        self.costeer_beta = costeer_beta
        self.costeer_player_lambda = costeer_player_lambda
        self.costeer_eta = costeer_eta
        self.enable_bonus_token = enable_bonus_token

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens + batch_size, vocab_size]
        logits: torch.Tensor,
        # [num_tokens, vocab_size_aux]
        base_logits: torch.Tensor | None,
        # [num_tokens, vocab_size_aux]
        steer_logits: torch.Tensor | None,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        self._validate_greedy_only(sampling_metadata)
        if metadata.specsteer is not None:
            if base_logits is None:
                base_logits = metadata.specsteer.base_verifier_logits
            if steer_logits is None:
                steer_logits = metadata.specsteer.augmented_drafter_logits

        if base_logits is None or steer_logits is None:
            raise ValueError(
                "SpecSteer requires both base and augmented auxiliary logits."
            )

        target_logits = logits[metadata.target_logits_indices].to(torch.float32)

        bonus_logits = logits[metadata.bonus_logits_indices]
        bonus_sampler_output = self.sampler(
            logits=bonus_logits,
            sampling_metadata=replace(sampling_metadata, max_num_logprobs=-1),
            predict_bonus_token=True,
        )
        bonus_token_ids = bonus_sampler_output.sampled_token_ids.squeeze(-1)

        base_logits = self._align_aux_logits(
            target_logits, base_logits.to(torch.float32)
        )
        steer_logits = self._align_aux_logits(
            target_logits, steer_logits.to(torch.float32)
        )

        target_probs = torch.softmax(target_logits, dim=-1)
        base_probs = torch.softmax(base_logits, dim=-1)

        output_token_ids = torch.full(
            (len(metadata.num_draft_tokens), metadata.max_spec_len + 1),
            PLACEHOLDER_TOKEN_ID,
            dtype=torch.int32,
            device=logits.device,
        )
        accepted_draft_token_counts = torch.empty(
            len(metadata.num_draft_tokens), dtype=torch.int32, device=logits.device
        )

        for req_idx, req_len in enumerate(metadata.num_draft_tokens):
            start = (
                0 if req_idx == 0 else int(metadata.cu_num_draft_tokens[req_idx - 1])
            )
            end = start + req_len
            draft = metadata.draft_token_ids[start:end].long()

            accepted_all = True
            reject_pos = req_len
            for i in range(req_len):
                token_id = draft[i]
                p_llm = target_probs[start + i, token_id]
                p_base = base_probs[start + i, token_id]
                if p_llm > (self.gamma * (p_base + self.eps)):
                    output_token_ids[req_idx, i] = token_id.to(torch.int32)
                else:
                    accepted_all = False
                    reject_pos = i
                    break

            if accepted_all:
                accepted_draft_token_counts[req_idx] = req_len
                if self.enable_bonus_token:
                    output_token_ids[req_idx, req_len] = bonus_token_ids[req_idx]
                continue

            accepted_draft_token_counts[req_idx] = reject_pos

            fused_logits = self._fuse_logits(
                target_logits[start + reject_pos : start + reject_pos + 1],
                base_logits[start + reject_pos : start + reject_pos + 1],
                steer_logits[start + reject_pos : start + reject_pos + 1],
            )
            recovered_token = fused_logits.argmax(dim=-1).to(torch.int32)
            output_token_ids[req_idx, reject_pos] = recovered_token[0]

        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=None,
            accepted_draft_token_counts=accepted_draft_token_counts,
        )

    def _fuse_logits(
        self,
        llm_logits: torch.Tensor,
        base_logits: torch.Tensor,
        steer_logits: torch.Tensor,
    ) -> torch.Tensor:
        llm_log = torch.log_softmax(torch.nan_to_num(llm_logits, nan=-100.0), dim=-1)
        base_log = torch.log_softmax(torch.nan_to_num(base_logits, nan=-100.0), dim=-1)
        steer_log = torch.log_softmax(
            torch.nan_to_num(steer_logits, nan=-100.0), dim=-1
        )
        delta = steer_log - base_log
        delta[torch.isnan(delta)] = 0.0

        if self.fusion_method == "linear":
            return llm_log + self.linear_coeff * delta

        q_sum = torch.zeros_like(llm_log)
        log_player = llm_log.clone()
        prev_log_player = llm_log.clone()
        for t in range(1, self.costeer_T + 1):
            q_sum += (
                self.costeer_alpha * (log_player - llm_log) + self.costeer_beta * delta
            )
            denom = t * self.costeer_player_lambda + 1.0 / self.costeer_eta
            log_player = (
                t * self.costeer_player_lambda * llm_log
                + q_sum
                + prev_log_player / self.costeer_eta
            ) / denom
            log_player = torch.log_softmax(log_player, dim=-1)
            prev_log_player = log_player
        return log_player

    @staticmethod
    def _align_aux_logits(
        ref_logits: torch.Tensor,
        aux_logits: torch.Tensor,
    ) -> torch.Tensor:
        ref_vocab = ref_logits.shape[-1]
        aux_vocab = aux_logits.shape[-1]
        if aux_vocab == ref_vocab:
            return aux_logits
        if aux_vocab < ref_vocab:
            padding = torch.full(
                (aux_logits.shape[0], ref_vocab - aux_vocab),
                -float("inf"),
                device=aux_logits.device,
                dtype=aux_logits.dtype,
            )
            return torch.cat([aux_logits, padding], dim=-1)
        return aux_logits[..., :ref_vocab]

    @staticmethod
    def _validate_greedy_only(sampling_metadata: SamplingMetadata) -> None:
        if not sampling_metadata.all_greedy:
            raise ValueError(
                "SpecSteer currently supports greedy-only decoding. "
                "Use temperature=0 for all requests."
            )
