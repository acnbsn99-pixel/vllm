# -*- coding: utf-8 -*-
# spec_sss.py

import torch
import time
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
from transformers import LogitsProcessorList
from transformers.generation.candidate_generator import AssistedCandidateGenerator
from transformers.generation.utils import _prepare_attention_mask, _prepare_token_type_ids

# =========================================================================
# 0. 辅助打印与 Input 处理 (缺少的正是这部分)
# =========================================================================
def _print_ctx(tag: str, ids: torch.Tensor, tok, args, max_tail: int = 48):
    if not hasattr(args, 'debug') or not args.debug:
        return
    ctx_list = ids[0].tolist()
    tail = ctx_list[-max_tail:]
    try:
        text = tok.decode(tail, skip_special_tokens=False)
    except Exception:
        text = "<decode_error>"
    print(f"[CTX] {tag}: len={ids.shape[1]}  tail_ids={tail}")
    print(f"[TXT] {tag}: {text}")

def messages_to_input_ids(tok, messages, device, add_generation_prompt=True):
    """通用的 Chat Template 处理"""
    try:
        # 尝试使用 enable_thinking=False (针对某些新版模型)
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False
        )
    except TypeError:
        try:
            # 标准调用
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        except Exception:
            # 兜底拼接 ChatML 格式
            text = ""
            for m in messages:
                text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            if add_generation_prompt:
                text += "<|im_start|>assistant\n"

    inputs = tok(text, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def initialize_specsteer_models(speculative_config, load_model_fn):
    """Initialize augmented and base verifier models for specsteer.

    The augmented drafter always comes from ``speculative_config.model``.
    The optional verifier comes from ``speculative_config.base_model``.
    When ``base_model`` is omitted (or equals ``model``), the same model object
    is reused and caller should keep distinct logical state/KV streams.
    """
    augmented_model_name = getattr(speculative_config, "model", None)
    if not augmented_model_name:
        raise ValueError("speculative_config.model must be set for specsteer")

    augmented_model = load_model_fn(augmented_model_name)

    base_model_name = getattr(speculative_config, "base_model", None)
    if base_model_name is None or base_model_name == augmented_model_name:
        # Reuse one weights object when feasible; verification/drafting state is
        # still isolated by constructing separate stream/generator state.
        return augmented_model, augmented_model, augmented_model_name

    # Behavior note: in this revision we use an explicit second runner/model for
    # a distinct verifier model id. Cross-model weight sharing is intentionally
    # not attempted here to keep correctness straightforward.
    base_model = load_model_fn(base_model_name)
    return augmented_model, base_model, base_model_name

# =========================================================================
# 1. 融合策略 (Fusion Strategy)
# =========================================================================

class FusionStrategy(ABC):
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def fuse(self, llm_logits, slm_wo_logits, slm_with_logits):
        """
        Args:
            llm_logits: [1, V_large]
            slm_wo_logits: [1, V_small] (Base)
            slm_with_logits: [1, V_small] (augmented)
        Returns:
            fused_log_probs: [1, V_target]
        """
        pass

class CoSteerFusion(FusionStrategy):
    def __init__(self, args):
        super().__init__(args)
        self.T = getattr(args, 'T', 20)
        self.alpha = getattr(args, 'alpha', 2.0)
        self.beta = getattr(args, 'beta', 1.0)
        self.player_lambda = getattr(args, 'player_lambda', 2.0)
        self.eta = getattr(args, 'eta', 10.0)

    @torch.no_grad()
    def fuse(self, llm_logits, slm_wo_logits, slm_with_logits):
        llm_logits = torch.nan_to_num(llm_logits, nan=-100.0)
        slm_with_logits = torch.nan_to_num(slm_with_logits, nan=-100.0)
        slm_wo_logits = torch.nan_to_num(slm_wo_logits, nan=-100.0)

        ref_log = torch.log_softmax(llm_logits, dim=-1)
        with_log = torch.log_softmax(slm_with_logits, dim=-1)
        wo_log  = torch.log_softmax(slm_wo_logits, dim=-1)

        # Context Gain: (Augmented - Base)
        delta = with_log - wo_log
        delta[torch.isnan(delta)] = 0.0

        Q_sum = torch.zeros_like(ref_log)
        log_player = ref_log.clone()
        prev_log_player = ref_log.clone()

        # Iterative Update (CoSteer Logic)
        for t in range(1, self.T + 1):
            Q_sum += self.alpha * (log_player - ref_log) + self.beta * delta
            denom = t * self.player_lambda + 1.0 / self.eta
            log_player = (t * self.player_lambda * ref_log + Q_sum + prev_log_player / self.eta) / denom
            log_player = torch.log_softmax(log_player, dim=-1)
            prev_log_player = log_player

        return log_player

class LinearFusion(FusionStrategy):
    def __init__(self, args):
        super().__init__(args)
        self.coeff = getattr(args, 'fusion_coeff', getattr(args, 'beta', 1.0))

    @torch.no_grad()
    def fuse(self, llm_logits, slm_wo_logits, slm_with_logits):
        llm_logits = torch.nan_to_num(llm_logits, nan=-100.0)
        slm_with_logits = torch.nan_to_num(slm_with_logits, nan=-100.0)
        slm_wo_logits = torch.nan_to_num(slm_wo_logits, nan=-100.0)

        ref_log = torch.log_softmax(llm_logits, dim=-1)
        with_log = torch.log_softmax(slm_with_logits, dim=-1)
        wo_log  = torch.log_softmax(slm_wo_logits, dim=-1)

        context_gain = with_log - wo_log
        context_gain[torch.isnan(context_gain)] = 0.0

        fused_log = ref_log + self.coeff * context_gain
        return fused_log

def get_fusion_strategy(args):
    """工厂函数：根据参数选择融合策略"""
    method = getattr(args, 'fusion_method', 'costeer').lower()
    if method == 'linear':
        return LinearFusion(args)
    return CoSteerFusion(args)

# =======================================================
# 2. 抽象：Verify Stream（LLM / Base SLM 共用）
# =======================================================
class VerifyStream:
    """
    把 LLM 与 SLM-wo 在 verify 阶段共同的逻辑抽象成一个可复用的“验证流”：
    - kwargs 准备（attention_mask/token_type_ids + cache_position 扩展）
    - forward 得到 logits_seq
    - 从 logits_seq gather draft token 概率
    - append 后 crop + _update_model_kwargs_for_generation 更新缓存
    """
    def __init__(
        self,
        model,
        init_cur_len: int,
        init_device: torch.device,
        init_model_kwargs: Optional[Dict[str, Any]] = None,
        output_device: Optional[torch.device] = None,
    ):
        self.model = model
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.model_device = model.device  # 推理所在 device（prepare_inputs/forward）
        self.output_device = output_device if output_device is not None else self.model_device

        if init_model_kwargs is None:
            init_model_kwargs = {}
        # 初始化 cache_position 等
        self.model_kwargs = model._get_initial_cache_position(init_cur_len, init_device, init_model_kwargs)

    def _build_candidate_kwargs(self, seq_len: int, draft_len: int) -> Dict[str, Any]:
        kwargs = self.model_kwargs.copy()
        kwargs = _prepare_attention_mask(kwargs, seq_len, self.is_encoder_decoder)
        kwargs = _prepare_token_type_ids(kwargs, seq_len)

        # 扩展 cache_position（保持与你原实现一致）
        if "cache_position" in kwargs:
            last_pos = kwargs["cache_position"][-1:]
            new_pos = torch.arange(1, draft_len + 1, device=self.model_device) + last_pos
            kwargs["cache_position"] = torch.cat([kwargs["cache_position"], new_pos], dim=0)

        return kwargs

    @torch.no_grad()
    def forward_verify(self, verify_input_ids: torch.LongTensor, draft_len: int):
        """
        输入 verify_input_ids（可能不在本 model device 上），在本 stream 的 model 上 forward，
        返回 outputs 与 logits_seq = outputs.logits[:, -draft_len-1:].
        """
        seq_len = verify_input_ids.shape[1]
        candidate_kwargs = self._build_candidate_kwargs(seq_len=seq_len, draft_len=draft_len)

        # 确保输入在本模型 device
        verify_ids_local = verify_input_ids.to(self.model_device)

        model_inputs = self.model.prepare_inputs_for_generation(verify_ids_local, **candidate_kwargs)
        outputs = self.model(**model_inputs, return_dict=True)
        logits_seq = outputs.logits[:, -draft_len - 1:].float()  # 保持原逻辑
        
        got = int(logits_seq.shape[1])
        exp = int(draft_len + 1)
        pkv = model_inputs.get("past_key_values", None)
        cp = model_inputs.get("cache_position", None)
        pkv_len = pkv.get_seq_length() if (hasattr(pkv, "get_seq_length") and pkv is not None) else None
        cp_len = int(cp.numel()) if (cp is not None and hasattr(cp, "numel")) else None
        in_len = int(model_inputs["input_ids"].shape[1]) if model_inputs.get("input_ids") is not None else None

        print(
            f"[VERIFY_WINDOW_CHECKING] logits_seq_len={got} != expected={exp}. "
            f"draft_len={draft_len}, input_ids_len(after_prepare)={in_len}, "
            f"cache_position_len={cp_len}, pkv_seq_len={pkv_len}. "
        )
        return outputs, logits_seq

    @torch.no_grad()
    def draft_token_probs(self, logits_seq: torch.Tensor, draft_tokens: torch.LongTensor) -> torch.Tensor:
        """
        从 logits_seq[:, :-1, :] 计算每个 draft token 的概率并 gather。
        返回 shape [1, draft_len]，并搬到 output_device（保持 base->model.device 的行为一致）。
        """
        probs = torch.softmax(logits_seq[:, :-1, :], dim=-1)  # [1, draft_len, V]

        gather_indices = draft_tokens.to(self.output_device).unsqueeze(-1)  # [1, draft_len, 1]
        probs = probs.to(self.output_device)
        p_val = torch.gather(probs, -1, gather_indices).squeeze(-1)  # [1, draft_len]
        return p_val

    def update_after_append(self, outputs, keep_len: int, n_new_tokens: int):
        """
        Efficiently update cache and model kwargs.
        Args:
            outputs: Model outputs containing past_key_values
            keep_len: The total length of tokens to keep in the cache (Context + Accepted)
            n_new_tokens: How many new tokens were accepted into the cache in this step
        """
        if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
            # Crop to the exact valid length. 
            # 4.57+ DynamicCache supports crop(len) efficiently.
            outputs.past_key_values.crop(keep_len)
            
            self.model_kwargs = self.model._update_model_kwargs_for_generation(
                outputs,
                self.model_kwargs,
                is_encoder_decoder=self.is_encoder_decoder,
                num_new_tokens=n_new_tokens,
            )

# =======================================================
# 2.5 抽象：Verification Policy（验收策略）
# =======================================================
class VerificationPolicy:
    """
    抽象验收逻辑：
    给定 p_llm_val 与 p_base_val（shape [1, draft_len]），计算 accept_mask 与 num_matches。
    """
    def __init__(self, gamma: float, eps: float = 1e-10):
        self.gamma = gamma
        self.eps = eps

    @torch.no_grad()
    def verify(self, p_llm_val: torch.Tensor, p_base_val: torch.Tensor):
        accept_mask = p_llm_val > (self.gamma * (p_base_val + self.eps))

        reject_indices = (~accept_mask).nonzero(as_tuple=False)
        if reject_indices.numel() == 0:
            num_matches = accept_mask.shape[1]
        else:
            num_matches = reject_indices[0, 1].item()

        return accept_mask, num_matches

# =======================================================
# 2.6 抽象：Vocabulary Alignment / Mapping（词表对齐与映射）
# =======================================================
class VocabAligner(ABC):
    """
    词表对齐/映射抽象层：把 auxiliary logits（base/pers）对齐到 reference logits（llm）的 vocab 空间。
    注意：当前实现保持你原来的 pad/truncate 行为，不引入任何新映射假设。
    """
    @abstractmethod
    def align_to_ref(
        self,
        ref_logits: torch.Tensor,
        *aux_logits: torch.Tensor,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            ref_logits: [1, V_ref]
            aux_logits: one or more tensors, each [1, V_aux]
            device: where to allocate padding if needed
        Returns:
            aligned_aux_logits: tuple of tensors, each [1, V_ref]
        """
        pass

class PadTruncateVocabAligner(VocabAligner):
    """
    等价于你原来的逻辑：若 V_ref > V_aux 则在末尾 pad(-inf)；若 V_ref < V_aux 则 truncate。
    这对应一个“前缀 identity 映射”：id i -> i（仅在 min(V_ref, V_aux) 上定义）。
    """
    def __init__(self, pad_value: float = -float("inf")):
        self.pad_value = pad_value

    def _align_one(self, ref_v: int, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        v = x.shape[-1]
        if v == ref_v:
            return x
        if ref_v > v:
            pad = torch.full((x.shape[0], ref_v - v), self.pad_value, device=device, dtype=x.dtype)
            return torch.cat([x, pad], dim=-1)
        # ref_v < v
        return x[..., :ref_v]

    def align_to_ref(
        self,
        ref_logits: torch.Tensor,
        *aux_logits: torch.Tensor,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = ref_logits.device
        ref_v = ref_logits.shape[-1]
        aligned = tuple(self._align_one(ref_v, x, device=device) for x in aux_logits)
        return aligned

def get_vocab_aligner(args) -> VocabAligner:
    """
    工厂函数：保留扩展点。当前默认行为等价于原实现（pad/truncate）。
    """
    _ = getattr(args, "vocab_align_method", "pad_truncate")
    return PadTruncateVocabAligner(pad_value=-float("inf"))

# =======================================================
# 3. 核心解码逻辑 (Single Draft, Dual Verify)
# =======================================================
def _context_assisted_decoding(
    model,
    assistant_model,
    main_input_ids: torch.LongTensor,
    asst_input_ids: torch.LongTensor,
    generation_config,
    logits_processor: Optional[LogitsProcessorList] = None,
    max_length: int = None,
    pad_token_id: int = None,
    eos_token_id: int = None,
    model_kwargs_main: Optional[Dict[str, Any]] = None,
    model_kwargs_asst: Optional[Dict[str, Any]] = None, # 这里的 asst 其实对应 augmented 的 cache
    fusion=None,
    args=None,
    tokenizer=None,
):
    """
    实现 "Knowledge-Aware Speculative Decoding" (Single Draft, Dual Verify)
    """
    if logits_processor is None: logits_processor = LogitsProcessorList()
    if model_kwargs_main is None: model_kwargs_main = {}
    if model_kwargs_asst is None: model_kwargs_asst = {}
    if max_length is None: max_length = generation_config.max_length

    # 核心超参 gamma
    gamma = getattr(args, 'gamma', 0.6)

    # --- Pad & Stop Token Fix (保持原样) ---
    if pad_token_id is None:
        if tokenizer.pad_token_id is not None:
            pad_token_id = tokenizer.pad_token_id
        else:
            pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 151643
    generation_config.pad_token_id = pad_token_id

    stop_token_ids = set()
    if eos_token_id is not None:
        if isinstance(eos_token_id, int): stop_token_ids.add(eos_token_id)
        else: stop_token_ids.update(eos_token_id)
    if generation_config.eos_token_id is not None:
        if isinstance(generation_config.eos_token_id, int): stop_token_ids.add(generation_config.eos_token_id)
        else: stop_token_ids.update(generation_config.eos_token_id)
    stop_token_ids.add(151643)
    stop_token_ids.add(151645)

    # --- Cache Init ---
    cur_len_main = main_input_ids.shape[1]

    # 1) LLM Verify Stream（维护 model_kwargs_main）
    llm_stream = VerifyStream(
        model=model,
        init_cur_len=cur_len_main,
        init_device=main_input_ids.device,
        init_model_kwargs=model_kwargs_main,
        output_device=model.device,
    )

    # 2) Base verifier stream (wo). If assistant_model is shared with the
    # augmented drafter, this still has an independent logical KV/cache stream.
    base_stream = VerifyStream(
        model=assistant_model,
        init_cur_len=cur_len_main,
        init_device=assistant_model.device,
        init_model_kwargs={},   # 保持你原来的 {}
        output_device=model.device,  # ratio 在 model.device 上做
    )

    # 2.5) Verification Policy
    verification_policy = VerificationPolicy(gamma=gamma, eps=1e-10)

    # 2.6) Vocab Aligner
    vocab_aligner = get_vocab_aligner(args)

    # 3) augmented SLM Cache (用于 Draft，对应 Asst Context)
    cur_len_asst = asst_input_ids.shape[1]
    asst_kwargs = assistant_model._get_initial_cache_position(cur_len_asst, assistant_model.device, model_kwargs_asst)

    # --- Generators ---
    gen_with = AssistedCandidateGenerator(
        input_ids=asst_input_ids.to(assistant_model.device),
        assistant_model=assistant_model,
        generation_config=generation_config,
        model_kwargs=asst_kwargs,
        logits_processor=logits_processor,
    )
    gen_with.num_assistant_tokens = getattr(generation_config, "num_assistant_tokens", args.num_assistant_tokens)

    total_drafted_tokens = 0
    total_accepted_tokens = 0
    total_decoding_steps = 0
    initial_main_len = main_input_ids.shape[1]
    start_time = time.perf_counter()

    with torch.inference_mode():
        while True:
            if main_input_ids.shape[1] >= max_length: break

            _print_ctx("MAIN_PREFIX", main_input_ids, tokenizer, args, max_tail=64)

            # ==========================================
            # 1. Single Draft (只用 With 模型生成)
            # ==========================================
            with_candidate_ids, with_candidate_logits = gen_with.get_candidates(asst_input_ids.to(assistant_model.device))

            # 截取 Draft Tokens
            draft_tokens = with_candidate_ids[:, cur_len_asst:]
            if draft_tokens.shape[1] == 0: break
            draft_len = draft_tokens.shape[1]

            total_decoding_steps += 1
            total_drafted_tokens += draft_len

            # ==========================================
            # 2. Dual Verify (LLM + Base SLM)
            # ==========================================
            # Verifier scoring must always run on the target prefix (main_input_ids),
            # never on augmented-prefix tokens from the drafter context.
            verify_input_ids = torch.cat([main_input_ids.to(model.device), draft_tokens.to(model.device)], dim=1)

            llm_outputs, llm_logits_seq = llm_stream.forward_verify(verify_input_ids, draft_len=draft_len)
            base_outputs, base_logits_seq = base_stream.forward_verify(verify_input_ids, draft_len=draft_len)

            # ==========================================
            # 3. Ratio Logic Check
            # ==========================================
            p_llm_val = llm_stream.draft_token_probs(llm_logits_seq, draft_tokens)     # [1, draft_len]
            p_base_val = base_stream.draft_token_probs(base_logits_seq, draft_tokens) # [1, draft_len]

            accept_mask, num_matches = verification_policy.verify(p_llm_val, p_base_val)

            total_accepted_tokens += num_matches

            # ==========================================
            # 4. Fusion & Update
            # ==========================================
            accepted_tokens = draft_tokens[:, :num_matches]

            final_next_token = None
            if num_matches == draft_len:
                tokens_to_append = accepted_tokens
            else:
                logits_llm_t = llm_logits_seq[:, num_matches, :]  # already on model.device
                logits_base_t = base_logits_seq[:, num_matches, :].to(model.device)
                logits_pers_t = with_candidate_logits[:, num_matches, :].to(model.device)

                # 词表对齐与映射（抽象出去，保持原 pad/truncate 行为不变）
                (logits_base_t, logits_pers_t) = vocab_aligner.align_to_ref(
                    logits_llm_t, logits_base_t, logits_pers_t, device=model.device
                )

                fused_log_probs = fusion.fuse(logits_llm_t, logits_base_t, logits_pers_t)
                fused_token = fused_log_probs.argmax(dim=-1, keepdim=True)

                final_next_token = fused_token
                tokens_to_append = torch.cat([accepted_tokens.to(model.device), fused_token], dim=1)

            # ==========================================
            # 5. Cache Management (Crucial)
            # ==========================================
            main_input_ids = torch.cat([main_input_ids.to(model.device), tokens_to_append.to(model.device)], dim=1)
            asst_input_ids = torch.cat([asst_input_ids.to(model.device), tokens_to_append.to(model.device)], dim=1)

            # Determine cache status
            if final_next_token is not None:
                # Fused: 'tokens_to_append' includes the fused token (last one).
                # We haven't processed the fused token yet.
                # Valid cache = Context + Accepted (num_matches)
                n_accepted = num_matches
                keep_len = main_input_ids.shape[1] - 1
            else:
                # All Match: 'tokens_to_append' is just accepted tokens.
                # We verified them all.
                # Valid cache = Context + Accepted (num_matches)
                n_accepted = num_matches
                keep_len = main_input_ids.shape[1]

            llm_stream.update_after_append(llm_outputs, keep_len=keep_len, n_new_tokens=n_accepted)
            base_stream.update_after_append(base_outputs, keep_len=keep_len, n_new_tokens=n_accepted)

            # Reset Drafter (augmented)
            cur_len_asst = asst_input_ids.shape[1]
            asst_kwargs = assistant_model._get_initial_cache_position(cur_len_asst, assistant_model.device, {})
            gen_with = AssistedCandidateGenerator(
                input_ids=asst_input_ids.to(assistant_model.device),
                assistant_model=assistant_model,
                generation_config=generation_config,
                model_kwargs=asst_kwargs,
                logits_processor=logits_processor,
            )
            gen_with.num_assistant_tokens = getattr(generation_config, "num_assistant_tokens", args.num_assistant_tokens)

            # --- Stop Checks ---
            last_token = main_input_ids[0, -1].item()
            if last_token in stop_token_ids:
                break

    end_time = time.perf_counter()
    total_latency_sec = end_time - start_time
    total_new_tokens = main_input_ids.shape[1] - initial_main_len

    metrics = {
        "total_new_tokens": total_new_tokens,
        "total_decoding_steps": total_decoding_steps,
        "avg_tokens_per_step": total_new_tokens / total_decoding_steps if total_decoding_steps > 0 else 0,
        "acceptance_rate": (total_accepted_tokens / total_drafted_tokens * 100) if total_drafted_tokens > 0 else 0,
        "total_latency_sec": total_latency_sec,
        "tokens_per_sec": total_new_tokens / total_latency_sec if total_latency_sec > 0 else 0,
        "total_drafted_tokens": total_drafted_tokens,
        "total_accepted_tokens": total_accepted_tokens,
    }

    return main_input_ids, metrics


# =========================================================================
# 3. Demo / usage
# =========================================================================
if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

    p = argparse.ArgumentParser()

    # --- fusion / costeer args ---
    p.add_argument("--fusion_method", type=str, default="costeer")  # costeer / linear
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--beta", type=float, default=1.5)
    p.add_argument("--player_lambda", type=float, default=2.0)
    p.add_argument("--eta", type=float, default=10.0)
    p.add_argument("--fusion_coeff", type=float, default=1.0)  # for linear

    # --- speculate/verify args ---
    p.add_argument("--gamma", type=float, default=0.6)
    p.add_argument("--num_assistant_tokens", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=64)   # used to set max_length
    p.add_argument("--debug", action="store_true", default=False)

    # --- model/device ---
    p.add_argument("--llm_model_name", type=str, default="/home/l84349578/models/Qwen/Qwen3-1.7B")
    p.add_argument("--slm_model_name", type=str, default="/home/l84349578/models/Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", type=str, default="cuda")  # 可改成 "npu:0" / "cuda" / "cpu"

    # --- chat template ---
    p.add_argument("--add_generation_prompt", type=int, default=1)

    args = p.parse_args()

    # ------------------------
    # 1) load tokenizer & models
    # ------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.llm_model_name, trust_remote_code=True).to(args.device).eval()

    assistant_tokenizer = AutoTokenizer.from_pretrained(args.slm_model_name, trust_remote_code=True)
    assistant_model = AutoModelForCausalLM.from_pretrained(args.slm_model_name, trust_remote_code=True).to(args.device).eval()
    # Disable heuristic adjustments (keep count fixed) 
    assistant_model.generation_config.num_assistant_tokens_schedule = "constant" 
    assistant_model.generation_config.assistant_confidence_threshold = 0.0 
    assistant_model.generation_config.num_assistant_tokens = args.num_assistant_tokens
    # ------------------------
    # 2) generation config
    # ------------------------
    gen_cfg = GenerationConfig(
        do_sample=False,
        num_assistant_tokens=args.num_assistant_tokens,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # ------------------------
    # 3) build fusion strategy
    # ------------------------
    fusion = get_fusion_strategy(args)

    # ------------------------
    # 4) build inputs (main vs augmented)
    # ------------------------
    msg_main = [{"role": "user", "content": "What is Transformer?"}]
    msg_asst = [{"role": "user", "content": "I am a toddler (example of a super long context). What is Transformer?"}]

    in_main = messages_to_input_ids(
        tokenizer, msg_main, device=args.device, add_generation_prompt=bool(args.add_generation_prompt)
    )
    in_asst = messages_to_input_ids(
        assistant_tokenizer, msg_asst, device=args.device, add_generation_prompt=bool(args.add_generation_prompt)
    )

    main_input_ids = in_main["input_ids"]
    asst_input_ids = in_asst["input_ids"]

    gen_cfg.max_length = int(main_input_ids.shape[1] + args.max_new_tokens)

    model_kwargs_main = {k: v for k, v in in_main.items() if k != "input_ids"}
    model_kwargs_asst = {k: v for k, v in in_asst.items() if k != "input_ids"}

    # ------------------------
    # 5) run decoding
    # ------------------------
    out_ids, metrics = _context_assisted_decoding(
        model=model,
        assistant_model=assistant_model,
        main_input_ids=main_input_ids,
        asst_input_ids=asst_input_ids,
        generation_config=gen_cfg,
        logits_processor=LogitsProcessorList(),
        max_length=gen_cfg.max_length,
        pad_token_id=gen_cfg.pad_token_id,
        eos_token_id=gen_cfg.eos_token_id,
        model_kwargs_main=model_kwargs_main,
        model_kwargs_asst=model_kwargs_asst,
        fusion=fusion,
        args=args,
        tokenizer=tokenizer,
    )

    # ------------------------
    # 6) print
    # ------------------------
    print("===== Metrics =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\n===== Decoded Output =====")
    print(tokenizer.decode(out_ids[0], skip_special_tokens=True))

