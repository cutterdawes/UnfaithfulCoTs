from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class GenerationResult:
    prompt: str
    text: str
    tokens: List[int]
    features: torch.Tensor  # 1D tensor feature vector


def _mean_last_hidden(hs: List[torch.Tensor], token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean-pool the last layer hidden states over selected tokens.

    hs: list of hidden states per layer; we use the last item shape (B, T, H).
    token_mask: optional boolean mask (B, T) to restrict to CoT tokens.
    Returns (H,) pooled vector.
    """
    last = hs[-1]  # (B,T,H)
    if token_mask is not None:
        masked = last.masked_select(token_mask.unsqueeze(-1)).view(-1, last.size(-1))
    else:
        masked = last.reshape(-1, last.size(-1))
    return masked.mean(dim=0)


class HFModel:
    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        device_map: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
        # Prefer CUDA if available; avoid defaulting to MPS due to stability issues on some models
        preferred_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device or preferred_device
        if self.device == "mps":
            # MPS can be unstable with some reasoning models. Use CPU unless explicitly overridden.
            print("[info] MPS selected; if you encounter MPS errors, try --device cpu.")

        if dtype is None:
            if self.device == "cuda" and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            elif self.device == "cuda":
                dtype = torch.float16
            else:
                dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with optional device_map. Avoid auto mapping by default on CPU/MPS to prevent offloading/meta params.
        kwargs = {"torch_dtype": dtype, "trust_remote_code": trust_remote_code}
        if device_map is not None:
            kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

        try:
            # If device_map wasn't used, move the whole model to the selected device
            if device_map is None:
                self.model.to(self.device)
        except Exception as e:
            # Fallback to CPU if movement fails (e.g., limited MPS/CUDA support)
            print(f"[warn] Could not move model to {self.device}: {e}. Using CPU.")
            self.device = "cpu"
            self.model.to(self.device)

        self.model.eval()

    @torch.no_grad()
    def generate_with_features(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.7) -> GenerationResult:
        tok = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate a response
        gen_out = self.model.generate(
            **tok,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )
        full_ids = gen_out.sequences  # (1, T_total)
        text = self.tokenizer.decode(full_ids[0], skip_special_tokens=True)

        # Recompute hidden states on the full sequence to extract features
        out = self.model(full_ids, output_hidden_states=True, use_cache=False)
        hidden_states: Tuple[torch.Tensor, ...] = out.hidden_states  # len=L+1

        # Mask only the generated continuation tokens (exclude prompt)
        gen_len = full_ids.size(1) - tok.input_ids.size(1)
        mask = torch.zeros_like(full_ids, dtype=torch.bool)
        if gen_len > 0:
            mask[:, -gen_len:] = True
        feat = _mean_last_hidden(list(hidden_states), token_mask=mask)

        return GenerationResult(
            prompt=prompt,
            text=text,
            tokens=full_ids[0].tolist(),
            features=feat.detach().cpu(),
        )

