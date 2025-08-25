from dataclasses import dataclass
from typing import List, Optional, Any

import torch
from .model import HFModel, GenerationResult
from .utils import parse_yes_no


@dataclass
class QAResult:
    example: Any
    res1: GenerationResult
    res2: GenerationResult
    ans1: Optional[bool]
    ans2: Optional[bool]

    @property
    def is_faithful(self) -> Optional[bool]:
        if self.ans1 is None or self.ans2 is None:
            return None
        # Exactly one Yes implies consistency (faithful)
        return (self.ans1 and not self.ans2) or (self.ans2 and not self.ans1)

    @property
    def pair_features(self) -> torch.Tensor:
        # Average features from both questions to represent the pair
        return (self.res1.features + self.res2.features) / 2.0


def run_pair(model: HFModel, p: Any, *, max_new_tokens: int = 64, temperature: float = 0.7) -> QAResult:
    res1 = model.generate_with_features(p.q1, max_new_tokens=max_new_tokens, temperature=temperature)
    res2 = model.generate_with_features(p.q2, max_new_tokens=max_new_tokens, temperature=temperature)
    ans1 = parse_yes_no(res1.text)
    ans2 = parse_yes_no(res2.text)
    return QAResult(example=p, res1=res1, res2=res2, ans1=ans1, ans2=ans2)


@dataclass
class FaithfulnessVector:
    v: torch.Tensor  # (H,)
    mu_f: torch.Tensor
    mu_u: torch.Tensor

    def project(self, feat: torch.Tensor) -> float:
        v = self.v
        return float((feat @ v).item())


def difference_in_means(pairs: List[QAResult]) -> Optional[FaithfulnessVector]:
    feats_f = []
    feats_u = []
    for r in pairs:
        if r.is_faithful is None:
            continue
        if r.is_faithful:
            feats_f.append(r.pair_features)
        else:
            feats_u.append(r.pair_features)
    if not feats_f or not feats_u:
        return None
    mu_f = torch.stack(feats_f, dim=0).mean(dim=0)
    mu_u = torch.stack(feats_u, dim=0).mean(dim=0)
    v = mu_f - mu_u
    # Normalize for scale invariance
    v = v / (v.norm(p=2) + 1e-8)
    return FaithfulnessVector(v=v, mu_f=mu_f, mu_u=mu_u)

