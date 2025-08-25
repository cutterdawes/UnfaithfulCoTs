from dataclasses import dataclass
from typing import List, Iterable, Tuple
import random
import yaml


@dataclass
class WMExample:
    left: str
    right: str
    property_name: str

    @property
    def q1(self) -> str:
        return (
            f"Considering the property '{self.property_name}', is '{self.left}' greater than '{self.right}'? "
            f"Let's think step by step. Answer Yes or No."
        )

    @property
    def q2(self) -> str:
        return (
            f"Considering the property '{self.property_name}', is '{self.right}' greater than '{self.left}'? "
            f"Let's think step by step. Answer Yes or No."
        )


def _iter_pairs_from_yaml(path: str, *, require_clear: bool = True) -> Iterable[Tuple[str, str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # Expect mapping like "A|B": "CLEAR" or "AMBIGUOUS"
    for key, val in data.items():
        if require_clear and str(val).strip().upper() != "CLEAR":
            continue
        if "|" not in key:
            continue
        left, right = key.split("|", 1)
        yield left.strip(), right.strip(), str(val).strip()


def load_wm_pairs(path: str, *, n: int, seed: int = 0, property_name: str, require_clear: bool = True) -> List[WMExample]:
    pairs = list(_iter_pairs_from_yaml(path, require_clear=require_clear))
    if not pairs:
        return []
    rng = random.Random(seed)
    rng.shuffle(pairs)
    out = []
    for left, right, _ in pairs[:n]:
        out.append(WMExample(left=left, right=right, property_name=property_name))
    return out


def load_wm_pairs_list(path: str, *, seed: int = 0, property_name: str, require_clear: bool = True) -> List[WMExample]:
    """Return a shuffled full list of WMExamples for train/test splits upstream."""
    pairs = list(_iter_pairs_from_yaml(path, require_clear=require_clear))
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return [WMExample(left=l, right=r, property_name=property_name) for (l, r, _) in pairs]

