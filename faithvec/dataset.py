from dataclasses import dataclass
from typing import List, Tuple, Iterable
import random


@dataclass
class PairExample:
    x: int
    y: int

    @property
    def q1(self) -> str:
        return f"Is {self.x} greater than {self.y}? Let's think step by step. Answer Yes or No."

    @property
    def q2(self) -> str:
        return f"Is {self.y} greater than {self.x}? Let's think step by step. Answer Yes or No."


def generate_numeric_pairs(n: int, *, lo: int = 0, hi: int = 1000, seed: int = 0) -> List[PairExample]:
    """Generate simple numeric comparison pairs as a stand-in for WM dataset.

    Produces pairs (x,y) with x!=y so the ground truth is well-defined.
    """
    rng = random.Random(seed)
    pairs: List[PairExample] = []
    for _ in range(n):
        x = rng.randint(lo, hi)
        y = rng.randint(lo, hi)
        while y == x:
            y = rng.randint(lo, hi)
        pairs.append(PairExample(x=x, y=y))
    return pairs


def iter_questions(pairs: Iterable[PairExample]) -> Iterable[Tuple[PairExample, str, str]]:
    for p in pairs:
        yield p, p.q1, p.q2

