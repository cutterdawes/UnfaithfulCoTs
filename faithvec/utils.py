import re
from typing import Optional


YES_PAT = re.compile(r"\b(yes|y|true|correct)\b", re.IGNORECASE)
NO_PAT = re.compile(r"\b(no|n|false|incorrect)\b", re.IGNORECASE)


def parse_yes_no(text: str) -> Optional[bool]:
    """Return True for Yes, False for No, or None if ambiguous.

    Uses simple keyword matching; prioritizes the last decisive token.
    """
    # Look for last decisive match to tolerate reasoning flips
    matches = list(re.finditer(r"\b(yes|no|true|false|correct|incorrect)\b", text, re.IGNORECASE))
    if not matches:
        # Fallback: first token-based scan
        if YES_PAT.search(text) and not NO_PAT.search(text):
            return True
        if NO_PAT.search(text) and not YES_PAT.search(text):
            return False
        return None
    last = matches[-1].group(0).lower()
    if last in {"yes", "y", "true", "correct"}:
        return True
    if last in {"no", "n", "false", "incorrect"}:
        return False
    return None


def dot(a, b):
    return float((a @ b).item())

