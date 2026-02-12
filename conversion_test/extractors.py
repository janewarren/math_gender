"""
Answer extraction and loss calculation.

Pure functions — no API calls, no heavy dependencies (only `re`).
"""

import re
from typing import Optional

# ── Unified Answer Extraction ────────────────────────────────────

_EXTRACTORS = {
    "timezone": lambda s: extract_time_string(s),
    "clothing": lambda s: extract_clothing_size(s),
}


def extract_answer(raw_response: str, domain: str, correct_answer, is_reasoning: bool):
    """Extract the model's answer from a raw response string.

    Returns the parsed answer (float, str, or None).
    """
    if is_reasoning:
        tagged = extract_answer_from_tags(raw_response)
        source = tagged if tagged is not None else raw_response
    else:
        source = raw_response

    answer_type = determine_answer_type(domain, correct_answer)
    extractor = _EXTRACTORS.get(answer_type, lambda s: extract_number(s))
    return extractor(source)


# ── Answer Extraction ────────────────────────────────────────────

def extract_answer_from_tags(response: str) -> Optional[str]:
    """Extract the last <answer>…</answer> value.  Returns None if no tag found."""
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
    return matches[-1].strip() if matches else None


def extract_number(answer: str) -> Optional[float]:
    """Extract a numeric value from *answer*, handling scientific notation."""
    if not answer or answer.startswith("ERROR:"):
        return None
    cleaned = answer.replace("[TRUNCATED]", "").replace(",", "").strip()

    # Scientific notation first
    sci = re.search(r"-?\d+\.?\d*[eE][+-]?\d+", cleaned)
    if sci:
        try:
            return float(sci.group(0))
        except ValueError:
            pass

    # Plain numbers (take the last reasonable one)
    matches = re.findall(r"-?\d+\.?\d*", cleaned)
    for m in reversed(matches):
        try:
            num = float(m)
            if abs(num) > 1e-10 or num == 0:
                return num
        except ValueError:
            continue
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            pass
    return None


def extract_time_string(answer: str) -> Optional[str]:
    """Extract a time string like '1AM' or '3:49PM'."""
    m = re.search(r"(\d{1,2}):(\d{2})\s*(AM|PM)", answer, re.IGNORECASE)
    if m and 1 <= int(m.group(1)) <= 12 and 0 <= int(m.group(2)) <= 59:
        return f"{int(m.group(1))}:{int(m.group(2)):02d}{m.group(3).upper()}"
    m = re.search(r"(\d{1,2})\s*(AM|PM)", answer, re.IGNORECASE)
    if m and 1 <= int(m.group(1)) <= 12:
        return f"{int(m.group(1))}{m.group(2).upper()}"
    return None


def extract_clothing_size(answer: str) -> Optional[str]:
    """Extract a clothing size (e.g. 'M', '32B', '46.5')."""
    if not answer:
        return None
    # Bra size: number + letter  (32B, 70A)
    m = re.search(r"\b(\d{2,3})([A-Z])\b", answer, re.IGNORECASE)
    if m:
        return f"{m.group(1)}{m.group(2).upper()}"
    # Alpha sizes
    m = re.search(r"\b(XS|S|M|L|XL|XXL|XXXL)\b", answer, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Numeric sizes (shoe sizes, pants)
    m = re.search(r"\b(\d{1,3}\.?\d*)\b", answer)
    if m:
        try:
            num = float(m.group(1))
            return str(int(num)) if num == int(num) else f"{num:.1f}".rstrip("0").rstrip(".")
        except ValueError:
            return m.group(1)
    return None


# ── Answer Type Detection ────────────────────────────────────────

def determine_answer_type(domain: str, answer) -> str:
    """Classify the expected answer type based on domain."""
    if "timezone" in domain:
        return "timezone"
    if "clothing_sizes" in domain or "bra_size" in domain:
        return "clothing"
    if isinstance(answer, str) and not answer.replace(".", "").replace("-", "").isdigit():
        return "string"
    return "numeric"


# ── Loss Calculation ─────────────────────────────────────────────

def calculate_loss(
    model_answer,
    correct_answer,
    answer_type: str = "numeric",
    tolerance_percent: float = 0.1,
    tolerance_minutes: float = 1.0,
) -> Optional[float]:
    """Return 0.0 if within tolerance, the error magnitude otherwise, or None on parse failure."""
    if model_answer is None:
        return None

    # ── Numeric ──────────────────────────────────────────────────
    if answer_type == "numeric":
        try:
            m, c = float(model_answer), float(correct_answer)
        except (ValueError, TypeError):
            return None
        if c == 0:
            return 0.0 if abs(m) < 0.001 else abs(m)
        rel = abs((m - c) / c) * 100
        return 0.0 if rel <= tolerance_percent else rel

    # ── Timezone ─────────────────────────────────────────────────
    if answer_type == "timezone":
        def _to_hours(s: str) -> float:
            s = s.strip().upper()
            m2 = re.match(r"(\d{1,2}):(\d{2})(AM|PM)", s)
            if m2:
                h, mi, p = int(m2.group(1)), int(m2.group(2)), m2.group(3)
                if p == "PM" and h != 12:
                    h += 12
                elif p == "AM" and h == 12:
                    h = 0
                return h + mi / 60.0
            m2 = re.match(r"(\d{1,2})(AM|PM)", s)
            if m2:
                h, p = int(m2.group(1)), m2.group(2)
                if p == "PM" and h != 12:
                    h += 12
                elif p == "AM" and h == 12:
                    h = 0
                return float(h)
            raise ValueError(f"Cannot parse time: {s}")
        try:
            diff = abs((_to_hours(str(model_answer)) - _to_hours(str(correct_answer))) * 60)
            if diff > 720:
                diff = 1440 - diff
            return 0.0 if diff <= tolerance_minutes + 1e-9 else round(diff, 4)
        except (ValueError, TypeError):
            return None

    # ── Clothing ─────────────────────────────────────────────────
    if answer_type == "clothing":
        try:
            return 0.0 if abs(float(str(model_answer).strip()) - float(str(correct_answer).strip())) < 0.001 else 1.0
        except (ValueError, TypeError):
            return 0.0 if str(model_answer).strip().upper() == str(correct_answer).strip().upper() else 1.0

    # ── Fallback (exact string match) ────────────────────────────
    return 0.0 if str(model_answer).strip() == str(correct_answer).strip() else 1.0
