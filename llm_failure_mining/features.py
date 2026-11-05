"""Feature engineering: describe each prompt, model and response with simple columns.

Three groups of features are produced:

* Prompt features: problem type, instruction style, stated numbers, wording.
* Model features: model family, approximate size, base vs. chat model.
* Response features: length, answer marker, and a rule-based error type.

Prompt and model features are known *before* a model answers, so only they
are used to predict failure. Response features describe *how* a model failed
and are used for clustering, rules and error analysis.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# Instruction styles, checked in this order (first match wins).
# Each entry: (style name, list of lowercase marker phrases).
STYLE_RULES: list[tuple[str, list[str]]] = [
    ("base_completion", ["### problem:"]),
    ("persona_scientist", ["scientist"]),
    ("gender_clarified", ["identify themselves as"]),
    ("confidence_request", ["level of confidence"]),
    ("think_step_by_step", ["step by step", "think carefully", "double check"]),
    ("restricted_format", ["do not output any text", "without any further text", "use only the following format"]),
    ("standard", ["solve this problem", "output your answer", "provide final answer", "output your final answer"]),
]

# Publicly reported parameter counts (billions) for models whose name has no size.
KNOWN_SIZES_B = {"phi-2": 2.7, "bloom": 176.0, "dbrx-instruct": 132.0, "snowflake-arctic-instruct": 480.0}

BASE_MODELS = {
    "bloom", "llama-2-7b", "llama-2-13b", "llama-2-70b", "mistral-7b-v0.1",
    "mixtral-8x7b-v0.1", "mixtral-8x22b", "phi-2", "codellama-34b-python",
}

FAMILY_RULES: list[tuple[str, str]] = [
    ("codellama", "codellama"), ("wizardcoder", "wizard"), ("wizardlm", "wizard"),
    ("hermes", "llama_finetune"), ("llama", "llama"), ("gpt", "gpt"), ("claude", "claude"),
    ("qwen", "qwen"), ("mixtral", "mistral"), ("mistral", "mistral"), ("gemma", "gemma"),
    ("gemini", "gemini"), ("command", "cohere"), ("dbrx", "dbrx"), ("deepseek", "deepseek"),
    ("yi-", "yi"), ("olmo", "olmo"), ("phi", "phi"), ("bloom", "bloom"), ("reka", "reka"),
    ("snowflake", "snowflake"),
]


# ── Prompt features ─────────────────────────────────────────────

def problem_text(prompt: str) -> str:
    """Return only the problem statement, without the answer-format instruction.

    The problem ends at the first question mark; everything after it is
    instruction text such as 'Provide the final answer in the form ### Answer:'.
    """
    text = prompt.replace("### Problem:", "").strip()
    end = text.find("?")
    return text[: end + 1] if end != -1 else text


def stated_numbers(prompt: str) -> list[int]:
    """Numbers mentioned in the problem statement, as digits or words.

    'a sister' counts as 1, so 'she also has a sister' yields 1.
    """
    text = problem_text(prompt).lower()
    numbers = [int(n) for n in re.findall(r"\b\d+\b", text)]
    numbers += [WORD_NUMBERS[w] for w in re.findall(r"\b(" + "|".join(WORD_NUMBERS) + r")\b", text)]
    numbers += [1] * len(re.findall(r"\ba (?:sister|brother|son|daughter)\b", text))
    return numbers


def instruction_style(prompt: str) -> str:
    lower = prompt.lower()
    for style, markers in STYLE_RULES:
        if any(marker in lower for marker in markers):
            return style
    if prompt.strip().endswith("?"):
        return "bare_question"
    return "other"


def prompt_features(prompt: str) -> dict:
    lower = prompt.lower()
    problem = problem_text(prompt).lower()
    sis = problem.find("sister")
    bro = problem.find("brother")
    if sis == -1 or bro == -1:
        order = "n/a"
    else:
        order = "sisters_first" if sis < bro else "brothers_first"
    return {
        "problem_type": "aiw_plus_cousins" if "cousin" in lower else "aiw_siblings",
        "prompt_style": instruction_style(prompt),
        "subject_name": problem.split()[0].capitalize() if problem else "",
        "numbers_as_words": bool(re.search(r"\b(" + "|".join(WORD_NUMBERS) + r")\b", problem))
        or bool(re.search(r"\ba (?:sister|brother)\b", problem)),
        "mention_order": order,
        "n_stated_numbers": len(stated_numbers(prompt)),
        "prompt_words": len(prompt.split()),
    }


# ── Model features ──────────────────────────────────────────────

def model_family(name: str) -> str:
    lower = name.lower()
    for marker, family in FAMILY_RULES:
        if marker in lower:
            return family
    return "other"


def model_size_b(name: str) -> float:
    """Approximate total parameters in billions, or NaN if not public.

    Mixture-of-experts names like '8x22b' are counted as 8 * 22 = 176.
    """
    lower = name.lower()
    if lower in KNOWN_SIZES_B:
        return KNOWN_SIZES_B[lower]
    moe = re.search(r"(\d+)x(\d+(?:\.\d+)?)b", lower)
    if moe:
        return float(moe.group(1)) * float(moe.group(2))
    dense = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)b\b", lower)
    if dense:
        return float(dense.group(1))
    return float("nan")


def size_bucket(size_b: float) -> str:
    if np.isnan(size_b):
        return "unknown_closed"
    if size_b < 10:
        return "small_<10B"
    if size_b < 40:
        return "medium_10-40B"
    if size_b <= 100:
        return "large_40-100B"
    return "xlarge_>100B"


def model_features(name: str) -> dict:
    size = model_size_b(name)
    return {
        "family": model_family(name),
        "size_b": size,
        "size_bucket": size_bucket(size),
        "is_base_model": name.lower() in BASE_MODELS,
        "is_code_model": "code" in name.lower(),
    }


# ── Response features and error typing ──────────────────────────

def error_type(parsed: float | None, right: float, correct: bool, numbers: list[int]) -> str:
    """Classify a response into an interpretable error category.

    The order matters: more specific explanations are checked first.
    For the sibling problems the right answer is (stated sisters + 1),
    because Alice is herself a sister; 'off_by_one_low' therefore means
    the model forgot to count Alice.
    """
    if correct:
        return "correct"
    if parsed is None or np.isnan(parsed):
        return "no_parsable_answer"
    if parsed == right:
        return "right_number_wrong_reasoning"
    if parsed == right - 1:
        return "off_by_one_low"
    if parsed == right + 1:
        return "off_by_one_high"
    if parsed in numbers:
        return "copies_given_number"
    sums = {a + b for i, a in enumerate(numbers) for b in numbers[i + 1:]}
    if parsed in sums:
        return "adds_given_numbers"
    return "other_wrong"


def length_bucket(words: int) -> str:
    if words < 20:
        return "short_<20w"
    if words < 150:
        return "medium_20-150w"
    return "long_>150w"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with all prompt, model and response features."""
    out = df.copy()

    prompt_table = pd.DataFrame([prompt_features(p) for p in out["prompt"].unique()])
    prompt_table.insert(0, "prompt", out["prompt"].unique())
    out = out.merge(prompt_table, on="prompt", how="left")

    model_table = pd.DataFrame([model_features(m) for m in out["model"].unique()])
    model_table.insert(0, "model", out["model"].unique())
    out = out.merge(model_table, on="model", how="left")

    numbers_by_prompt = {p: stated_numbers(p) for p in out["prompt"].unique()}
    out["error_type"] = [
        error_type(pa, ra, c, numbers_by_prompt[p])
        for pa, ra, c, p in zip(out["parsed_answer"], out["right_answer"], out["correct"], out["prompt"])
    ]
    out["response_words"] = out["response"].str.split().str.len().fillna(0).astype(int)
    out["length_bucket"] = out["response_words"].map(length_bucket)
    out["has_answer_marker"] = out["response"].str.contains("### Answer", regex=False)
    out.attrs = dict(df.attrs)  # merges drop attrs; keep data-quality info
    return out
