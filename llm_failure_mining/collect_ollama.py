"""Optional extension: collect new responses from local models through Ollama.

This lets you add your own models (e.g. llama3.2, qwen2.5, mistral) to the
analysis. It sends every distinct AIW prompt to a local Ollama server and
saves the answers in the same JSON format as the AIW data, so the main
pipeline can analyse them unchanged.

Requirements: Ollama installed and running (https://ollama.com), and a model
pulled, e.g. ``ollama pull llama3.2:3b``.

Usage:
    python -m llm_failure_mining.collect_ollama --model llama3.2:3b --trials 5

Note: correctness here is automatic (parsed number == right answer). The AIW
authors additionally checked some responses by hand; we cannot do that
automatically, so labels for collected data are slightly less strict.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .data import load_records

ANSWER_PATTERN = re.compile(r"###\s*Answer:?\s*\**\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
DEFAULT_URL = "http://localhost:11434/api/generate"


def parse_answer(text: str) -> float | None:
    """Return the number after the last '### Answer:' marker, or None."""
    matches = ANSWER_PATTERN.findall(text or "")
    return float(matches[-1]) if matches else None


def distinct_prompts(records: list[dict]) -> list[tuple[str, float]]:
    seen: dict[str, float] = {}
    for rec in records:
        seen.setdefault(rec["prompt"], float(rec["right_answer"]))
    return sorted(seen.items())


def query_ollama(prompt: str, model: str, url: str, temperature: float, timeout: float, retries: int = 2) -> str:
    payload = json.dumps(
        {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}}
    ).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))["response"]
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))  # bounded, growing wait between retries
    raise RuntimeError(f"Ollama request failed after {retries + 1} attempts: {last_error}")


def collect(
    prompts: list[tuple[str, float]], model: str, trials: int, url: str = DEFAULT_URL,
    temperature: float = 0.7, timeout: float = 120.0,
) -> list[dict]:
    rows = []
    total = len(prompts) * trials
    for i, (prompt, right) in enumerate(prompts):
        for t in range(trials):
            response = query_ollama(prompt, model, url, temperature, timeout)
            parsed = parse_answer(response)
            rows.append(
                {
                    "model": f"ollama:{model}",
                    "prompt": prompt,
                    "model_response": response,
                    "parsed_answer": parsed,
                    "right_answer": right,
                    "correct": parsed is not None and parsed == right,
                    "manual_inspection": False,
                }
            )
            print(f"  {len(rows)}/{total}", end="\r", flush=True)
    print()
    return rows


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Ollama model name, e.g. llama3.2:3b")
    parser.add_argument("--trials", type=int, default=5, help="answers per prompt (default 5)")
    parser.add_argument("--prompts-from", type=Path, default=root / "data" / "sample" / "aiw_sample.json")
    parser.add_argument("--max-prompts", type=int, default=None, help="limit prompts for a quick test")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    prompts = distinct_prompts(load_records(args.prompts_from))[: args.max_prompts]
    print(f"Collecting {args.trials} trial(s) for {len(prompts)} prompts from {args.model}")
    try:
        rows = collect(prompts, args.model, args.trials, args.url)
    except RuntimeError as exc:
        print(f"Error: {exc}\nIs Ollama running? Start it and run 'ollama pull {args.model}'.", file=sys.stderr)
        return 1

    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", args.model)
    out = args.output or root / "data" / "collected" / f"ollama_{safe}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    acc = sum(r["correct"] for r in rows) / len(rows)
    print(f"Saved {len(rows)} responses to {out} (accuracy {acc:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
