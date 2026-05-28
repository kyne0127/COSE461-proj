"""B5 baseline: re-query with a general multimodal LLM consistency judge.

B5 gives the initial image and checkpoint image to a GPT-4V-class model and
asks it to directly choose CONTINUE / ASK / STOP. It intentionally avoids
structured G0 memory, coordinate matching, and AmbRes structured outputs.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
import sys
from pathlib import Path
from typing import Any, Callable

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from baselines.common import BaselineResult
from monitoring.consistency_monitor import Decision


METHOD_NAME = "B5_LLM_CONSISTENCY_JUDGE"
LLMClient = Callable[..., Any]


def build_prompt(task_description: str, checkpoint: str) -> str:
    """Build the fixed B5 consistency-judge prompt."""
    return f"""
Task: {task_description}
Checkpoint: {checkpoint}

You are given two images:
1. Initial scene before the robot started.
2. Current checkpoint scene during execution.

Is the original grounding still valid?
- Is the target still identifiable and unambiguous?
- Is the destination still available and unambiguous?
- If the target is missing before pick, answer STOP.
- If the target or destination is ambiguous, answer ASK.
- If the original grounding is still valid, answer CONTINUE.

Answer only in this JSON format:
{{
  "decision": "CONTINUE" | "ASK" | "STOP",
  "reason": "short explanation"
}}
""".strip()


def _strip_code_fence(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def parse_llm_decision(output: Any) -> tuple[Decision, str, dict[str, Any]]:
    """Parse a GPT-style JSON decision response."""
    if isinstance(output, dict):
        data = dict(output)
    elif isinstance(output, str):
        text = _strip_code_fence(output)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM output is not valid JSON: {output!r}") from exc
    else:
        raise TypeError(f"LLM output must be dict or str, got {type(output).__name__}")

    raw_decision = str(data.get("decision", "")).strip().upper()
    try:
        decision = Decision(raw_decision)
    except ValueError as exc:
        raise ValueError(f"Invalid LLM decision {raw_decision!r}") from exc

    reason = str(data.get("reason", "")).strip()
    return decision, reason, data


def _image_data_url(path: str | Path) -> str:
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _default_openai_client(
    *,
    prompt: str,
    image_paths: list[str | Path],
    model: str,
    temperature: float,
) -> str:
    """Call OpenAI Responses API if no explicit test/client adapter is supplied."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "B5 default client requires the openai package. "
            "Install it or pass llm_client=... to run_b5_llm_judge()."
        ) from exc

    client = OpenAI()
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for image_path in image_paths:
        content.append({"type": "input_image", "image_url": _image_data_url(image_path)})

    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        temperature=temperature,
    )
    return response.output_text


def run_b5_llm_judge(
    initial_img: str | Path,
    checkpoint_img: str | Path,
    task_description: str,
    *,
    checkpoint: str = "C1",
    llm_client: LLMClient | None = None,
    model: str = "gpt-4o",
    temperature: float = 0.0,
) -> BaselineResult:
    """Run B5 by asking a general multimodal LLM to judge consistency.

    Args:
        initial_img: Initial t0 image.
        checkpoint_img: Current checkpoint image.
        task_description: Natural-language task instruction.
        checkpoint: Checkpoint label stored in metadata ("C1" or "C2").
        llm_client: Injectable client for tests or alternate providers.
                    Called with prompt, image_paths, model, temperature.
        model: Model name forwarded to the client.
        temperature: Temperature forwarded to the client.

    Returns:
        BaselineResult with the parsed LLM decision.
    """
    if checkpoint not in ("C1", "C2"):
        raise ValueError(f"checkpoint must be 'C1' or 'C2', got {checkpoint!r}")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")

    prompt = build_prompt(task_description, checkpoint)
    client = llm_client or _default_openai_client
    image_paths = [initial_img, checkpoint_img]
    output = client(
        prompt=prompt,
        image_paths=image_paths,
        model=model,
        temperature=temperature,
    )
    decision, reason, parsed = parse_llm_decision(output)

    return BaselineResult(
        method=METHOD_NAME,
        decision=decision,
        reason=reason,
        question=reason if decision == Decision.ASK else "",
        raw_output={"llm_output": parsed},
        metadata={
            "checkpoint": checkpoint,
            "model": model,
            "temperature": temperature,
            "uses_checkpoint": True,
            "stores_g0": False,
            "uses_coord": False,
            "uses_taxonomy": False,
            "initial_img": str(initial_img),
            "checkpoint_img": str(checkpoint_img),
            "prompt": prompt,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B5 LLM consistency judge baseline.")
    parser.add_argument("initial_img")
    parser.add_argument("checkpoint_img")
    parser.add_argument("task_description")
    parser.add_argument("--checkpoint", default="C1", choices=["C1", "C2"])
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    result = run_b5_llm_judge(
        args.initial_img,
        args.checkpoint_img,
        args.task_description,
        checkpoint=args.checkpoint,
        model=args.model,
        temperature=args.temperature,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
