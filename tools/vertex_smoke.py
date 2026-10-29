from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apex.env import load_project_env

ALLOWED_MODEL = "gemini-2.5-flash"
FLASH_STANDARD_INPUT_USD_PER_MILLION = 0.30
FLASH_STANDARD_OUTPUT_USD_PER_MILLION = 2.50


def _get_attr(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(
            f"Missing {name}. Refusing to run live Vertex validation without explicit configuration."
        )
    return value


def resolve_project_id() -> tuple[str, str]:
    project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
    if project:
        return project, "env"

    try:
        import google.auth

        _, project = google.auth.default()
    except Exception as exc:
        raise SystemExit(
            "Missing GOOGLE_CLOUD_PROJECT and ADC did not provide a project id. "
            "Set GOOGLE_CLOUD_PROJECT in the local ignored .env file."
        ) from exc

    if not project:
        raise SystemExit(
            "ADC loaded, but no project id was available. Set GOOGLE_CLOUD_PROJECT "
            "in the local ignored .env file."
        )
    return project, "adc"


def percentile(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    idx = max(0, min(len(vals) - 1, int(p * len(vals)) - 1))
    return round(sorted(vals)[idx], 2)


def summarize_values(vals: list[float]) -> dict[str, float | None]:
    return {
        "min": round(min(vals), 2) if vals else None,
        "p50": round(statistics.median(vals), 2) if vals else None,
        "p95": percentile(vals, 0.95),
        "max": round(max(vals), 2) if vals else None,
    }


def usage_field(resp: Any, name: str) -> int | None:
    usage = _get_attr(resp, "usage_metadata")
    value = _get_attr(usage, name)
    return int(value) if isinstance(value, int | float) else None


def estimate_flash_cost_usd(prompt_tokens: int | None, output_tokens: int | None) -> float | None:
    if prompt_tokens is None or output_tokens is None:
        return None
    return round(
        (prompt_tokens / 1_000_000.0) * FLASH_STANDARD_INPUT_USD_PER_MILLION
        + (output_tokens / 1_000_000.0) * FLASH_STANDARD_OUTPUT_USD_PER_MILLION,
        8,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--prompt", default="Explain local-first LLM overflow in one concise paragraph.")
    ap.add_argument("--public-out", default=os.getenv("APEX_VERTEX_PUBLIC_OUT", "analysis/vertex_validation_summary.json"))
    ap.add_argument("--private-out", default=os.getenv("APEX_VERTEX_PRIVATE_OUT", "analysis/private/vertex_validation_raw.json"))
    ap.add_argument("--max-output-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--thinking-budget", type=int, default=0)
    ap.add_argument(
        "--max-estimated-cost-usd",
        type=float,
        default=10.0,
        help="Stop before the next request if the estimated Gemini cost reaches this cap.",
    )
    args = ap.parse_args()

    load_project_env(Path(__file__).resolve(), Path.cwd())

    credentials_path = require_env("GOOGLE_APPLICATION_CREDENTIALS")
    if not Path(credentials_path).is_file():
        raise SystemExit("GOOGLE_APPLICATION_CREDENTIALS is set, but the credential file is not readable.")

    location = require_env("GOOGLE_CLOUD_LOCATION")
    model = require_env("APEX_VERTEX_MODEL")
    if model != ALLOWED_MODEL:
        raise SystemExit(f"Refusing live validation with {model!r}. Set APEX_VERTEX_MODEL={ALLOWED_MODEL}.")

    project, project_source = resolve_project_id()

    from google import genai
    from google.genai import types

    client = genai.Client(vertexai=True, project=project, location=location)
    config = types.GenerateContentConfig(
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        thinking_config=types.ThinkingConfig(thinking_budget=args.thinking_budget),
    )
    rows = []
    spent_estimate = 0.0
    stopped_reason = None
    for i in range(args.n):
        if spent_estimate >= args.max_estimated_cost_usd:
            stopped_reason = "estimated_cost_cap_reached"
            break
        t0 = time.perf_counter()
        ok, err, text_len = True, None, 0
        resp = None
        try:
            resp = client.models.generate_content(model=model, contents=args.prompt, config=config)
            text = getattr(resp, "text", "") or ""
            text_len = len(text)
        except Exception as exc:
            ok, err = False, str(exc)
        t1 = time.perf_counter()
        prompt_tokens = usage_field(resp, "prompt_token_count")
        output_tokens = usage_field(resp, "candidates_token_count")
        thoughts_tokens = usage_field(resp, "thoughts_token_count")
        rows.append({
            "idx": i,
            "ok": ok,
            "error": err,
            "wall_ms": round((t1 - t0) * 1000.0, 2),
            "text_len_chars": text_len,
            "prompt_token_count": prompt_tokens,
            "output_token_count": output_tokens,
            "thoughts_token_count": thoughts_tokens,
            "total_token_count": usage_field(resp, "total_token_count"),
            "estimated_cost_usd": estimate_flash_cost_usd(prompt_tokens, output_tokens),
        })
        if rows[-1]["estimated_cost_usd"] is not None:
            spent_estimate += rows[-1]["estimated_cost_usd"]

    private_out = Path(args.private_out)
    private_out.parent.mkdir(parents=True, exist_ok=True)
    private_payload = {
        "model": model,
        "location": location,
        "n": args.n,
        "max_output_tokens": args.max_output_tokens,
        "temperature": args.temperature,
        "thinking_budget": args.thinking_budget,
        "created_unix": time.time(),
        "rows": rows,
    }
    private_out.write_text(json.dumps(private_payload, indent=2), encoding="utf-8")

    vals = [r["wall_ms"] for r in rows if r["ok"]]
    costs = [r["estimated_cost_usd"] for r in rows if r["ok"] and r["estimated_cost_usd"] is not None]
    summary = {
        "model": model,
        "location": location,
        "n": args.n,
        "n_attempted": len(rows),
        "ok": sum(1 for r in rows if r["ok"]),
        "stopped_reason": stopped_reason,
        "project_configured": True,
        "project_source": project_source,
        "max_estimated_cost_usd": args.max_estimated_cost_usd,
        "latency_wall_ms": summarize_values(vals),
        "estimated_cost_usd_total": round(sum(costs), 8) if costs else None,
        "estimated_cost_note": (
            "Estimated from token counts for bounded validation only; not a billing claim. "
            f"Gemini 2.5 Flash standard text rates assumed: input ${FLASH_STANDARD_INPUT_USD_PER_MILLION}/1M, "
            f"output ${FLASH_STANDARD_OUTPUT_USD_PER_MILLION}/1M."
        ),
        "private_rows_file": "analysis/private/<redacted>",
        "created_unix": time.time(),
    }
    public_out = Path(args.public_out)
    public_out.parent.mkdir(parents=True, exist_ok=True)
    public_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote sanitized summary to {public_out}")
    print("Wrote private provider rows to ignored analysis/private output.")


if __name__ == "__main__":
    main()
