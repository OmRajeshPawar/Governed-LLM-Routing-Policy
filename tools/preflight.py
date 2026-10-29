from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OK, WARN, BAD = "PASS", "WARN", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    tag = {OK: "  ok  ", WARN: " warn ", BAD: " FAIL "}[status]
    print(f"[{tag}] {name}" + (f"  --  {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def check_python() -> None:
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) == (3, 12):
        check("Python 3.12", OK, ver)
    elif (v.major, v.minor) == (3, 13):
        check("Python version", BAD, f"{ver} - pandas/numpy disagree at the C ABI on 3.13; use 3.12")
    else:
        check("Python version", WARN, f"{ver} - project is verified on 3.12")


def check_imports() -> None:
    missing = []
    for mod in ("numpy", "pandas", "sklearn", "scipy", "matplotlib",
                "pyarrow", "yaml", "pydantic", "ollama", "psutil"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if missing:
        check("Dependencies importable", BAD, "missing: " + ", ".join(missing))
    else:
        check("Dependencies importable", OK)


def check_ollama() -> None:
    try:
        import ollama
        models = {m.model for m in ollama.Client().list().models}
    except Exception as exc:
        check("Ollama reachable", BAD, f"{type(exc).__name__}: {exc}")
        return
    check("Ollama reachable", OK, f"{len(models)} models installed")
    for needed in ("llama3.1:8b", "gemma4:e2b"):
        if needed in models:
            check(f"Model {needed}", OK)
        else:
            check(f"Model {needed}", BAD, "not pulled - the live run cannot start")


def check_gpu() -> None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True, timeout=20).strip().splitlines()[0]
        temp, used, total = [int(x.strip()) for x in out.split(",")]
    except Exception:
        check("GPU telemetry", WARN, "nvidia-smi unavailable - cooldown gate will no-op")
        return
    detail = f"{temp}C, {used}/{total} MB used"
    if temp >= 58:
        check("GPU cool enough", WARN, detail + " - run_all would wait for cooldown")
    else:
        check("GPU cool enough", OK, detail)


def check_config() -> None:
    try:
        from apex.core import load_config
        cfg = load_config()
    except Exception as exc:
        check("config.yaml loads", BAD, f"{type(exc).__name__}: {exc}")
        return
    check("config.yaml loads", OK)
    if cfg.governor.enabled:
        check("Governor enabled", OK)
    else:
        check("Governor enabled", BAD, "config.yaml governor.enabled is false; the runner will refuse")
    if cfg.providers.cloud.mode == "calibrated":
        check("Cloud tier is offline model", OK, "no provider credentials needed")
    else:
        check("Cloud tier is offline model", WARN, f"mode={cfg.providers.cloud.mode}")


def check_governor_builds() -> None:
    try:
        from apex.core import load_config
        from apex.policies import make_guarded_governor
        gov = make_guarded_governor(load_config(), episode_requests=100,
                                    frozen_cloud_p95_ms=1000.0)
        regimes = set(gov.artifact.regime_map.values())
    except Exception as exc:
        check("Governor builds from frozen artifact", BAD, f"{type(exc).__name__}: {exc}")
        return
    check("Governor builds from frozen artifact", OK, gov.artifact.artifact_id)
    if regimes == {"tail_aware"}:
        check("Frozen artifact still tail-aware everywhere", OK)
    else:
        check("Frozen artifact still tail-aware everywhere", WARN, f"regimes -> {sorted(regimes)}")


def check_headline_numbers() -> None:
    path = ROOT / "analysis/summary_metrics.json"
    if not path.exists():
        check("Headline results present", BAD, "analysis/summary_metrics.json missing")
        return
    d = json.loads(path.read_text(encoding="utf-8"))
    rows, files = d.get("n_rows"), len(d.get("shadow_files", []))
    if rows == 4500 and files == 15:
        check("Results built from the frozen corpus", OK, f"{rows} rows, {files} files")
    else:
        check("Results built from the frozen corpus", BAD,
              f"{rows} rows / {files} files - a demo run has polluted the table. "
              "Rebuild with APEX_SHADOW_GLOB=shadow_*_bursty_n300_s*.jsonl")
    rq0 = d.get("rq0", {})
    tail = round(rq0.get("tail_aware_sla_violation_rate", 0) * 100, 1)
    ml = round(rq0.get("ml_dual_risk_sla_violation_rate", 0) * 100, 1)
    if (tail, ml) == (6.5, 6.9):
        check("RQ0 verdict unchanged", OK, f"tail-aware {tail}% vs ML dual-risk {ml}%")
    else:
        check("RQ0 verdict unchanged", BAD,
              f"tail-aware {tail}% vs ML dual-risk {ml}% - expected 6.5 / 6.9")
    table = ROOT / "analysis/results_table.md"
    if table.exists() and "| always_local | 0 | 0 | 35.0 |" in table.read_text(encoding="utf-8"):
        check("Results table shows 35.0% always-local", OK)
    else:
        check("Results table shows 35.0% always-local", BAD, "table does not match the shipped results")


def check_files() -> None:
    wanted = {
        "governance/agreement_v4.yaml": "the routing agreement",
        "docs/architecture.md": "architecture writeup",
        "tools/demo_governor.py": "the guard walkthrough",
        "ml/artifacts/governor_v4.json": "the promoted policy artifact",
        "ml/artifacts/ttft_model.joblib": "trained risk model",
        "analysis/figures/latency_cdf.png": "figures rendered",
    }
    missing = [f"{p} ({why})" for p, why in wanted.items() if not (ROOT / p).exists()]
    if missing:
        check("Demo files present", BAD, "; ".join(missing))
    else:
        check("Demo files present", OK, f"{len(wanted)} checked")


def check_no_stray_venv() -> None:
    if (ROOT / ".venv").exists():
        check("No broken venv in the tree", WARN,
              "a stale .venv exists - make sure it is not the one you demo from")
    else:
        check("No broken venv in the tree", OK)


def run_cmd(name: str, args: list[str], expect: str | None = None,
            timeout: int = 300) -> None:
    t0 = time.perf_counter()
    try:
        r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        check(name, BAD, f"timed out after {timeout}s")
        return
    dt = time.perf_counter() - t0
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        tail = out.strip().splitlines()[-1] if out.strip() else "no output"
        check(name, BAD, f"exit {r.returncode}: {tail[:120]}")
        return
    if expect and expect not in out:
        check(name, BAD, f"expected {expect!r} in output")
        return
    check(name, OK, f"{dt:.1f}s")


def check_demo_governor() -> None:
    run_cmd("tools/demo_governor.py runs", ["tools/demo_governor.py"],
            expect="budget_denied", timeout=120)


def check_tests() -> None:
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT,
                       capture_output=True, text=True, timeout=900)
    dt = time.perf_counter() - t0
    out = (r.stdout or "") + (r.stderr or "")
    last = [ln for ln in out.strip().splitlines() if ln.strip()]
    summary = last[-1] if last else "no output"
    if r.returncode == 0 and "passed" in summary:
        check("Test suite", OK, f"{summary.strip()} in {dt:.1f}s")
    else:
        check("Test suite", BAD, summary.strip()[:140])


def check_gates() -> None:
    r = subprocess.run([sys.executable, "analysis/validity_gates.py"], cwd=ROOT,
                       capture_output=True, text=True, timeout=600)
    out = (r.stdout or "").strip()
    try:
        d = json.loads(out)
    except Exception:
        check("Validity gates report", BAD, out[:120] or "no JSON output")
        return
    if d.get("local") == "fail" and d.get("cloud") == "fail":
        check("Validity gates report", OK, "local fail, cloud fail - matches the docs")
    else:
        check("Validity gates report", BAD,
              f"local={d.get('local')} cloud={d.get('cloud')} - docs say both fail")


def check_confirmatory() -> None:
    r = subprocess.run([sys.executable, "analysis/confirmatory_statistics.py"], cwd=ROOT,
                       capture_output=True, text=True, timeout=600)
    out = (r.stdout or "")
    if '"superiority_claim_allowed": false' in out:
        check("Confirmatory analysis refuses to claim", OK, "not_run, as designed")
    else:
        check("Confirmatory analysis refuses to claim", BAD, "unexpected output")


def check_no_credentials_needed() -> None:
    leaked = [v for v in ("GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_API_KEY")
              if os.getenv(v)]
    if leaked:
        check("No cloud credentials in the demo path", WARN,
              "set in the environment: " + ", ".join(leaked) + " (unused, but unset them to be safe)")
    else:
        check("No cloud credentials in the demo path", OK, "fully local")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-tests", action="store_true", help="skip pytest and the analysis scripts")
    args = ap.parse_args()

    print("APEX-LLM pre-demo check")
    print(f"repo: {ROOT}")
    print(f"python: {sys.executable}")

    section("Environment")
    check_python()
    check_imports()
    check_ollama()
    check_gpu()
    check_no_credentials_needed()

    section("Project integrity")
    check_config()
    check_governor_builds()
    check_headline_numbers()
    check_files()
    check_no_stray_venv()

    section("Things you will actually run")
    check_demo_governor()
    if not args.skip_tests:
        check_tests()
        check_gates()
        check_confirmatory()
    else:
        print("  (skipped: pytest, validity gates, confirmatory analysis)")

    fails = [r for r in results if r[1] == BAD]
    warns = [r for r in results if r[1] == WARN]
    section("Verdict")
    if fails:
        print(f"NOT READY - {len(fails)} check(s) failed:")
        for name, _, detail in fails:
            print(f"  - {name}: {detail}")
        return 1
    print(f"READY. {len(results)} checks passed" + (f", {len(warns)} warning(s)." if warns else "."))
    for name, _, detail in warns:
        print(f"  warn  {name}: {detail}")
    print("\nNext: run")
    print("  python tools/demo_governor.py")
    print("  python -m apex.runner --mode replay --policy guarded_governor --pattern bursty --n 24 --seed 42")
    return 0


if __name__ == "__main__":
    sys.exit(main())
