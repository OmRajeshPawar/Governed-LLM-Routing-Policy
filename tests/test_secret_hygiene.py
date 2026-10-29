from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_secret_hygiene_module():
    script = Path(__file__).resolve().parents[1] / "tools" / "secret_hygiene.py"
    spec = importlib.util.spec_from_file_location("secret_hygiene", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_publishable_files_do_not_expose_credentials() -> None:
    module = load_secret_hygiene_module()
    root = Path(__file__).resolve().parents[2]
    findings = module.scan_workspace(root)
    assert findings == [], "Secret hygiene findings: " + ", ".join(
        finding.public_message(root) for finding in findings
    )
