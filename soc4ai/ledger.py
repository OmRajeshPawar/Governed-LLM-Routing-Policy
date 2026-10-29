"""Persistent spend ledger with a hard USD kill switch.

Reserve-then-commit: the worst-case cost of a call is reserved synchronously
(no await between the check and the reservation), so concurrent asyncio calls
cannot race past the cap. The actual estimated cost is committed afterwards.
Once the switch fires, the ledger is marked halted on disk and refuses every
later call, including in a restarted process.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

INPUT_USD_PER_TOKEN = 0.30 / 1_000_000
OUTPUT_USD_PER_TOKEN = 2.50 / 1_000_000


class BudgetKillSwitch(RuntimeError):
    pass


def estimate_cost(prompt_tokens: int, output_tokens: int) -> float:
    return prompt_tokens * INPUT_USD_PER_TOKEN + output_tokens * OUTPUT_USD_PER_TOKEN


def worst_case_cost(prompt: str, max_output_tokens: int) -> float:
    # Gemini averages ~4 chars/token; assume 2 chars/token (2x margin) plus overhead.
    prompt_tokens = len(prompt) // 2 + 16
    return estimate_cost(prompt_tokens, int(max_output_tokens))


class SpendLedger:

    def __init__(self, path: str | Path, cap_usd: float = 20.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cap = float(cap_usd)
        self._lock = threading.Lock()
        self._reserved: dict[str, float] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            # A reservation that never reconciled (crash) is counted as spent.
            self.committed = float(data.get("committed_usd", 0.0)) + float(data.get("reserved_usd", 0.0))
            self.n_calls = int(data.get("n_calls", 0))
            self.halted = bool(data.get("halted", False))
        else:
            self.committed, self.n_calls, self.halted = 0.0, 0, False
        self._persist()

    @property
    def reserved(self) -> float:
        return sum(self._reserved.values())

    @property
    def total(self) -> float:
        return self.committed + self.reserved

    def _persist(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "cap_usd": self.cap, "committed_usd": self.committed,
            "reserved_usd": self.reserved, "n_calls": self.n_calls,
            "halted": self.halted, "updated_unix": time.time(),
        }, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def reserve(self, worst_usd: float) -> str:
        with self._lock:
            if self.halted:
                raise BudgetKillSwitch(f"ledger halted at ${self.total:.4f}")
            if self.total + float(worst_usd) >= self.cap:
                self.halted = True
                self._persist()
                raise BudgetKillSwitch(
                    f"kill switch: ${self.total:.4f} + worst case ${worst_usd:.4f} "
                    f"would reach the ${self.cap:.2f} cap")
            rid = uuid.uuid4().hex
            self._reserved[rid] = float(worst_usd)
            self._persist()
            return rid

    def commit(self, rid: str, actual_usd: float) -> None:
        with self._lock:
            self._reserved.pop(rid, None)
            self.committed += max(0.0, float(actual_usd))
            self.n_calls += 1
            # If an actual cost ever exceeds its reservation and pushes committed
            # spend to the cap, halt at once so no further call can start.
            if self.committed >= self.cap:
                self.halted = True
            self._persist()

    def add_external(self, usd: float, n_calls: int = 0) -> None:
        """Record spend made outside the ledger (e.g. the smoke test)."""
        with self._lock:
            self.committed += max(0.0, float(usd))
            self.n_calls += int(n_calls)
            self._persist()
