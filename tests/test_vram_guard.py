from __future__ import annotations

import pytest

from apex.providers import VramGuardError, assert_ollama_vram_resident, parse_ollama_ps


def test_parse_ollama_ps_accepts_full_gpu_residency() -> None:
    text = """NAME              ID              SIZE      PROCESSOR    UNTIL
llama3.1:8b       abc123          4.9 GB    100% GPU     4 minutes from now
"""
    entries = parse_ollama_ps(text)
    assert len(entries) == 1
    assert entries[0].name == "llama3.1:8b"
    assert entries[0].gpu_percent == 100
    assert not entries[0].has_cpu_split
    assert_ollama_vram_resident("llama3.1:8b", ps_text=text)


def test_vram_guard_rejects_cpu_split() -> None:
    text = """NAME              ID              SIZE      PROCESSOR        UNTIL
llama3.1:8b       abc123          4.9 GB    57% GPU/43% CPU  4 minutes from now
"""
    with pytest.raises(VramGuardError):
        assert_ollama_vram_resident("llama3.1:8b", ps_text=text)


def test_vram_guard_rejects_unexpected_second_model() -> None:
    text = """NAME              ID              SIZE      PROCESSOR    UNTIL
llama3.1:8b       abc123          4.9 GB    100% GPU     4 minutes from now
gemma3:4b         def456          2.7 GB    100% GPU     4 minutes from now
"""
    with pytest.raises(VramGuardError):
        assert_ollama_vram_resident("llama3.1:8b", ps_text=text, max_loaded_models=1)
