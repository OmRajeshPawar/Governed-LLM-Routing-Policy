"""Credential guard for live Gemini (Vertex AI) calls.

Configuration comes only from environment variables, so no key path, project,
or fingerprint lives in source, and nothing in this package loads .env files:

  SOC4AI_GCP_KEY            path to a service-account JSON key (required)
  SOC4AI_GCP_PROJECT        expected GCP project; defaults to the key's project_id
  SOC4AI_GCP_LOCATION       Vertex AI region (default us-central1)
  SOC4AI_KEY_SHA256_PREFIX  optional: pin the key file by its SHA-256 prefix

A key is accepted only if its project_id matches the expected project (and its
fingerprint matches the pin, when one is set), and the Google client variables
point at that same key and project.
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

KEY_ENV = "SOC4AI_GCP_KEY"
PROJECT_ENV = "SOC4AI_GCP_PROJECT"
PIN_ENV = "SOC4AI_KEY_SHA256_PREFIX"
LOCATION = os.environ.get("SOC4AI_GCP_LOCATION", "us-central1")
MODEL = "gemini-2.5-flash"
PROJECT = os.environ.get(PROJECT_ENV, "")  # resolved from the key by force_credentials()


class CredentialGuardError(RuntimeError):
    pass


def _key_path() -> Path:
    raw = os.environ.get(KEY_ENV, "")
    if not raw:
        raise CredentialGuardError(f"{KEY_ENV} is not set")
    path = Path(raw).resolve()
    if not path.is_file():
        raise CredentialGuardError(f"{KEY_ENV} does not point to a readable file")
    return path


@lru_cache(maxsize=8)
def _fingerprint(path: str) -> tuple[str, str]:
    data = Path(path).read_bytes()
    project = json.loads(data.decode("utf-8")).get("project_id", "")
    return hashlib.sha256(data).hexdigest()[:16], project


def _expected_project(key_project: str) -> str:
    return os.environ.get(PROJECT_ENV, "") or key_project


def force_credentials() -> None:
    """Point the Google client at the guarded key and project, then verify."""
    global PROJECT
    key = _key_path()
    _, key_project = _fingerprint(str(key))
    PROJECT = _expected_project(key_project)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key)
    os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT
    os.environ["GCLOUD_PROJECT"] = PROJECT
    os.environ["GOOGLE_CLOUD_LOCATION"] = LOCATION
    os.environ["APEX_VERTEX_MODEL"] = MODEL
    assert_credentials()


def assert_credentials() -> None:
    key = _key_path()
    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not gac or Path(gac).resolve() != key:
        raise CredentialGuardError("GOOGLE_APPLICATION_CREDENTIALS does not match the guarded key")
    fingerprint, key_project = _fingerprint(str(key))
    expected = _expected_project(key_project)
    if not key_project or key_project != expected:
        raise CredentialGuardError("key project_id does not match the expected project")
    pin = os.environ.get(PIN_ENV, "")
    if pin and fingerprint != pin:
        raise CredentialGuardError("key fingerprint does not match the pinned key")
    for var in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"):
        if os.environ.get(var) != expected:
            raise CredentialGuardError(f"{var} is not {expected}")
