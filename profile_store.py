"""
Student profile storage with optional Supabase sync (Step 6).

Behavior:
- JSON file (student_profile.json) is always read/written — it stays the
  local source of truth and the offline fallback.
- If SUPABASE_URL and SUPABASE_KEY are set in .env, the profile is also
  synced to a `student_profiles` table via Supabase's PostgREST API.
  On load, the cloud copy wins when it exists (so progress survives
  Streamlit Cloud restarts).

Create the table once in the Supabase SQL editor (see supabase_setup.sql).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROFILE_FILE = BASE_DIR / "student_profile.json"

load_dotenv(BASE_DIR / ".env", override=False)

DEFAULT_PROFILE = {
    "student_name": "Student",
    "current_index": 0,
    "completed": [],
    "weak": [],
    "history": [],
}

_TABLE = "student_profiles"
_TIMEOUT = 8  # seconds; keep the app responsive if Supabase is unreachable


def _supabase_config() -> tuple[str, str] | None:
    url = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
    key = (os.getenv("SUPABASE_KEY") or "").strip()
    if url and key:
        return url, key
    return None


def _headers(key: str) -> dict:
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def _profile_key(profile: dict) -> str:
    """Stable row key. One row per student name (single-student app for now)."""
    return (profile.get("student_name") or "Student").strip() or "Student"


def _load_local(path: Path) -> dict:
    if not path.exists():
        return dict(DEFAULT_PROFILE)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return dict(DEFAULT_PROFILE)


def _save_local(profile: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)


def load_profile(path: Path | None = None) -> dict:
    """
    Load the profile. Prefers the Supabase copy when configured and present,
    otherwise falls back to the local JSON file.
    """
    local_path = Path(path) if path else PROFILE_FILE
    local = _load_local(local_path)

    cfg = _supabase_config()
    if not cfg:
        return local

    url, key = cfg
    student = _profile_key(local)
    try:
        resp = requests.get(
            f"{url}/rest/v1/{_TABLE}",
            headers=_headers(key),
            params={"select": "data", "student": f"eq.{student}", "limit": 1},
            timeout=_TIMEOUT,
        )
        if resp.ok:
            rows = resp.json()
            if rows and isinstance(rows[0].get("data"), dict):
                cloud = rows[0]["data"]
                # Keep local JSON in sync with the cloud copy
                _save_local(cloud, local_path)
                return cloud
    except requests.RequestException:
        pass  # offline / misconfigured — JSON fallback keeps the app working
    return local


def save_profile(profile: dict, path: Path | None = None) -> None:
    """
    Save the profile locally and, when configured, upsert it to Supabase.
    Local JSON always succeeds. Supabase errors are logged (not raised) so
    the Streamlit app keeps working offline.
    """
    local_path = Path(path) if path else PROFILE_FILE
    _save_local(profile, local_path)

    cfg = _supabase_config()
    if not cfg:
        return

    url, key = cfg
    student = _profile_key(profile)
    try:
        resp = requests.post(
            f"{url}/rest/v1/{_TABLE}",
            headers={
                **_headers(key),
                "Prefer": "resolution=merge-duplicates,return=minimal",
            },
            json={"student": student, "data": profile},
            timeout=_TIMEOUT,
        )
        if not resp.ok:
            # Surface the real cause (e.g. RLS) instead of failing silently
            print(
                f"[profile_store] Supabase save failed "
                f"({resp.status_code}): {resp.text[:300]}"
            )
    except requests.RequestException as exc:
        print(f"[profile_store] Supabase unreachable: {exc}")


def supabase_enabled() -> bool:
    """Report whether cloud sync is active (for UI display)."""
    return _supabase_config() is not None
