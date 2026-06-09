import base64
import json
import os
from typing import Any
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

# Always load .env relative to this file's location
_env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_env_path)
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
SUPABASE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "datasets").strip() or "datasets"

PLACEHOLDER_FRAGMENTS = (
    "your_project_ref",
    "your-supabase",
    "your_supabase",
    "your_service_role",
    "your_anon",
    "replace_me",
    "placeholder",
    "example.supabase.co",
)


def _is_placeholder(value: str) -> bool:
    lowered = (value or "").strip().lower()
    if not lowered:
        return True
    return any(fragment in lowered for fragment in PLACEHOLDER_FRAGMENTS)


def _valid_supabase_url(url: str) -> bool:
    if _is_placeholder(url):
        return False
    try:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc) and parsed.netloc.endswith("supabase.co")
    except Exception:
        return False


def enabled() -> bool:
    """Return True only when real Supabase credentials are configured.

    Placeholders must not trigger network calls. When Supabase is not ready,
    the backend uses local SQLite so the website stays demo-safe.
    """
    return _valid_supabase_url(SUPABASE_URL) and not _is_placeholder(SUPABASE_SERVICE_ROLE_KEY)


def status() -> dict[str, Any]:
    return {
        "enabled": enabled(),
        "mode": "supabase" if enabled() else "local_sqlite_demo",
        "url_configured": _valid_supabase_url(SUPABASE_URL),
        "service_key_configured": not _is_placeholder(SUPABASE_SERVICE_ROLE_KEY),
        "bucket": SUPABASE_BUCKET,
        "note": "Supabase is optional. Add real backend/.env keys later to switch from local SQLite to Supabase.",
    }


def _headers() -> dict[str, str]:
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def decode_user_id(auth_header: str | None) -> str | None:
    if not auth_header or not auth_header.lower().startswith("bearer "):
        return None
    token = auth_header.split(" ", 1)[1]
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload.encode()))
        return data.get("sub")
    except Exception:
        return None


def insert_row(table: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not enabled():
        raise RuntimeError("Supabase is not configured. The app should be using local SQLite fallback instead.")
    try:
        res = requests.post(f"{SUPABASE_URL}/rest/v1/{table}", headers=_headers(), json=payload, timeout=30)
        res.raise_for_status()
        data = res.json()
        return data[0] if data else {}
    except Exception as e:
        import logging
        logging.getLogger("smart_ai_dw_api").warning(f"Supabase insert to {table} failed: {e}")
        return {"id": None}


def list_rows(table: str, owner_id: str | None, limit: int = 20) -> list[dict[str, Any]]:
    if not enabled():
        return []
    params = {"select": "*", "order": "created_at.desc", "limit": str(limit)}
    if owner_id:
        params["owner_id"] = f"eq.{owner_id}"
    try:
        res = requests.get(f"{SUPABASE_URL}/rest/v1/{table}", headers=_headers(), params=params, timeout=30)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        return []


def upload_storage(path: str, content: bytes, content_type: str = "application/octet-stream") -> str:
    if not enabled():
        raise RuntimeError("Supabase is not configured. The app should be using local storage fallback instead.")
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": content_type,
        "x-upsert": "true",
    }
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{path}"
    try:
        res = requests.post(url, headers=headers, data=content, timeout=60)
        res.raise_for_status()
    except Exception as e:
        import logging
        logging.getLogger("smart_ai_dw_api").warning(f"Storage upload failed: {e}")
    return path
