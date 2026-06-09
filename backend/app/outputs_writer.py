"""
Writes model output directly to Supabase outputs table.
Uses its own connection so it works even when sb.enabled() is False.
"""
import json
import logging
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_env_path)
load_dotenv()

log = logging.getLogger("smart_ai_dw_api")

OUTPUTS_SUPABASE_URL = os.getenv("OUTPUTS_SUPABASE_URL", "").strip().rstrip("/")
OUTPUTS_SERVICE_KEY  = os.getenv("OUTPUTS_SERVICE_KEY", "").strip()


def _resolve_smote_bool(smote_value: Any) -> bool:
    """
    Extract a plain bool from smote regardless of plan shape.
    - Old plans (model_client / PlanGenerator): smote is absent, False, or a plain bool.
    - New plans (plan_dispatcher):              smote is {"enabled": bool, "reason": str}.
    """
    if isinstance(smote_value, dict):
        return bool(smote_value.get("enabled", False))
    return bool(smote_value)


def save_to_outputs(
    dataset_name: str,
    target_column: str | None,
    plan: dict[str, Any] | None,
    model_meta: dict[str, Any],
) -> None:
    if not OUTPUTS_SUPABASE_URL or not OUTPUTS_SERVICE_KEY:
        log.warning("OUTPUTS_SUPABASE_URL or OUTPUTS_SERVICE_KEY not set — skipping outputs table")
        return

    preprocessing = plan.get("preprocessing", {}) if plan else {}
    split = plan.get("split_strategy", preprocessing.get("split_strategy", {})) if plan else {}
    smote_raw = plan.get("smote", preprocessing.get("smote", False)) if plan else False

    payload = {
        "dataset_name": dataset_name,
        "target_column": (plan.get("target_column") if plan else None) or target_column,
        "task_type": (plan.get("task_type") if plan else None) or "unknown",
        "drop_columns": json.dumps(preprocessing.get("drop_columns", [])),
        "imputation": json.dumps(preprocessing.get("imputation", {})),
        "encoding": json.dumps(preprocessing.get("encoding", {})),
        "scaling": json.dumps(preprocessing.get("scaling", {})),
        "log_transform": json.dumps(preprocessing.get("log_transform", [])),
        "split_strategy": json.dumps(split),
        "smote": _resolve_smote_bool(smote_raw),
        "plan_source": model_meta.get("source", "unknown"),
        "model_api_url": model_meta.get("model_api_url", ""),
        "raw_plan": json.dumps(plan) if plan else None,
    }

    headers = {
        "apikey": OUTPUTS_SERVICE_KEY,
        "Authorization": f"Bearer {OUTPUTS_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

    try:
        res = requests.post(
            f"{OUTPUTS_SUPABASE_URL}/rest/v1/outputs",
            headers=headers,
            json=payload,
            timeout=30,
        )
        res.raise_for_status()
        log.info("Saved to Supabase outputs table")
    except Exception as e:
        log.warning(f"Could not save to outputs table: {e}")
