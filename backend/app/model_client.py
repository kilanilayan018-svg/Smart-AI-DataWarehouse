import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

_here = Path(__file__).resolve().parent
load_dotenv(_here.parent / ".env")
load_dotenv()


def generate_plan_with_model(
    schema_description: str,
    *,
    target_column: str | None = None,
    dataset_name: str | None = None,
    file_path: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:

    model_api_url = os.getenv("MODEL_API_URL", "").strip().rstrip("/")
    api_key = os.getenv("MODEL_API_KEY", "").strip()

    if not model_api_url:
        return None, {"source": "rule_fallback", "model_enabled": False, "reason": "MODEL_API_URL is not set"}

    if not file_path or not Path(file_path).exists():
        return None, {"source": "rule_fallback", "model_enabled": True, "reason": f"No file path provided or file does not exist: {file_path}"}

    timeout = float(os.getenv("MODEL_API_TIMEOUT_SECONDS", "180"))

    try:
        with open(file_path, "rb") as f:
            files = {"file": (Path(file_path).name, f, "text/csv")}
            headers = {"X-API-Key": api_key} if api_key else {}
            response = requests.post(model_api_url, files=files, headers=headers, timeout=timeout)
            response.raise_for_status()

        body = response.json()

        plan = {
            "dataset_name": dataset_name,
            "target_column": body.get("target_column", target_column),
            "task_type": body.get("task_type", "auto"),
            "preprocessing": body,
            "_meta": {
                "plan_source": "deepseek_lora_api",
                "model_api_url_configured": True,
            }
        }

        return plan, {
            "source": "deepseek_lora_api",
            "model_enabled": True,
            "model_api_url": model_api_url,
        }

    except Exception as exc:
        return None, {
            "source": "rule_fallback",
            "model_enabled": True,
            "model_api_url": model_api_url,
            "reason": str(exc),
        }
