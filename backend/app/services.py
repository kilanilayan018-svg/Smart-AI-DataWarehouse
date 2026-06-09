import json
import logging
import mimetypes
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .db import fetch_recent_datasets, fetch_recent_runs, insert_dataset, insert_plan, insert_run
from . import supabase_client as sb
from .model_client import generate_plan_with_model
from .outputs_writer import save_to_outputs
from .schema_prompt import schema_to_model_prompt
from .plan_executor import execute_plan

log = logging.getLogger("smart_ai_dw_api")

# ── This file's absolute location (used to anchor all paths) ──────────────────
_HERE = Path(__file__).resolve()          # .../backend/app/services.py
_BACKEND = _HERE.parent.parent            # .../backend
_PROJECT_ROOT = _BACKEND.parent           # .../smart_ai_dw_rebuilt_ngrok

log.info(f"services.py loaded from: {_HERE}")
log.info(f"Project root resolved to: {_PROJECT_ROOT}")


def _import_existing_modules():
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from pipelines.ingestion import DataIngestionModule
    from pipelines.schema_extractor import extract_schema
    from pipelines.validation import DataValidationModule
    from pipelines.plan_generator import PlanGenerator
    return DataIngestionModule, extract_schema, DataValidationModule, PlanGenerator


def _import_dispatcher_bridge():
    """Lazy import so a problem in the dispatcher never breaks module load."""
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from pipelines.dispatcher_bridge import dispatch_from_model_response
    return dispatch_from_model_response


def _read_dataframe(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path, sep=None, engine="python", on_bad_lines="skip")
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    raise ValueError("Unsupported file type. Only CSV and Excel are allowed.")


def _plan_source(plan: dict) -> str:
    """
    Read plan_source from wherever it lives.
    - Dispatcher plans:  plan["_debug"]["source"] / ["final_decision"]
    - Old plans:         plan["_meta"]["plan_source"]
    Falls back to "auto".
    """
    debug = plan.get("_debug", {}) if isinstance(plan.get("_debug"), dict) else {}
    return (
        debug.get("source")
        or debug.get("final_decision")
        or plan.get("_meta", {}).get("plan_source")
        or "auto"
    )


def _plan_model_status(plan: dict, model_meta: dict) -> Any:
    """
    Return a model_status value that the frontend can display.
    - Old plans carry model_meta injected via plan["_meta"]["model_status"].
    - New plans from plan_dispatcher carry debug info in plan["_debug"].
    We prefer the live model_meta dict when available, and fall back
    to whatever debug info the plan already carries.
    """
    if model_meta:
        return model_meta
    return plan.get("_meta", {}).get("model_status") or plan.get("_debug")


def _smote_bool(plan: dict) -> bool:
    """
    Extract a plain bool from smote regardless of plan shape.
    - Old plans: smote may be absent, False, or a bool directly on the plan.
    - New plans (plan_dispatcher): smote is always {"enabled": bool, "reason": str}.
    """
    smote = plan.get("smote", False)
    if isinstance(smote, dict):
        return bool(smote.get("enabled", False))
    return bool(smote)


def process_uploaded_file(uploaded_file, target_column: str | None, owner_id: str | None = None) -> dict[str, Any]:
    started = time.time()
    DataIngestionModule, extract_schema, DataValidationModule, PlanGenerator = _import_existing_modules()

    suffix = Path(uploaded_file.filename or "dataset.csv").suffix or ".csv"
    raw_bytes = uploaded_file.file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw_bytes)
        temp_path = Path(tmp.name)

    # Always use absolute paths anchored to this file's location
    raw_dir      = _PROJECT_ROOT / "data" / "raw"
    metadata_dir = _PROJECT_ROOT / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    ingestor = DataIngestionModule(
        raw_dir=str(raw_dir),
        metadata_dir=str(metadata_dir),
    )
    metadata = ingestor.ingest(str(temp_path))
    stored_path = Path(metadata["stored_path"])

    log.info(f"stored_path = {stored_path}")
    log.info(f"stored_path exists = {stored_path.exists()}")

    dataset_name = Path(metadata["stored_filename"]).stem

    schema = extract_schema(str(stored_path))
    if target_column:
        schema.setdefault("_meta", {})["target_column"] = target_column
        if target_column in schema and isinstance(schema[target_column], dict):
            for _col, _info in schema.items():
                if isinstance(_info, dict):
                    _info["is_target"] = (_col == target_column)

    schema_dir = _PROJECT_ROOT / "data" / "schema"
    schema_dir.mkdir(parents=True, exist_ok=True)
    schema_path = schema_dir / f"{dataset_name}_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    df = _read_dataframe(stored_path)
    validation = DataValidationModule(df=df, schema=schema, target_column=target_column or None).validate()

    schema_description = schema_to_model_prompt(schema, dataset_name, target_column)

    log.info(f"Calling model with file_path={stored_path}")
    raw_model_plan, model_meta = generate_plan_with_model(
        schema_description,
        target_column=target_column or None,
        dataset_name=dataset_name,
        file_path=str(stored_path),
    )
    log.info(f"Model result: {model_meta}")

    # ── Dispatcher bridge: validate model output, fall back if weak/invalid ───
    # The bridge always returns a valid normalized plan (model → PlanGenerator
    # → internal safe fallback). It never returns None and never raises.
    try:
        dispatch_from_model_response = _import_dispatcher_bridge()
        plan = dispatch_from_model_response(
            model_response=raw_model_plan,
            schema=schema,
            dataset_name=dataset_name,
            target_column=target_column or None,
        )
        plan.setdefault("_meta", {})["model_status"] = model_meta
        log.info(f"Plan decision: {plan.get('_debug', {}).get('final_decision')}")
    except Exception as e:
        # Absolute last-resort safety net so the endpoint never 500s on the demo.
        log.error(f"Dispatcher bridge crashed ({e}); using direct fallback")
        if raw_model_plan is not None:
            plan = raw_model_plan
            plan.setdefault("_meta", {})["model_status"] = model_meta
        else:
            generator = PlanGenerator(target_column=target_column or None)
            plan = generator.generate_plan(schema, dataset_name, metadata=metadata, validation=validation)
            plan.setdefault("_meta", {})["plan_source"] = "rule_based_fallback"
            plan["_meta"]["model_status"] = model_meta

    # Save model output to Supabase outputs table
    save_to_outputs(dataset_name, target_column, plan, model_meta)

    # ── Persist the plan to disk ──────────────────────────────────────────────
    plan_dir = _PROJECT_ROOT / "metadata" / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_path = str(plan_dir / f"{dataset_name}_plan.json")
    Path(plan_path).write_text(json.dumps(plan, indent=2), encoding="utf-8")

    created_at = datetime.now().isoformat(timespec="seconds")
    storage_path = None

    # ── Execute the plan and produce a curated CSV ────────────────────────────
    curated_path, exec_status = execute_plan(plan, stored_path, dataset_name)
    log.info(f"Plan execution status: {exec_status}, curated_path: {curated_path}")
    download_ready = curated_path is not None and Path(curated_path).exists()

    # Resolve plan metadata using helper functions that understand both
    # old (_meta) and new (_debug) plan shapes from plan_dispatcher.
    resolved_plan_source = _plan_source(plan)
    resolved_model_status = _plan_model_status(plan, model_meta)

    if sb.enabled():
        folder = owner_id or "anonymous"
        storage_path = f"{folder}/{metadata['stored_filename']}"
        content_type = mimetypes.guess_type(metadata["stored_filename"])[0] or "application/octet-stream"
        try:
            sb.upload_storage(storage_path, raw_bytes, content_type)
        except Exception as e:
            log.warning(f"Storage upload skipped: {e}")
            storage_path = None
        dataset_row = sb.insert_row("datasets", {
            "owner_id": owner_id,
            "original_filename": metadata["original_filename"],
            "stored_filename": metadata["stored_filename"],
            "stored_path": metadata["stored_path"],
            "storage_path": storage_path,
            "version_id": metadata["version_id"],
            "file_type": metadata["file_type"],
            "rows_count": metadata["rows"],
            "columns_count": metadata["columns_count"],
            "status": "planned",
        })
        plan_row = sb.insert_row("plans", {
            "dataset_id": dataset_row["id"],
            "owner_id": owner_id,
            "target_column": plan.get("target_column"),
            "task_type": plan.get("task_type"),
            "plan_source": resolved_plan_source,
            "plan": plan,
            "schema": schema,
            "validation": validation,
            "plan_path": str(plan_path),
        })
        run_row = sb.insert_row("runs", {
            "dataset_id": dataset_row["id"],
            "plan_id": plan_row["id"],
            "owner_id": owner_id,
            "status": "success",
            "step": "upload-and-plan",
            "message": "Dataset uploaded, profiled, validated, and planned.",
            "duration_ms": int((time.time() - started) * 1000),
        })
        dataset_id, plan_id, run_id = dataset_row["id"], plan_row["id"], run_row["id"]
    else:
        dataset_id = insert_dataset(metadata["original_filename"], metadata["stored_filename"], metadata["stored_path"], metadata["version_id"], metadata["file_type"], metadata["rows"], metadata["columns_count"], created_at)
        plan_id = insert_plan(dataset_id, plan.get("target_column", ""), plan.get("task_type", ""), resolved_plan_source, str(plan_path), created_at)
        run_id = insert_run(dataset_id, plan_id, "success", "upload-and-plan", "Dataset uploaded, profiled, validated, and planned.", created_at)

    return {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "plan_id": plan_id,
        "dataset_name": dataset_name,
        "target_column": plan.get("target_column"),
        "task_type": plan.get("task_type"),
        "schema": schema,
        "validation": validation,
        "plan": plan,
        "model_status": resolved_model_status,
        "storage_path": storage_path,
        "curated_path": curated_path,
        "exec_status": exec_status,
        "download_ready": download_ready,
    }


def save_manual_plan(dataset_name: str, target_column: str | None, schema: dict[str, Any], plan: dict[str, Any], owner_id: str | None = None) -> dict[str, Any]:
    plan_dir = _PROJECT_ROOT / "metadata" / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plan_dir / f"{dataset_name}_manual_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if sb.enabled():
        dataset_row = sb.insert_row("datasets", {"owner_id": owner_id, "original_filename": dataset_name, "status": "manual_plan", "rows_count": 0, "columns_count": len(schema.get("columns", [])) if isinstance(schema.get("columns"), list) else 0})
        plan_row = sb.insert_row("plans", {"dataset_id": dataset_row["id"], "owner_id": owner_id, "target_column": target_column, "task_type": plan.get("task_type"), "plan_source": "manual", "plan": plan, "schema": schema, "validation": {}, "plan_path": str(plan_path)})
        run_row = sb.insert_row("runs", {"dataset_id": dataset_row["id"], "plan_id": plan_row["id"], "owner_id": owner_id, "status": "saved", "step": "manual-plan", "message": "Manual plan saved."})
        return {"run_id": run_row["id"], "plan_path": str(plan_path), "status": "manual_plan_saved"}
    return {"run_id": None, "plan_path": str(plan_path), "status": "manual_plan_saved", "created_at": datetime.now().isoformat(timespec="seconds")}


def list_runs(limit: int = 20, owner_id: str | None = None) -> list[dict[str, Any]]:
    if sb.enabled():
        return sb.list_rows("runs", owner_id, limit)
    return fetch_recent_runs(limit)


def list_datasets(limit: int = 50, owner_id: str | None = None) -> list[dict[str, Any]]:
    if sb.enabled():
        return sb.list_rows("datasets", owner_id, limit)
    return fetch_recent_datasets(limit)
