import json
import shutil
import tempfile
from pathlib import Path
from typing import Any
from datetime import datetime

import pandas as pd

from .db import insert_dataset, insert_plan, insert_run, fetch_recent_runs


def _import_existing_modules():
    from pipelines.ingestion import DataIngestionModule
    from pipelines.schema_extractor import extract_schema
    from pipelines.validation import DataValidationModule
    from pipelines.plan_generator import PlanGenerator
    return DataIngestionModule, extract_schema, DataValidationModule, PlanGenerator


def _read_dataframe(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path, sep=None, engine="python", on_bad_lines="skip")
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    raise ValueError("Unsupported file type. Only CSV and Excel are allowed.")


def process_uploaded_file(uploaded_file, target_column: str | None) -> dict[str, Any]:
    DataIngestionModule, extract_schema, DataValidationModule, PlanGenerator = _import_existing_modules()

    suffix = Path(uploaded_file.filename).suffix or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(uploaded_file.file, tmp)
        temp_path = Path(tmp.name)

    ingestor = DataIngestionModule()
    metadata = ingestor.ingest(str(temp_path))
    stored_path = Path(metadata["stored_path"])
    dataset_name = Path(metadata["stored_filename"]).stem

    schema = extract_schema(str(stored_path))

    schema_dir = Path("data/schema")
    schema_dir.mkdir(parents=True, exist_ok=True)
    schema_path = schema_dir / f"{dataset_name}_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    df = _read_dataframe(stored_path)
    validation = DataValidationModule(
        df=df,
        schema=schema,
        target_column=target_column or None
    ).validate()

    generator = PlanGenerator(target_column=target_column or None)
    plan = generator.generate_plan(schema, dataset_name)

    if hasattr(generator, "save_plan"):
        plan_path = generator.save_plan(plan, dataset_name)
    else:
        plan_dir = Path("metadata/plans")
        plan_dir.mkdir(parents=True, exist_ok=True)
        plan_path = str(plan_dir / f"{dataset_name}_plan.json")
        Path(plan_path).write_text(json.dumps(plan, indent=2), encoding="utf-8")

    created_at = datetime.now().isoformat(timespec="seconds")

    dataset_id = insert_dataset(
        original_filename=metadata["original_filename"],
        stored_filename=metadata["stored_filename"],
        stored_path=metadata["stored_path"],
        version_id=metadata["version_id"],
        file_type=metadata["file_type"],
        rows_count=metadata["rows"],
        columns_count=metadata["columns_count"],
        created_at=created_at,
    )

    plan_id = insert_plan(
        dataset_id=dataset_id,
        target_column=plan.get("target_column", ""),
        task_type=plan.get("task_type", ""),
        plan_source="auto",
        plan_path=str(plan_path),
        created_at=created_at,
    )

    run_id = insert_run(
        dataset_id=dataset_id,
        plan_id=plan_id,
        status="success",
        step="generate-plan",
        message="Schema extracted and plan generated successfully",
        created_at=created_at,
    )

    return {
        "run_id": run_id,
        "dataset_name": dataset_name,
        "target_column": plan.get("target_column"),
        "task_type": plan.get("task_type"),
        "schema": schema,
        "validation": validation,
        "plan": plan,
        "download_ready": Path(
            plan.get("export_paths", {}).get("curated_data", "")
        ).exists(),
    }


def save_manual_plan(
    dataset_name: str,
    target_column: str | None,
    schema: dict[str, Any],
    plan: dict[str, Any]
) -> dict[str, Any]:
    plan_dir = Path("metadata/plans")
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plan_dir / f"{dataset_name}_manual_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    created_at = datetime.now().isoformat(timespec="seconds")

    # Manual plans may not yet be tied to a dataset row if no upload happened first.
    # So for now we only save a lightweight run entry if you want to keep the flow simple.
    # If you want, later we can make manual plan saving require an existing dataset_id.
    return {
        "run_id": None,
        "plan_path": str(plan_path),
        "status": "manual_plan_saved",
        "created_at": created_at,
    }


def list_runs(limit: int = 20) -> list[dict[str, Any]]:
    return fetch_recent_runs(limit)