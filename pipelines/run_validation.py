import os
import glob
import json
import time
import pandas as pd

from pipelines.validation import DataValidationModule
from utils.pipeline_logger import PipelineLogger


RAW_DIR = "data/raw"
SCHEMA_DIR = "data/schema"
OUTPUT_DIR = "logs/validation"


def parse_versioned_filename(raw_path):
    """
    Example:
    20260331_173925_1acb0a5a_automobile_dataset.csv
    -> version = 1acb0a5a
    -> dataset_name = automobile_dataset
    """
    filename = os.path.basename(raw_path)
    base_name = os.path.splitext(filename)[0]

    parts = base_name.split("_", 3)
    if len(parts) < 4:
        return None, None, base_name

    _, _, version, dataset_name = parts
    return version, dataset_name, base_name


def save_invalid_report(dataset_name, version, raw_path, schema_path, error_message):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    report = {
        "is_valid": False,
        "errors": [error_message],
        "name": dataset_name,
        "version": version,
        "raw_file_path": raw_path.replace("\\", "/"),
        "schema_path": schema_path.replace("\\", "/") if schema_path else None,
        "validation_status": "invalid"
    }

    report_path = os.path.join(
        OUTPUT_DIR, f"{dataset_name}_v{version}_validation.json"
    )

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    return report, report_path


def read_csv_safely(raw_path):
    """
    Try reading CSV with utf-8 first, then latin1.
    If parsing still fails, raise the exception.
    """
    try:
        return pd.read_csv(raw_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(raw_path, encoding="latin1")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger = PipelineLogger()
    run_id = logger.new_run_id()

    print(f"\n🆔 Pipeline Run ID: {run_id}")

    all_reports = []
    raw_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))

    for raw_path in raw_files:
        step_start_time = time.perf_counter()

        version, dataset_name, base_name = parse_versioned_filename(raw_path)

        if version is None:
            print(f"⚠️ Skipping file with unexpected name format: {raw_path}")
            continue

        schema_path = os.path.join(SCHEMA_DIR, f"{base_name}_schema.json")

        if not os.path.exists(schema_path):
            error_message = "Schema file not found"

            report, report_path = save_invalid_report(
                dataset_name=dataset_name,
                version=version,
                raw_path=raw_path,
                schema_path="",
                error_message=error_message
            )

            logger.log_event(
                run_id=run_id,
                step="validation",
                status="failed",
                start_time=step_start_time,
                dataset_name=dataset_name,
                version=version,
                error_detail=error_message,
                details={
                    "raw_file_path": raw_path.replace("\\", "/"),
                    "schema_path": None,
                    "report_path": report_path.replace("\\", "/")
                }
            )

            all_reports.append(report)
            print(f"❌ No schema found, saved invalid report: {report_path}")
            continue

        print(f"\n🔍 Validating: {raw_path}")

        try:
            df = read_csv_safely(raw_path)

            with open(schema_path) as f:
                schema = json.load(f)

            validator = DataValidationModule(df, schema)

            report_path = validator.save_result(
                dataset_name=dataset_name,
                version=version,
                raw_file_path=raw_path,
                schema_path=schema_path,
                output_dir=OUTPUT_DIR
            )

            with open(report_path) as f:
                report = json.load(f)

            logger.log_event(
                run_id=run_id,
                step="validation",
                status="completed",
                start_time=step_start_time,
                dataset_name=dataset_name,
                version=version,
                error_detail=None,
                details={
                    "raw_file_path": raw_path.replace("\\", "/"),
                    "schema_path": schema_path.replace("\\", "/"),
                    "report_path": report_path.replace("\\", "/"),
                    "validation_status": report["validation_status"],
                    "errors_count": len(report.get("errors", []))
                }
            )

            all_reports.append(report)
            print(f"✅ Saved report: {report_path}")

        except Exception as e:
            error_message = f"Failed to read or validate dataset: {str(e)}"

            report, report_path = save_invalid_report(
                dataset_name=dataset_name,
                version=version,
                raw_path=raw_path,
                schema_path=schema_path,
                error_message=error_message
            )

            logger.log_event(
                run_id=run_id,
                step="validation",
                status="failed",
                start_time=step_start_time,
                dataset_name=dataset_name,
                version=version,
                error_detail=error_message,
                details={
                    "raw_file_path": raw_path.replace("\\", "/"),
                    "schema_path": schema_path.replace("\\", "/"),
                    "report_path": report_path.replace("\\", "/")
                }
            )

            all_reports.append(report)
            print(f"❌ Validation failed, saved invalid report: {report_path}")
            continue

    valid_datasets = [
        r for r in all_reports
        if r["validation_status"] == "valid"
    ]

    summary = {
        "all_datasets": all_reports,
        "valid_datasets": valid_datasets
    }

    summary_path = os.path.join(OUTPUT_DIR, "validation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n📦 Validation summary saved to: {summary_path}")
    print(f"✅ Valid datasets count: {len(valid_datasets)}")
    print(f"📊 Total validated datasets: {len(all_reports)}")

    summary_start_time = time.perf_counter()

    logger.log_event(
        run_id=run_id,
        step="validation_summary",
        status="completed",
        start_time=summary_start_time,
        dataset_name="all_datasets",
        version=None,
        details={
            "summary_path": summary_path.replace("\\", "/"),
            "total_datasets": len(all_reports),
            "valid_datasets": len(valid_datasets),
            "invalid_datasets": len(all_reports) - len(valid_datasets)
        }
    )

    print("📝 Pipeline events logged to: logs/pipeline_events.jsonl")


if __name__ == "__main__":
    main()