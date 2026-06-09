import os
import sys
import json
import time
from pathlib import Path

import pandas as pd
from pandas.errors import ParserError

# --------------------------------------------------
# Project root setup
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Make imports work on any machine / any PyCharm working directory
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.validation import DataValidationModule
from utils.pipeline_logger import PipelineLogger


# --------------------------------------------------
# Paths
# --------------------------------------------------
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SCHEMA_DIR = PROJECT_ROOT / "data" / "schema"
OUTPUT_DIR = PROJECT_ROOT / "logs" / "validation"


def to_posix_path(path):
    """
    Convert Path/string to clean JSON-friendly path.
    """
    if path is None:
        return None
    return str(path).replace("\\", "/")


def parse_versioned_filename(raw_path):
    """
    Expected example:
    20260331_173925_1acb0a5a_automobile_dataset.csv

    Returns:
    version = 1acb0a5a
    dataset_name = automobile_dataset
    base_name = 20260331_173925_1acb0a5a_automobile_dataset
    """
    raw_path = Path(raw_path)
    base_name = raw_path.stem

    parts = base_name.split("_", 3)

    if len(parts) < 4:
        return None, None, base_name

    _, _, version, dataset_name = parts
    return version, dataset_name, base_name


def get_schema_columns(schema):
    """
    Return real dataset columns from schema.
    Excludes _meta because it is metadata, not an actual CSV column.
    """
    if not isinstance(schema, dict):
        return set()

    return {
        col for col in schema.keys()
        if col != "_meta"
    }


def score_dataframe_against_schema(df, schema_columns):
    """
    Score how well a parsed DataFrame matches schema columns.

    Higher score = better parsing.
    This solves cases like cardio_train where comma parsing gives:
    one column called 'id;age;gender;...'
    while semicolon parsing gives the real columns.
    """
    if df is None or df.empty:
        return -9999

    df_columns = set(str(c).strip() for c in df.columns)

    if not schema_columns:
        return len(df_columns)

    matched = len(df_columns.intersection(schema_columns))
    missing = len(schema_columns - df_columns)
    extra = len(df_columns - schema_columns)

    # Strongly punish single-column parsing when schema expects many columns
    one_column_penalty = 0
    if len(df_columns) == 1 and len(schema_columns) > 1:
        only_col = next(iter(df_columns))
        if ";" in only_col or "," in only_col or "\t" in only_col or "|" in only_col:
            one_column_penalty = 100

    return (matched * 10) - (missing * 3) - extra - one_column_penalty


def read_csv_safely(raw_path, schema=None):
    """
    Robust, schema-aware CSV reader.

    Why schema-aware?
    Because some files are not comma-separated. Example:
    cardio_train.csv uses semicolons (;).
    Normal pd.read_csv(file) may succeed but incorrectly return one column.
    So we try multiple separators and choose the parsing that best matches schema columns.

    Returns:
    df, read_info
    """
    raw_path = Path(raw_path)

    schema_columns = get_schema_columns(schema)

    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]

    separator_candidates = [
        (",", "comma"),
        (";", "semicolon"),
        ("\t", "tab"),
        ("|", "pipe"),
    ]

    attempts = []
    last_error = None

    # --------------------------------------------------
    # 1) Try explicit separators
    # --------------------------------------------------
    for encoding in encodings:
        for sep, sep_name in separator_candidates:
            try:
                df = pd.read_csv(
                    raw_path,
                    encoding=encoding,
                    sep=sep,
                    engine="python",
                    on_bad_lines="error"
                )

                score = score_dataframe_against_schema(df, schema_columns)

                attempts.append({
                    "df": df,
                    "score": score,
                    "encoding": encoding,
                    "separator": sep_name,
                    "separator_value": sep,
                    "engine": "python",
                    "on_bad_lines": "error"
                })

            except UnicodeDecodeError as e:
                last_error = e
                continue
            except ParserError as e:
                last_error = e
                continue
            except Exception as e:
                last_error = e
                continue

    # --------------------------------------------------
    # 2) Try automatic separator detection
    # --------------------------------------------------
    for encoding in encodings:
        try:
            df = pd.read_csv(
                raw_path,
                encoding=encoding,
                sep=None,
                engine="python",
                on_bad_lines="error"
            )

            score = score_dataframe_against_schema(df, schema_columns)

            attempts.append({
                "df": df,
                "score": score,
                "encoding": encoding,
                "separator": "auto",
                "separator_value": None,
                "engine": "python",
                "on_bad_lines": "error"
            })

        except UnicodeDecodeError as e:
            last_error = e
            continue
        except ParserError as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            continue

    # --------------------------------------------------
    # 3) Last fallback: tolerate bad lines
    # --------------------------------------------------
    for encoding in encodings:
        for sep, sep_name in separator_candidates:
            try:
                df = pd.read_csv(
                    raw_path,
                    encoding=encoding,
                    sep=sep,
                    engine="python",
                    on_bad_lines="skip"
                )

                score = score_dataframe_against_schema(df, schema_columns)

                attempts.append({
                    "df": df,
                    "score": score,
                    "encoding": encoding,
                    "separator": sep_name,
                    "separator_value": sep,
                    "engine": "python",
                    "on_bad_lines": "skip"
                })

            except Exception as e:
                last_error = e
                continue

    if not attempts:
        raise RuntimeError(f"Could not read CSV file after multiple attempts: {last_error}")

    # Pick the parse attempt that best matches the schema
    best_attempt = max(attempts, key=lambda x: x["score"])

    df = best_attempt["df"]

    read_info = {
        "encoding": best_attempt["encoding"],
        "separator": best_attempt["separator"],
        "separator_value": best_attempt["separator_value"],
        "engine": best_attempt["engine"],
        "on_bad_lines": best_attempt["on_bad_lines"],
        "schema_match_score": best_attempt["score"],
        "columns_read": len(df.columns),
        "rows_read": len(df),
    }

    return df, read_info


def save_invalid_report(dataset_name, version, raw_path, schema_path, error_message):
    """
    Save invalid validation report.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "is_valid": False,
        "errors": [error_message],
        "warnings": [],
        "name": dataset_name,
        "version": version,
        "raw_file_path": to_posix_path(raw_path),
        "schema_path": to_posix_path(schema_path) if schema_path else None,
        "validation_status": "invalid"
    }

    report_path = OUTPUT_DIR / f"{dataset_name}_v{version}_validation.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    return report, report_path


def print_dataset_result(dataset_name, version, report, report_path):
    """
    Print detailed validation status for each dataset.
    """
    status = report.get("validation_status", "unknown")
    errors = report.get("errors", [])
    warnings = report.get("warnings", [])

    errors_count = len(errors)
    warnings_count = len(warnings)

    if status == "valid":
        print(
            f"✅ VALID   | {dataset_name} | "
            f"version={version} | errors={errors_count} | warnings={warnings_count}"
        )
    else:
        print(
            f"⚠️ INVALID | {dataset_name} | "
            f"version={version} | errors={errors_count} | warnings={warnings_count}"
        )

        if errors_count > 0:
            print("   🔎 First errors:")
            for err in errors[:3]:
                print(f"      - {err}")

    if warnings_count > 0:
        print("   ⚠️ First warnings:")
        for warning in warnings[:3]:
            print(f"      - {warning}")

    print(f"   📄 Report: {report_path}")


def normalize_validation_report(report):
    """
    Ensure report always has expected keys.
    This protects run_validation.py from older/newer validation.py versions.
    """
    if not isinstance(report, dict):
        return {
            "is_valid": False,
            "errors": ["Validation module returned non-dictionary report"],
            "warnings": [],
            "validation_status": "invalid"
        }

    report.setdefault("errors", [])
    report.setdefault("warnings", [])

    if "is_valid" not in report:
        report["is_valid"] = len(report.get("errors", [])) == 0

    if "validation_status" not in report:
        report["validation_status"] = "valid" if report["is_valid"] else "invalid"

    return report


def main():
    # Force execution from project root so paths work on any team member's machine
    os.chdir(PROJECT_ROOT)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger = PipelineLogger()
    run_id = logger.new_run_id()

    print(f"\n🆔 Pipeline Run ID: {run_id}")

    all_reports = []
    dataset_results = []

    raw_files = sorted(RAW_DIR.glob("*.csv"))

    print("📍 Project root:", PROJECT_ROOT)
    print("📍 Current working directory:", Path.cwd())
    print("📁 RAW_DIR:", RAW_DIR)
    print("📁 SCHEMA_DIR:", SCHEMA_DIR)
    print("📄 Raw CSV files found:", len(raw_files))
    print("📄 First raw files:", [to_posix_path(p) for p in raw_files[:5]])

    if not raw_files:
        print("\n⚠️ No CSV files found in data/raw.")
        print("👉 Run pipelines/run_ingestion.py first.")
        print("👉 Expected folder:")
        print(f"   {RAW_DIR}")

    print("\n" + "=" * 80)
    print("📋 VALIDATION PER-DATASET STATUS")
    print("=" * 80)

    for raw_path in raw_files:
        step_start_time = time.perf_counter()

        version, dataset_name, base_name = parse_versioned_filename(raw_path)

        if version is None:
            print(f"⚠️ SKIPPED | Unexpected filename format: {raw_path}")
            dataset_results.append({
                "dataset_name": base_name,
                "version": None,
                "status": "skipped",
                "errors_count": 1,
                "warnings_count": 0,
                "first_error": "Unexpected filename format"
            })
            continue

        schema_path = SCHEMA_DIR / f"{base_name}_schema.json"

        print(f"\n🔍 Checking dataset: {dataset_name}")
        print(f"   📦 Raw file: {raw_path.name}")
        print(f"   🧾 Schema: {schema_path.name}")

        if not schema_path.exists():
            error_message = "Schema file not found"

            report, report_path = save_invalid_report(
                dataset_name=dataset_name,
                version=version,
                raw_path=raw_path,
                schema_path=None,
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
                    "raw_file_path": to_posix_path(raw_path),
                    "schema_path": None,
                    "report_path": to_posix_path(report_path),
                    "validation_status": "invalid",
                    "errors_count": 1,
                    "warnings_count": 0
                }
            )

            all_reports.append(report)
            dataset_results.append({
                "dataset_name": dataset_name,
                "version": version,
                "status": "invalid",
                "errors_count": 1,
                "warnings_count": 0,
                "first_error": error_message
            })

            print_dataset_result(dataset_name, version, report, report_path)
            continue

        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)

            df, read_info = read_csv_safely(raw_path, schema=schema)

            print(
                "   📖 Read CSV:",
                f"encoding={read_info['encoding']},",
                f"separator={read_info['separator']},",
                f"engine={read_info['engine']},",
                f"on_bad_lines={read_info['on_bad_lines']},",
                f"shape={df.shape},",
                f"schema_match_score={read_info['schema_match_score']}"
            )

            validator = DataValidationModule(df, schema)

            report_path = validator.save_result(
                dataset_name=dataset_name,
                version=version,
                raw_file_path=to_posix_path(raw_path),
                schema_path=to_posix_path(schema_path),
                output_dir=str(OUTPUT_DIR)
            )

            report_path = Path(report_path)

            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)

            report = normalize_validation_report(report)

            validation_status = report.get("validation_status", "unknown")
            errors_count = len(report.get("errors", []))
            warnings_count = len(report.get("warnings", []))

            logger.log_event(
                run_id=run_id,
                step="validation",
                status="completed",
                start_time=step_start_time,
                dataset_name=dataset_name,
                version=version,
                error_detail=None if validation_status == "valid" else "Dataset validation returned invalid",
                details={
                    "raw_file_path": to_posix_path(raw_path),
                    "schema_path": to_posix_path(schema_path),
                    "report_path": to_posix_path(report_path),
                    "validation_status": validation_status,
                    "errors_count": errors_count,
                    "warnings_count": warnings_count,
                    "read_info": read_info,
                    "rows": int(df.shape[0]),
                    "columns": int(df.shape[1])
                }
            )

            all_reports.append(report)

            first_error = report.get("errors", [None])[0] if report.get("errors") else None

            dataset_results.append({
                "dataset_name": dataset_name,
                "version": version,
                "status": validation_status,
                "errors_count": errors_count,
                "warnings_count": warnings_count,
                "first_error": first_error
            })

            print_dataset_result(dataset_name, version, report, report_path)

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
                    "raw_file_path": to_posix_path(raw_path),
                    "schema_path": to_posix_path(schema_path),
                    "report_path": to_posix_path(report_path),
                    "validation_status": "invalid",
                    "errors_count": 1,
                    "warnings_count": 0
                }
            )

            all_reports.append(report)
            dataset_results.append({
                "dataset_name": dataset_name,
                "version": version,
                "status": "failed",
                "errors_count": 1,
                "warnings_count": 0,
                "first_error": error_message
            })

            print(f"❌ FAILED  | {dataset_name} | version={version}")
            print(f"   🔎 Error: {error_message}")
            print(f"   📄 Report: {report_path}")
            continue

    valid_datasets = [
        r for r in all_reports
        if r.get("validation_status") == "valid"
    ]

    invalid_datasets = [
        r for r in all_reports
        if r.get("validation_status") != "valid"
    ]

    summary = {
        "all_datasets": all_reports,
        "valid_datasets": valid_datasets,
        "invalid_datasets": invalid_datasets,
        "dataset_results": dataset_results
    }

    summary_path = OUTPUT_DIR / "validation_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 80)
    print("📊 VALIDATION FINAL SUMMARY")
    print("=" * 80)

    print(f"📦 Validation summary saved to: {summary_path}")
    print(f"📊 Total datasets found: {len(raw_files)}")
    print(f"📊 Total validated datasets: {len(all_reports)}")
    print(f"✅ Valid datasets count: {len(valid_datasets)}")
    print(f"⚠️ Invalid/failed datasets count: {len(invalid_datasets)}")

    if invalid_datasets:
        print("\n⚠️ Invalid / failed datasets details:")
        for item in dataset_results:
            if item["status"] != "valid":
                print(
                    f"   - {item['dataset_name']} | "
                    f"status={item['status']} | "
                    f"errors={item['errors_count']} | "
                    f"warnings={item['warnings_count']} | "
                    f"first_error={item['first_error']}"
                )
    else:
        print("\n🎉 All datasets are valid. Full dataset validation passed.")

    summary_start_time = time.perf_counter()

    logger.log_event(
        run_id=run_id,
        step="validation_summary",
        status="completed",
        start_time=summary_start_time,
        dataset_name="all_datasets",
        version=None,
        details={
            "summary_path": to_posix_path(summary_path),
            "total_datasets_found": len(raw_files),
            "total_validated_datasets": len(all_reports),
            "valid_datasets": len(valid_datasets),
            "invalid_datasets": len(invalid_datasets)
        }
    )

    print("\n📝 Pipeline events logged to: logs/pipeline_events.jsonl")


if __name__ == "__main__":
    main()