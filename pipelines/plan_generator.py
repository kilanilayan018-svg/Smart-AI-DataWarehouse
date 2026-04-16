"""
Plan Generator - T1.6
Author: Laith Habash

Pipeline-aware plan generator that uses:
- ingestion metadata
- schema output
- validation result
- transformation output contract
- feature engineering output contract

This module generates a final structured preprocessing plan JSON.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = PROJECT_ROOT / "data" / "schema"
PLAN_DIR = PROJECT_ROOT / "metadata" / "plans"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CURATED_DIR = PROJECT_ROOT / "data" / "curated"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
FEATURES_SAMPLE_DIR = PROJECT_ROOT / "data" / "features_samples"
VALIDATION_LOG_DIR = PROJECT_ROOT / "logs" / "validation"
TRANSFORMATION_LOG_FILE = PROJECT_ROOT / "logs" / "transformation_log.json"


class PlanGenerator:
    def __init__(self, target_column: Optional[str] = None):
        self.target_column = target_column
        self.numeric_dtypes = {"int64", "float64", "int32", "float32", "int", "float"}

    def generate_plan(
        self,
        schema: Dict[str, Dict[str, Any]],
        dataset_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        validation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not schema:
            raise ValueError("Schema is empty.")

        target_column = self._resolve_target(schema)
        if not target_column:
            raise ValueError(
                f"Could not determine target column for dataset '{dataset_name}'. "
                f"Pass target_column manually."
            )

        numeric_columns = []
        categorical_columns = []

        impute_numeric_columns = []
        impute_categorical_columns = []

        one_hot_columns = []
        label_encode_columns = []

        scaling_columns = []
        log_transform_columns = []
        drop_columns = []

        for col_name, col_info in schema.items():
            if col_name == target_column:
                continue

            normalized_col = col_name.strip()
            dtype = str(col_info.get("dtype", "")).lower()
            missing_percentage = float(col_info.get("missing_percentage", 0) or 0)
            unique_count = int(col_info.get("unique_count", 0) or 0)
            min_value = col_info.get("min")
            max_value = col_info.get("max")
            mean_value = col_info.get("mean")

            if self._looks_like_id(normalized_col):
                drop_columns.append(col_name)
                continue

            if self._is_numeric(dtype):
                numeric_columns.append(col_name)
                scaling_columns.append(col_name)

                if missing_percentage > 0:
                    impute_numeric_columns.append(col_name)

                if self._should_log_transform(min_value, max_value, mean_value):
                    log_transform_columns.append(col_name)
            else:
                categorical_columns.append(col_name)

                if missing_percentage > 0:
                    impute_categorical_columns.append(col_name)

                if unique_count <= 10:
                    one_hot_columns.append(col_name)
                else:
                    label_encode_columns.append(col_name)

        task_type = self._infer_task_type(schema, target_column)

        plan = {
            "dataset_name": dataset_name,
            "target_column": target_column,
            "task_type": task_type,
            "pipeline_dependencies": {
                "ingestion": {
                    "used": metadata is not None,
                    "source_file": metadata.get("stored_path") if metadata else None,
                    "original_filename": metadata.get("original_filename") if metadata else None,
                    "stored_filename": metadata.get("stored_filename") if metadata else None,
                    "version_id": metadata.get("version_id") if metadata else None,
                    "rows": metadata.get("rows") if metadata else None,
                    "columns_count": metadata.get("columns_count") if metadata else None,
                },
                "schema_extraction": {
                    "used": True,
                    "schema_path": str(SCHEMA_DIR / f"{dataset_name}_schema.json"),
                    "columns_total": len(schema),
                },
                "validation": {
                    "used": validation is not None,
                    "is_valid": validation.get("is_valid") if validation else None,
                    "errors": validation.get("errors", []) if validation else [],
                    "validation_log_path": str(VALIDATION_LOG_DIR / f"{dataset_name}_validation.json"),
                },
                "transformation": {
                    "used": True,
                    "input_path": str(RAW_DIR / f"{dataset_name}.csv"),
                    "output_path": str(CURATED_DIR / f"{dataset_name}_cleaned.csv"),
                    "log_path": str(TRANSFORMATION_LOG_FILE),
                },
                "feature_engineering": {
                    "used": True,
                    "input_path": str(CURATED_DIR / f"{dataset_name}_cleaned.csv"),
                    "output_path": str(FEATURES_DIR / f"{dataset_name}_features.csv"),
                    "sample_output_path": str(FEATURES_SAMPLE_DIR / f"{dataset_name}_sample_50_features.csv"),
                },
            },
            "input_summary": {
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "columns_to_drop": drop_columns,
            },
            "preprocessing": {
                "drop_columns": drop_columns,
                "imputation": {
                    "numeric_strategy": "median",
                    "numeric_columns": impute_numeric_columns,
                    "categorical_strategy": "most_frequent",
                    "categorical_columns": impute_categorical_columns,
                },
                "encoding": {
                    "one_hot_columns": one_hot_columns,
                    "label_encoding_columns": label_encode_columns,
                },
                "scaling": {
                    "strategy": "standard",
                    "columns": scaling_columns,
                },
                "log_transform": log_transform_columns,
            },
            "split_strategy": {
                "test_size": 0.2,
                "random_state": 42,
                "stratify": task_type == "classification",
            },
            "smote": {
                "enabled": False,
                "reason": "Cannot infer class imbalance from schema only"
            },
            "export_paths": {
                "source_schema": str(SCHEMA_DIR / f"{dataset_name}_schema.json"),
                "curated_data": str(CURATED_DIR / f"{dataset_name}_cleaned.csv"),
                "features_data": str(FEATURES_DIR / f"{dataset_name}_features.csv"),
                "features_sample_data": str(FEATURES_SAMPLE_DIR / f"{dataset_name}_sample_50_features.csv"),
                "plan_json": str(PLAN_DIR / f"{dataset_name}_plan.json"),
            },
        }

        return plan

    def save_plan(self, plan: dict, dataset_name: str, output_folder: Optional[str] = None) -> str:
        output_dir = Path(output_folder) if output_folder else PLAN_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{dataset_name}_plan.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2)

        return str(output_path)

    def _resolve_target(self, schema: dict):
        if self.target_column and self.target_column in schema:
            return self.target_column

        strong_targets = [
            "target", "label", "class", "species", "price",
            "output", "y", "churn", "attrition", "cardio", "diagnosis"
        ]

        schema_keys = list(schema.keys())

        for col_name in schema_keys:
            if col_name.strip().lower() in strong_targets:
                return col_name

        non_id_columns = [col for col in schema_keys if not self._looks_like_id(col)]

        for col in reversed(non_id_columns):
            dtype = str(schema[col].get("dtype", "")).lower()
            unique_count = int(schema[col].get("unique_count", 0) or 0)

            if dtype not in self.numeric_dtypes:
                return col

            if unique_count <= 10:
                return col

        if non_id_columns:
            return non_id_columns[-1]

        return None

    def _infer_task_type(self, schema: dict, target_column: str) -> str:
        target_info = schema.get(target_column, {})
        dtype = str(target_info.get("dtype", "")).lower()
        unique_count = int(target_info.get("unique_count", 0) or 0)

        if not self._is_numeric(dtype):
            return "classification"

        if unique_count <= 10:
            return "classification"

        return "regression"

    def _is_numeric(self, dtype: str) -> bool:
        return dtype in self.numeric_dtypes

    def _looks_like_id(self, col_name: str) -> bool:
        lowered = col_name.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
        return lowered == "id" or lowered.endswith("id")

    def _should_log_transform(self, min_value, max_value, mean_value) -> bool:
        try:
            min_value = float(min_value)
            max_value = float(max_value)
            mean_value = float(mean_value)

            if min_value < 0:
                return False

            if mean_value <= 0:
                return False

            return (max_value / mean_value) >= 3
        except (TypeError, ValueError):
            return False


def load_json_file(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PLAN GENERATOR - T1.6")
    print("Generating preprocessing plans from pipeline artifacts")
    print("=" * 60 + "\n")

    PLAN_DIR.mkdir(parents=True, exist_ok=True)

    if not SCHEMA_DIR.exists():
        print("❌ data/schema folder not found.")
        raise SystemExit(1)

    schema_files = list(SCHEMA_DIR.glob("*_schema.json"))

    if not schema_files:
        print("❌ No schema JSON files found in data/schema/")
        raise SystemExit(1)

    print(f"📁 Found {len(schema_files)} schema files.\n")

    generator = PlanGenerator()

    for i, schema_file in enumerate(schema_files, 1):
        print(f"{i}. Processing: {schema_file.name}")

        try:
            dataset_name = schema_file.stem.replace("_schema", "")
            schema = load_json_file(schema_file)

            plan = generator.generate_plan(
                schema=schema,
                dataset_name=dataset_name,
                metadata=None,
                validation=None,
            )
            saved_path = generator.save_plan(plan, dataset_name)

            print(f"   ✅ Target column: {plan['target_column']}")
            print(f"   ✅ Task type: {plan['task_type']}")
            print(f"   ✅ Saved plan: {saved_path}")
            print(f"   ✅ Drop columns: {plan['preprocessing']['drop_columns']}")
            print(f"   ✅ Scale columns: {plan['preprocessing']['scaling']['columns']}")
            print()

        except Exception as e:
            print(f"   ❌ Error: {e}\n")

    print("=" * 60)
    print("✅ Plan generation complete!")
    print(f"📁 Check: {PLAN_DIR}")
    print("=" * 60)