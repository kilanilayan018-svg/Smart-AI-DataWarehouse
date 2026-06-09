"""
Schema-Plan Pair Generator for FLAN-T5 Finetuning - T1.7
=========================================================
Generates (input_prompt, output_plan) pairs from all pipeline artifacts.

Matches artifacts by version_id across:
  - metadata/          -> ingestion metadata
  - data/schema/       -> schema JSON
  - logs/validation/   -> validation report
  - logs/transform/    -> transformation log
  - metadata/plans/    -> preprocessing plan

Output:
  - data/finetuning/finetuning_pairs.jsonl   (primary - HuggingFace format)
  - data/finetuning/finetuning_pairs.csv     (secondary - human inspection)
  - data/finetuning/finetuning_summary.json  (stats + coverage report)

Augmentation:
  3 prompt variations per dataset => up to 30 pairs from 10 datasets
"""

import json
import csv
import os
import re
from pathlib import Path
from datetime import datetime


# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_DIR    = PROJECT_ROOT / "metadata"
SCHEMA_DIR      = PROJECT_ROOT / "data" / "schema"
VALIDATION_DIR  = PROJECT_ROOT / "logs" / "validation"
TRANSFORM_DIR   = PROJECT_ROOT / "logs" / "transform"
PLAN_DIR        = PROJECT_ROOT / "metadata" / "plans"
OUTPUT_DIR      = PROJECT_ROOT / "data" / "finetuning"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ARTIFACT LOADERS
# ============================================================

def load_json(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_ingestion_metadata(version_id: str) -> dict | None:
    """Find ingestion metadata JSON by version_id in filename."""
    for f in METADATA_DIR.glob("*_metadata.json"):
        if version_id in f.name:
            try:
                return load_json(f)
            except Exception:
                return None
    return None


def find_schema(version_id: str, dataset_name: str) -> dict | None:
    """
    Find schema JSON by version_id first, then fall back to dataset name match.
    Schema files are named like: {timestamp}_{version_id}_{dataset_name}_schema.json
    """
    # Try version_id match first
    for f in SCHEMA_DIR.glob("*_schema.json"):
        if version_id in f.name:
            try:
                return load_json(f)
            except Exception:
                return None

    # Fall back to dataset name match
    clean_name = dataset_name.lower().replace(" ", "").replace("_", "").replace("-", "")
    for f in SCHEMA_DIR.glob("*_schema.json"):
        candidate = f.stem.lower().replace(" ", "").replace("_", "").replace("-", "")
        if clean_name in candidate:
            try:
                return load_json(f)
            except Exception:
                return None
    return None


def find_validation_report(version_id: str, dataset_name: str) -> dict | None:
    """Find validation report by version_id or dataset name."""
    for f in VALIDATION_DIR.glob("*.json"):
        if f.name == "validation_summary.json":
            continue
        if version_id in f.name:
            try:
                return load_json(f)
            except Exception:
                return None

    # Fall back to dataset name
    clean_name = dataset_name.lower().replace(" ", "_")
    for f in VALIDATION_DIR.glob("*.json"):
        if f.name == "validation_summary.json":
            continue
        if clean_name in f.name.lower():
            try:
                return load_json(f)
            except Exception:
                return None
    return None


def find_transformation_log(version_id: str, dataset_name: str) -> dict | None:
    """Find transformation log by version_id or dataset name."""
    for f in TRANSFORM_DIR.glob("*.json"):
        if version_id in f.name:
            try:
                return load_json(f)
            except Exception:
                return None

    clean_name = dataset_name.lower().replace(" ", "_")
    for f in TRANSFORM_DIR.glob("*.json"):
        if clean_name in f.name.lower():
            try:
                return load_json(f)
            except Exception:
                return None
    return None


def find_plan(version_id: str, dataset_name: str) -> dict | None:
    """Find preprocessing plan by version_id or dataset name."""
    for f in PLAN_DIR.glob("*_plan.json"):
        if version_id in f.name:
            try:
                return load_json(f)
            except Exception:
                return None

    clean_name = dataset_name.lower().replace(" ", "_")
    for f in PLAN_DIR.glob("*_plan.json"):
        if clean_name in f.name.lower():
            try:
                return load_json(f)
            except Exception:
                return None
    return None


# ============================================================
# COMPREHENSIVE PLAN BUILDER
# ============================================================

def build_comprehensive_plan(
    metadata: dict,
    schema: dict,
    validation: dict | None,
    transformation: dict | None,
    plan: dict | None,
) -> dict:
    """
    Merge all pipeline artifacts into one comprehensive plan JSON.
    This is the TARGET output that FLAN-T5 will learn to generate.
    """

    meta_block = schema.get("_meta", {}) if schema else {}
    target_column = meta_block.get("target_column")
    target_detection = meta_block.get("target_detection_method", "unknown")

    # Build column summaries from schema (exclude _meta)
    columns_summary = []
    if schema:
        for col_name, col_info in schema.items():
            if col_name == "_meta":
                continue
            summary = {
                "name": col_name,
                "dtype": col_info.get("dtype"),
                "missing_percentage": col_info.get("missing_percentage", 0),
                "unique_count": col_info.get("unique_count"),
                "is_target": col_info.get("is_target", False),
            }
            if "min" in col_info:
                summary["min"] = col_info["min"]
                summary["max"] = col_info["max"]
                summary["mean"] = col_info["mean"]
            if "sample_values" in col_info:
                summary["sample_values"] = col_info["sample_values"]
            columns_summary.append(summary)

    # Infer task type from plan or schema
    task_type = "classification"
    if plan:
        task_type = plan.get("task_type", "classification")
    elif schema and target_column and target_column in schema:
        t_info = schema[target_column]
        dtype = str(t_info.get("dtype", "")).lower()
        unique_count = int(t_info.get("unique_count", 0) or 0)
        numeric_dtypes = {"int64", "float64", "int32", "float32"}
        if dtype in numeric_dtypes and unique_count > 10:
            task_type = "regression"

    # Build preprocessing steps from plan or derive from schema
    preprocessing = {}
    if plan:
        preprocessing = plan.get("preprocessing", {})
    else:
        # Derive basic preprocessing from schema
        numeric_cols = []
        categorical_cols = []
        drop_cols = []
        impute_numeric = []
        impute_categorical = []
        one_hot = []
        label_enc = []

        if schema:
            for col_name, col_info in schema.items():
                if col_name == "_meta":
                    continue
                if col_info.get("is_target"):
                    continue
                dtype = str(col_info.get("dtype", "")).lower()
                missing_pct = float(col_info.get("missing_percentage", 0) or 0)
                unique_count = int(col_info.get("unique_count", 0) or 0)
                col_lower = col_name.strip().lower().replace("_", "").replace("-", "")

                if col_lower == "id" or col_lower.endswith("id"):
                    drop_cols.append(col_name)
                    continue

                if dtype in {"int64", "float64", "int32", "float32"}:
                    numeric_cols.append(col_name)
                    if missing_pct > 0:
                        impute_numeric.append(col_name)
                else:
                    categorical_cols.append(col_name)
                    if missing_pct > 0:
                        impute_categorical.append(col_name)
                    if unique_count <= 10:
                        one_hot.append(col_name)
                    else:
                        label_enc.append(col_name)

        preprocessing = {
            "drop_columns": drop_cols,
            "imputation": {
                "numeric_strategy": "median",
                "numeric_columns": impute_numeric,
                "categorical_strategy": "most_frequent",
                "categorical_columns": impute_categorical,
            },
            "encoding": {
                "one_hot_columns": one_hot,
                "label_encoding_columns": label_enc,
            },
            "scaling": {
                "strategy": "standard",
                "columns": numeric_cols,
            },
        }

    # Validation summary
    validation_summary = {
        "performed": validation is not None,
        "is_valid": validation.get("is_valid") if validation else None,
        "errors": validation.get("errors", []) if validation else [],
        "validation_status": validation.get("validation_status") if validation else "unknown",
    }

    # Transformation summary
    transformation_summary = {
        "performed": transformation is not None,
        "status": transformation.get("status") if transformation else "unknown",
        "duplicates_removed": transformation.get("duplicates_removed", 0) if transformation else 0,
        "constant_columns_dropped": transformation.get("constant_columns_dropped", []) if transformation else [],
        "steps_applied": transformation.get("steps", []) if transformation else [],
        "original_shape": transformation.get("original_shape") if transformation else None,
        "final_shape": transformation.get("final_shape") if transformation else None,
    }

    comprehensive_plan = {
        # ── Ingestion metadata (exactly as you showed) ──────────────────
        "ingestion": {
            "original_filename": metadata.get("original_filename"),
            "stored_filename": metadata.get("stored_filename"),
            "stored_path": metadata.get("stored_path"),
            "version_id": metadata.get("version_id"),
            "timestamp": metadata.get("timestamp"),
            "file_type": metadata.get("file_type"),
            "rows": metadata.get("rows"),
            "columns_count": metadata.get("columns_count"),
            "columns": metadata.get("columns", []),
        },

        # ── Schema summary ───────────────────────────────────────────────
        "schema": {
            "target_column": target_column,
            "target_detection_method": target_detection,
            "task_type": task_type,
            "total_columns": meta_block.get("column_count", len(columns_summary)),
            "row_count": meta_block.get("row_count", metadata.get("rows")),
            "columns": columns_summary,
        },

        # ── Validation ───────────────────────────────────────────────────
        "validation": validation_summary,

        # ── Transformation ───────────────────────────────────────────────
        "transformation": transformation_summary,

        # ── Preprocessing plan ───────────────────────────────────────────
        "preprocessing_plan": {
            "target_column": target_column,
            "task_type": task_type,
            "input_summary": plan.get("input_summary", {}) if plan else {},
            "preprocessing": preprocessing,
            "split_strategy": plan.get("split_strategy", {
                "test_size": 0.2,
                "random_state": 42,
                "stratify": task_type == "classification",
            }) if plan else {},
            "smote": plan.get("smote", {
                "enabled": False,
                "reason": "Cannot infer class imbalance from schema only"
            }) if plan else {},
        },

        # ── Export paths ─────────────────────────────────────────────────
        "export_paths": plan.get("export_paths", {}) if plan else {},
    }

    return comprehensive_plan


# ============================================================
# PROMPT BUILDERS  (3 variations per dataset)
# ============================================================

def build_prompt_v1(metadata: dict, schema: dict) -> str:
    """
    Variation 1 — Full structured description.
    Most informative. Best for learning preprocessing decisions.
    """
    meta = schema.get("_meta", {}) if schema else {}
    target = meta.get("target_column", "unknown")
    rows = metadata.get("rows", "?")
    cols_count = metadata.get("columns_count", "?")
    filename = metadata.get("original_filename", "unknown")

    lines = [
        f"Generate a comprehensive preprocessing plan for the following dataset.",
        f"",
        f"Dataset: {filename}",
        f"Rows: {rows} | Columns: {cols_count}",
        f"Target column: {target}",
        f"",
        f"Column details:",
    ]

    if schema:
        for col_name, col_info in schema.items():
            if col_name == "_meta":
                continue
            dtype = col_info.get("dtype", "unknown")
            unique = col_info.get("unique_count", "?")
            missing_pct = col_info.get("missing_percentage", 0)
            is_target = col_info.get("is_target", False)

            tag = " [TARGET]" if is_target else ""
            missing_tag = f", {missing_pct}% missing" if missing_pct > 0 else ""

            if "min" in col_info:
                lines.append(
                    f"  - {col_name}{tag}: {dtype}, "
                    f"min={col_info['min']}, max={col_info['max']}, "
                    f"mean={round(col_info['mean'], 2)}, "
                    f"{unique} unique{missing_tag}"
                )
            else:
                samples = col_info.get("sample_values", [])
                sample_str = f", samples={samples[:3]}" if samples else ""
                lines.append(
                    f"  - {col_name}{tag}: {dtype}, "
                    f"{unique} unique{missing_tag}{sample_str}"
                )

    return "\n".join(lines)


def build_prompt_v2(metadata: dict, schema: dict) -> str:
    """
    Variation 2 — Concise stats-focused prompt.
    Emphasizes missing values and data quality for imputation decisions.
    """
    meta = schema.get("_meta", {}) if schema else {}
    target = meta.get("target_column", "unknown")
    filename = metadata.get("original_filename", "unknown")
    rows = metadata.get("rows", "?")

    numeric_cols = []
    categorical_cols = []
    cols_with_missing = []

    if schema:
        for col_name, col_info in schema.items():
            if col_name == "_meta":
                continue
            dtype = str(col_info.get("dtype", "")).lower()
            missing_pct = float(col_info.get("missing_percentage", 0) or 0)
            is_target = col_info.get("is_target", False)

            if is_target:
                continue

            if dtype in {"int64", "float64", "int32", "float32", "int", "float"}:
                numeric_cols.append(col_name)
            else:
                categorical_cols.append(col_name)

            if missing_pct > 0:
                cols_with_missing.append(f"{col_name} ({missing_pct}%)")

    lines = [
        f"Create a machine learning preprocessing plan.",
        f"",
        f"File: {filename} | Rows: {rows} | Target: {target}",
        f"Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols) if numeric_cols else 'none'}",
        f"Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols) if categorical_cols else 'none'}",
        f"Columns with missing values: {', '.join(cols_with_missing) if cols_with_missing else 'none'}",
    ]

    return "\n".join(lines)


def build_prompt_v3(metadata: dict, schema: dict) -> str:
    """
    Variation 3 — Task-type focused prompt.
    Emphasizes the prediction task and target variable properties.
    """
    meta = schema.get("_meta", {}) if schema else {}
    target = meta.get("target_column", "unknown")
    filename = metadata.get("original_filename", "unknown")
    rows = metadata.get("rows", "?")
    cols_count = metadata.get("columns_count", "?")

    # Get target info
    task_type = "classification"
    target_unique = "?"
    target_dtype = "unknown"
    target_samples = []

    if schema and target and target in schema:
        t_info = schema[target]
        target_dtype = str(t_info.get("dtype", "unknown"))
        target_unique = t_info.get("unique_count", "?")
        target_samples = t_info.get("sample_values", [])
        dtype_lower = target_dtype.lower()
        numeric_dtypes = {"int64", "float64", "int32", "float32"}
        if dtype_lower in numeric_dtypes and isinstance(target_unique, int) and target_unique > 10:
            task_type = "regression"

    col_names = [
        col for col in (schema.keys() if schema else [])
        if col != "_meta"
    ]

    lines = [
        f"Generate preprocessing steps for a {task_type} task.",
        f"",
        f"Dataset: {filename}",
        f"Shape: {rows} rows x {cols_count} columns",
        f"Prediction target: '{target}' (dtype={target_dtype}, {target_unique} unique values)",
    ]

    if target_samples:
        lines.append(f"Target class samples: {target_samples[:5]}")

    lines += [
        f"",
        f"All feature columns: {', '.join([c for c in col_names if c != target])}",
    ]

    return "\n".join(lines)


PROMPT_BUILDERS = [build_prompt_v1, build_prompt_v2, build_prompt_v3]
PROMPT_LABELS   = ["full_structured", "stats_focused", "task_focused"]


# ============================================================
# PAIR GENERATOR
# ============================================================

def generate_pairs() -> list[dict]:
    """
    Main function. Iterates over all ingestion metadata files,
    finds matching artifacts, builds comprehensive plans,
    and generates 3 prompt variations per dataset.
    """
    pairs = []
    coverage_report = []

    metadata_files = sorted(METADATA_DIR.glob("*_metadata.json"))

    if not metadata_files:
        print("❌ No metadata files found in metadata/")
        return pairs

    print(f"📁 Found {len(metadata_files)} ingestion metadata files\n")

    for i, meta_file in enumerate(metadata_files, 1):
        print(f"{'=' * 65}")
        print(f"[{i}/{len(metadata_files)}] Processing: {meta_file.name}")

        try:
            metadata = load_json(meta_file)
        except Exception as e:
            print(f"   ❌ Could not load metadata: {e}")
            continue

        version_id   = metadata.get("version_id", "")
        dataset_name = metadata.get("original_filename", "").replace(".csv", "").replace(".xlsx", "")

        print(f"   📌 Version ID:   {version_id}")
        print(f"   📌 Dataset name: {dataset_name}")

        # ── Find all artifacts ───────────────────────────────────────
        schema         = find_schema(version_id, dataset_name)
        validation     = find_validation_report(version_id, dataset_name)
        transformation = find_transformation_log(version_id, dataset_name)
        plan           = find_plan(version_id, dataset_name)

        coverage = {
            "dataset": dataset_name,
            "version_id": version_id,
            "schema_found":         schema is not None,
            "validation_found":     validation is not None,
            "transformation_found": transformation is not None,
            "plan_found":           plan is not None,
        }
        coverage_report.append(coverage)

        print(f"   {'✅' if schema         else '⚠️ '} Schema:         {'found' if schema         else 'NOT FOUND'}")
        print(f"   {'✅' if validation     else '⚠️ '} Validation:     {'found' if validation     else 'NOT FOUND'}")
        print(f"   {'✅' if transformation else '⚠️ '} Transformation: {'found' if transformation else 'NOT FOUND'}")
        print(f"   {'✅' if plan           else '⚠️ '} Plan:           {'found' if plan           else 'NOT FOUND (will derive)'}")

        # Schema is required — skip if missing
        if schema is None:
            print(f"   ❌ Skipping: schema is required but not found\n")
            continue

        # ── Build comprehensive plan (TARGET output) ─────────────────
        try:
            comprehensive_plan = build_comprehensive_plan(
                metadata=metadata,
                schema=schema,
                validation=validation,
                transformation=transformation,
                plan=plan,
            )
        except Exception as e:
            print(f"   ❌ Could not build comprehensive plan: {e}")
            continue

        plan_str = json.dumps(comprehensive_plan, indent=2, ensure_ascii=False)

        # ── Generate 3 prompt variations ─────────────────────────────
        generated = 0
        for builder, label in zip(PROMPT_BUILDERS, PROMPT_LABELS):
            try:
                prompt = builder(metadata, schema)

                pair = {
                    "id":               f"{version_id}_{label}",
                    "dataset":          dataset_name,
                    "version_id":       version_id,
                    "prompt_variation": label,
                    "input":            prompt,
                    "output":           plan_str,
                }
                pairs.append(pair)
                generated += 1

            except Exception as e:
                print(f"   ⚠️  Could not generate prompt '{label}': {e}")

        print(f"   ✅ Generated {generated} pairs\n")

    return pairs, coverage_report


# ============================================================
# SAVERS
# ============================================================

def save_jsonl(pairs: list[dict], output_path: Path):
    """Save as JSONL — one JSON object per line (HuggingFace format)."""
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            # HuggingFace expects: {"input": "...", "output": "..."}
            hf_pair = {
                "id":               pair["id"],
                "dataset":          pair["dataset"],
                "version_id":       pair["version_id"],
                "prompt_variation": pair["prompt_variation"],
                "input":            pair["input"],
                "output":           pair["output"],
            }
            f.write(json.dumps(hf_pair, ensure_ascii=False) + "\n")
    print(f"   ✅ JSONL saved: {output_path.name}")


def save_csv(pairs: list[dict], output_path: Path):
    """Save as CSV for human inspection."""
    if not pairs:
        return
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "dataset", "version_id",
                                                "prompt_variation", "input", "output"])
        writer.writeheader()
        writer.writerows(pairs)
    print(f"   ✅ CSV saved:  {output_path.name}")


def save_summary(pairs: list[dict], coverage: list[dict], output_path: Path):
    """Save a summary report."""
    datasets_covered = len(set(p["dataset"] for p in pairs))
    variations_count = {}
    for p in pairs:
        variations_count[p["prompt_variation"]] = variations_count.get(p["prompt_variation"], 0) + 1

    summary = {
        "generated_at":        datetime.now().isoformat(),
        "total_pairs":         len(pairs),
        "datasets_processed":  len(coverage),
        "datasets_with_pairs": datasets_covered,
        "pairs_per_variation": variations_count,
        "coverage_report":     coverage,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Summary saved: {output_path.name}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("SCHEMA-PLAN PAIR GENERATOR FOR FLAN-T5 FINETUNING - T1.7")
    print("=" * 65)
    print(f"📁 Metadata dir:    {METADATA_DIR}")
    print(f"📁 Schema dir:      {SCHEMA_DIR}")
    print(f"📁 Validation dir:  {VALIDATION_DIR}")
    print(f"📁 Transform dir:   {TRANSFORM_DIR}")
    print(f"📁 Plan dir:        {PLAN_DIR}")
    print(f"📁 Output dir:      {OUTPUT_DIR}")
    print("=" * 65 + "\n")

    # ── Check required directories ───────────────────────────────────
    missing_dirs = []
    for d, name in [
        (METADATA_DIR,   "metadata/"),
        (SCHEMA_DIR,     "data/schema/"),
    ]:
        if not d.exists():
            missing_dirs.append(name)

    if missing_dirs:
        print(f"❌ Required directories not found: {missing_dirs}")
        raise SystemExit(1)

    # ── Generate pairs ───────────────────────────────────────────────
    pairs, coverage_report = generate_pairs()

    if not pairs:
        print("❌ No pairs generated. Check that schemas exist in data/schema/")
        raise SystemExit(1)

    # ── Save outputs ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SAVING OUTPUTS")
    print("=" * 65)

    save_jsonl(pairs,    OUTPUT_DIR / "finetuning_pairs.jsonl")
    save_csv(pairs,      OUTPUT_DIR / "finetuning_pairs.csv")
    save_summary(pairs, coverage_report, OUTPUT_DIR / "finetuning_summary.json")

    # ── Final report ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("✅ PAIR GENERATION COMPLETE!")
    print("=" * 65)
    print(f"   📊 Total pairs generated:  {len(pairs)}")
    print(f"   📊 Datasets covered:       {len(set(p['dataset'] for p in pairs))}")
    print(f"   📊 Prompt variations:      {len(PROMPT_BUILDERS)}")
    print(f"\n   📁 Output folder: {OUTPUT_DIR}")
    print(f"      - finetuning_pairs.jsonl  ← use this for HuggingFace training")
    print(f"      - finetuning_pairs.csv    ← use this for inspection")
    print(f"      - finetuning_summary.json ← coverage + stats report")

    # ── Coverage table ───────────────────────────────────────────────
    print("\n📋 ARTIFACT COVERAGE PER DATASET:")
    print("-" * 65)
    print(f"  {'Dataset':<35} {'Schema':^8} {'Valid':^8} {'Trans':^8} {'Plan':^8}")
    print("-" * 65)
    for c in coverage_report:
        print(
            f"  {c['dataset'][:35]:<35} "
            f"{'✅' if c['schema_found']         else '❌':^8} "
            f"{'✅' if c['validation_found']     else '❌':^8} "
            f"{'✅' if c['transformation_found'] else '❌':^8} "
            f"{'✅' if c['plan_found']           else '❌':^8}"
        )
    print("=" * 65)
