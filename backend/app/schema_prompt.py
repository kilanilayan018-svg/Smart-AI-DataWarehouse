from typing import Any


def schema_to_model_prompt(schema: dict[str, Any], dataset_name: str, target_column: str | None = None) -> str:
    """Convert the pipeline schema JSON into the prompt format used by DeepSeek fine-tuning.

    Supports both schema formats:
    1. {"columns": [...], "_meta": {...}}
    2. {"Age": {"dtype": ...}, "Sex": {...}, "_meta": {...}}
    """
    meta = schema.get("_meta", {}) if isinstance(schema.get("_meta"), dict) else {}

    columns = schema.get("columns")
    if isinstance(columns, dict):
        iterable_columns = [{"name": name, **(info if isinstance(info, dict) else {})} for name, info in columns.items()]
    elif isinstance(columns, list):
        iterable_columns = columns
    else:
        iterable_columns = []
        for name, info in schema.items():
            if name == "_meta":
                continue
            if isinstance(info, dict):
                iterable_columns.append({"name": name, **info})

    row_count = schema.get("row_count") or schema.get("rows") or meta.get("row_count")
    col_count = schema.get("column_count") or schema.get("columns_count") or meta.get("column_count") or len(iterable_columns)
    detected_target = target_column or schema.get("target_column") or schema.get("detected_target") or meta.get("target_column")
    task_type = schema.get("task_type") or "classification or regression"

    feature_names: list[str] = []
    detail_lines: list[str] = []
    for col in iterable_columns:
        if not isinstance(col, dict):
            continue
        name = col.get("name") or col.get("column") or col.get("column_name")
        if not name:
            continue
        if name != detected_target:
            feature_names.append(str(name))
        dtype = col.get("dtype") or col.get("type") or col.get("data_type") or "unknown"
        missing = col.get("missing_percentage", col.get("missing_pct", col.get("missing_percent", col.get("missing", 0))))
        unique = col.get("unique_values", col.get("unique_count", col.get("nunique", "unknown")))
        mean = col.get("mean")
        extra = f", mean={mean}" if mean is not None else ""
        detail_lines.append(f"  - {name}: {dtype}, {unique} unique, {missing}% missing{extra}")

    prompt = [
        "Generate preprocessing steps for a machine learning task.",
        "",
        f"Dataset: {dataset_name}",
        f"Shape: {row_count or 'unknown'} rows x {col_count or 'unknown'} columns",
        f"Prediction target: '{detected_target or 'auto-detect'}'",
        "",
        f"All feature columns: {', '.join(feature_names) if feature_names else 'unknown'}",
        "",
        "Column details:",
        *detail_lines[:80],
    ]
    return "\n".join(prompt)
