"""
Plan Executor
Reads the plan JSON produced by the model or PlanGenerator fallback
and applies every preprocessing step to the raw DataFrame in the
correct order, saving the result to data/curated/.

Execution order (order matters — do NOT change):
    1. drop_columns
    2. imputation   (must happen before encoding/scaling)
    3. log_transform (before scaling so scaler sees log values)
    4. encoding     (label encode first, then one-hot)
    5. scaling

Safe by design:
    - Every column name is validated against the real DataFrame before use.
    - Hallucinated / typo column names from the model are silently skipped.
    - Target column is never touched by any preprocessing step.
    - If the whole executor fails the raw file is returned as-is so the
      rest of the pipeline never crashes.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

log = logging.getLogger("smart_ai_dw_api")

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # backend/app/ -> backend/ -> project root
CURATED_DIR  = PROJECT_ROOT / "data" / "curated"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _real_cols(df: pd.DataFrame, candidates: Any, exclude: list[str] = []) -> list[str]:
    """
    Filter a list of column names to only those that actually exist in df
    and are not in the exclude list.
    Silently drops hallucinated / typo names from the model.
    """
    if not isinstance(candidates, list):
        return []
    valid = set(df.columns)
    excluded = set(exclude)
    seen = []
    for c in candidates:
        if isinstance(c, str) and c in valid and c not in excluded and c not in seen:
            seen.append(c)
    return seen


def _numeric_cols_with_missing(df: pd.DataFrame, exclude: list[str]) -> list[str]:
    return [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude and df[c].isnull().any()
    ]


def _categorical_cols_with_missing(df: pd.DataFrame, exclude: list[str]) -> list[str]:
    return [
        c for c in df.select_dtypes(exclude=[np.number]).columns
        if c not in exclude and df[c].isnull().any()
    ]


# ── Step functions ────────────────────────────────────────────────────────────

def _step_drop(df: pd.DataFrame, drop_columns: Any, target: str) -> pd.DataFrame:
    cols = _real_cols(df, drop_columns, exclude=[target])
    if cols:
        log.info(f"[executor] dropping: {cols}")
        df = df.drop(columns=cols)
    return df


def _step_impute(df: pd.DataFrame, imputation: Any, target: str) -> pd.DataFrame:
    if not isinstance(imputation, dict):
        return df

    numeric_strategy   = imputation.get("numeric_strategy", "median")
    categorical_strategy = imputation.get("categorical_strategy", "most_frequent")

    # Column lists — use from plan if provided, otherwise infer from schema
    raw_numeric_cols    = imputation.get("numeric_columns")
    raw_categorical_cols = imputation.get("categorical_columns")

    numeric_cols = (
        _real_cols(df, raw_numeric_cols, exclude=[target])
        if isinstance(raw_numeric_cols, list)
        else _numeric_cols_with_missing(df, exclude=[target])
    )

    categorical_cols = (
        _real_cols(df, raw_categorical_cols, exclude=[target])
        if isinstance(raw_categorical_cols, list)
        else _categorical_cols_with_missing(df, exclude=[target])
    )

    for col in numeric_cols:
        if df[col].isnull().any():
            fill = df[col].median() if numeric_strategy == "median" else df[col].mean()
            df[col] = df[col].fillna(fill)
            log.info(f"[executor] imputed numeric '{col}' with {numeric_strategy}={fill:.4f}")

    for col in categorical_cols:
        if df[col].isnull().any():
            if categorical_strategy == "most_frequent":
                fill = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            else:
                fill = imputation.get("constant_value", "Unknown")
            df[col] = df[col].fillna(fill)
            log.info(f"[executor] imputed categorical '{col}' with '{fill}'")

    return df


def _step_log_transform(df: pd.DataFrame, log_transform: Any, target: str) -> pd.DataFrame:
    cols = _real_cols(df, log_transform, exclude=[target])
    for col in cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Only apply if all values are >= 0 (log1p requires non-negative)
            if df[col].min() >= 0:
                df[col] = np.log1p(df[col])
                log.info(f"[executor] log1p transform applied to '{col}'")
            else:
                log.warning(f"[executor] skipped log transform for '{col}' — negative values present")
    return df


def _step_encode(df: pd.DataFrame, encoding: Any, target: str) -> pd.DataFrame:
    if not isinstance(encoding, dict):
        return df

    from sklearn.preprocessing import LabelEncoder

    label_cols  = _real_cols(df, encoding.get("label_encoding_columns", []), exclude=[target])
    one_hot_cols = _real_cols(df, encoding.get("one_hot_columns", []), exclude=[target])

    # Label encode first (adds no new columns, safe before one-hot)
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        log.info(f"[executor] label encoded '{col}'")

    # One-hot encode
    if one_hot_cols:
        df = pd.get_dummies(df, columns=one_hot_cols, drop_first=False)
        log.info(f"[executor] one-hot encoded: {one_hot_cols}")

    return df


def _step_scale(df: pd.DataFrame, scaling: Any, target: str) -> pd.DataFrame:
    if not isinstance(scaling, dict):
        return df

    strategy = scaling.get("strategy", "standard")
    if strategy == "none":
        return df

    cols = _real_cols(df, scaling.get("columns", []), exclude=[target])
    # After one-hot encoding, some columns may no longer exist — filter again
    cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    if not cols:
        return df

    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    scaler = StandardScaler() if strategy == "standard" else MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    log.info(f"[executor] scaled {len(cols)} columns with {strategy}")

    return df


# ── Public entry point ────────────────────────────────────────────────────────

def execute_plan(
    plan: dict[str, Any],
    raw_file_path: str | Path,
    dataset_name: str,
) -> tuple[str | None, str]:
    """
    Apply the plan to the raw CSV and save to data/curated/.

    Returns:
        (curated_path, status_message)
        curated_path is None if execution failed.
    """
    try:
        raw_path = Path(raw_file_path)
        if not raw_path.exists():
            return None, f"Raw file not found: {raw_file_path}"

        # Read raw file
        suffix = raw_path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(raw_path, sep=None, engine="python", on_bad_lines="skip")
        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(raw_path)
        else:
            return None, f"Unsupported file type: {suffix}"

        log.info(f"[executor] loaded {df.shape[0]} rows x {df.shape[1]} cols from {raw_path.name}")

        # Resolve preprocessing block — handles both model output shape and fallback shape
        preprocessing = plan.get("preprocessing", {})
        if not isinstance(preprocessing, dict):
            preprocessing = {}

        target = plan.get("target_column") or preprocessing.get("target_column", "")

        # ── Execute steps in order ────────────────────────────────────────────
        df = _step_drop(df, preprocessing.get("drop_columns", []), target)
        df = _step_impute(df, preprocessing.get("imputation", {}), target)
        df = _step_log_transform(df, preprocessing.get("log_transform", []), target)
        df = _step_encode(df, preprocessing.get("encoding", {}), target)
        df = _step_scale(df, preprocessing.get("scaling", {}), target)

        # ── Save curated CSV ──────────────────────────────────────────────────
        CURATED_DIR.mkdir(parents=True, exist_ok=True)
        curated_filename = f"{dataset_name}_curated.csv"
        curated_path = CURATED_DIR / curated_filename

        df.to_csv(curated_path, index=False)
        log.info(f"[executor] saved curated CSV: {curated_path} ({df.shape[0]} rows x {df.shape[1]} cols)")

        return str(curated_path), "success"

    except Exception as e:
        log.error(f"[executor] plan execution failed: {e}")
        return None, str(e)
