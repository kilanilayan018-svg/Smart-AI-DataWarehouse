"""
Feature Engineering Module - T1.5
Reads all versioned curated files from data/curated/
Excludes target column from encoding/scaling using schema _meta
Saves full feature files to data/features/

FIXES:
- Target column read from schema _meta block (not re-detected)
- Target excluded from scaling and encoding, label-encoded separately
- Verification block fixed for pandas 2/3 StringDtype compatibility
- Scaling failure on string columns handled gracefully
"""

from pathlib import Path
import argparse
import traceback
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "curated"
OUTPUT_DIR = PROJECT_ROOT / "data" / "features"
SAMPLE_OUTPUT_DIR = PROJECT_ROOT / "data" / "features_samples"
SCHEMA_DIR = PROJECT_ROOT / "data" / "schema"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def get_cleaned_files(input_dir: Path):
    return sorted(input_dir.glob("*_cleaned.csv"))


def parse_dataset_name(file_path: Path) -> str:
    """
    Extract original dataset name from versioned filename.
    e.g. 20260416_173528_20aa9739_customer churn_cleaned.csv
      -> customer churn
    """
    stem = file_path.stem
    if stem.endswith("_cleaned"):
        stem = stem[:-len("_cleaned")]
    parts = stem.split("_", 3)
    if len(parts) == 4:
        return parts[3]
    return stem


def find_schema_for_file(file_path: Path) -> dict | None:
    """
    Find the matching schema JSON for a curated file.
    Matches by version_id embedded in filename, then falls back to dataset name.
    """
    stem = file_path.stem
    if stem.endswith("_cleaned"):
        stem = stem[:-len("_cleaned")]

    # Try exact versioned filename match first
    for schema_file in SCHEMA_DIR.glob("*_schema.json"):
        schema_stem = schema_file.stem.replace("_schema", "")
        if schema_stem == stem:
            try:
                with open(schema_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None

    # Fall back: match by dataset name substring
    dataset_name = parse_dataset_name(file_path)
    clean_name = dataset_name.lower().replace(" ", "").replace("_", "").replace("-", "")

    for schema_file in SCHEMA_DIR.glob("*_schema.json"):
        candidate = schema_file.stem.lower().replace(" ", "").replace("_", "").replace("-", "")
        if clean_name in candidate:
            try:
                with open(schema_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
    return None


def get_target_column(schema: dict | None) -> str | None:
    """Extract target column name from schema _meta block."""
    if schema is None:
        return None
    meta = schema.get("_meta", {})
    return meta.get("target_column")


def detect_id_columns(df: pd.DataFrame) -> list:
    id_cols = []
    for col in df.columns:
        normalized = str(col).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
        if normalized == "id" or normalized.endswith("id"):
            id_cols.append(col)
        elif normalized in ["index", "employeenumber", "imdbtitleid"]:
            id_cols.append(col)
        elif df[col].nunique() == len(df) and len(df) > 100:
            id_cols.append(col)
    return id_cols


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(0 if pd.isna(median_val) else median_val)

    for col in df.columns:
        if df[col].dtype == "object" or str(df[col].dtype) == "string":
            if df[col].isnull().any():
                df[col] = df[col].fillna("unknown")

    return df


def replace_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(0)
    return df


def split_column_types(df: pd.DataFrame, exclude_cols: list) -> tuple:
    """
    Split columns into categorical and numeric, excluding target and ID columns.
    """
    categorical_cols = []
    numeric_cols = []

    for col in df.columns:
        if col in exclude_cols:
            continue
        dtype_str = str(df[col].dtype)
        if dtype_str == "object" or dtype_str == "string" or "String" in dtype_str:
            categorical_cols.append(col)
        elif df[col].nunique() <= 10:
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)

    return categorical_cols, numeric_cols


def encode_categorical_columns(df: pd.DataFrame, categorical_cols: list) -> tuple:
    encoding_log = {}
    for col in categorical_cols:
        try:
            df[col] = df[col].astype(str).fillna("unknown")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoding_log[col] = dict(zip(
                le.classes_.tolist(),
                range(len(le.classes_))
            ))
        except Exception as e:
            print(f"      ⚠️  Could not encode '{col}': {e}")
    return df, encoding_log


def scale_numeric_columns(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    if not numeric_cols:
        return df

    # Only scale columns that are actually numeric — skip any that slipped through
    truly_numeric = []
    for col in numeric_cols:
        try:
            pd.to_numeric(df[col], errors="raise")
            truly_numeric.append(col)
        except Exception:
            print(f"      ⚠️  Skipping non-numeric column for scaling: '{col}'")

    if not truly_numeric:
        return df

    try:
        for col in truly_numeric:
            if df[col].isnull().any():
                df[col] = df[col].fillna(0)
        scaler = StandardScaler()
        df[truly_numeric] = scaler.fit_transform(df[truly_numeric])
        print(f"   📏 Scaled: {len(truly_numeric)} numeric columns")
    except Exception as e:
        print(f"      ⚠️  Scaling failed: {e}")
    return df


def encode_target_column(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Label-encode target if it is non-numeric (no scaling applied).
    """
    if target_col not in df.columns:
        return df
    dtype_str = str(df[target_col].dtype)
    if dtype_str == "object" or dtype_str == "string" or "String" in dtype_str:
        try:
            le = LabelEncoder()
            df[target_col] = le.fit_transform(df[target_col].astype(str))
            mapping = dict(zip(le.classes_, range(len(le.classes_))))
            print(f"   🎯 Target '{target_col}' encoded: {dict(list(mapping.items())[:5])}")
        except Exception as e:
            print(f"   ⚠️  Could not encode target '{target_col}': {e}")
    return df


def is_column_numeric(series: pd.Series) -> bool:
    """Pandas 2/3 safe numeric check."""
    try:
        return pd.api.types.is_numeric_dtype(series)
    except Exception:
        return False


# ============================================================
# MAIN PROCESS
# ============================================================

def process_file(file_path: Path, sample_mode: bool = False, sample_rows: int = 50):
    print(f"\n{'=' * 70}")
    print(f"📊 Processing: {file_path.name}")
    print(f"{'=' * 70}")

    # --- Load ---
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"   📂 Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"   ❌ Could not read file: {e}")
        return None

    # --- Load schema & get target ---
    schema = find_schema_for_file(file_path)
    target_col = get_target_column(schema)

    if target_col and target_col in df.columns:
        print(f"   🎯 Target column: '{target_col}' (from schema)")
    else:
        print(f"   ⚠️  No target column found in schema, proceeding without one")
        target_col = None

    # --- Drop ID columns (never drop target) ---
    id_cols = detect_id_columns(df)
    id_cols = [c for c in id_cols if c != target_col]
    if id_cols:
        df = df.drop(columns=id_cols)
        print(f"   🗑️  Dropped ID columns: {id_cols}")

    # --- Drop constant columns (never drop target) ---
    constant_cols = [
        col for col in df.columns
        if col != target_col and df[col].nunique() <= 1
    ]
    if constant_cols:
        df = df.drop(columns=constant_cols)
        print(f"   🗑️  Dropped constant columns: {constant_cols}")

    # --- Handle missing & infinite values ---
    df = handle_missing_values(df)
    df = replace_infinite_values(df)

    # --- Split features vs target ---
    exclude = [target_col] if target_col else []
    categorical_cols, numeric_cols = split_column_types(df, exclude_cols=exclude)

    print(f"   🏷️  Categorical feature columns: {len(categorical_cols)}")
    print(f"   🔢 Numeric feature columns: {len(numeric_cols)}")

    # --- Encode features ---
    df, encoding_log = encode_categorical_columns(df, categorical_cols)
    for col, mapping in encoding_log.items():
        short_mapping = dict(list(mapping.items())[:5])
        print(f"      🔄 Encoded '{col}': {short_mapping}")

    # --- Scale features ---
    df = scale_numeric_columns(df, numeric_cols)

    # --- Encode target separately (label only, no scaling) ---
    if target_col:
        df = encode_target_column(df, target_col)

    # --- Final cleanup ---
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)

    # --- Save ---
    dataset_name = parse_dataset_name(file_path)
    output_name = f"{dataset_name}_features.csv"
    output_path = OUTPUT_DIR / output_name

    try:
        df.to_csv(output_path, index=False)
        print(f"\n   ✅ Saved: {output_name}")
        print(f"      📊 Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"      🔍 Nulls remaining: {df.isnull().any().any()}")
        # Pandas 2/3 safe inf check
        numeric_df = df.select_dtypes(include=[np.number])
        has_inf = bool(np.isinf(numeric_df.values).any()) if len(numeric_df.columns) > 0 else False
        print(f"      🔍 Infs remaining:  {has_inf}")
    except Exception as e:
        print(f"   ❌ Could not save: {e}")
        return None

    # --- Optional sample ---
    sample_path = None
    if sample_mode:
        sample_df = df.head(sample_rows)
        sample_name = f"{dataset_name}_sample_{sample_rows}_features.csv"
        sample_path = SAMPLE_OUTPUT_DIR / sample_name
        sample_df.to_csv(sample_path, index=False)
        print(f"   ✅ Saved sample: {sample_name}")

    return {
        "output_path": str(output_path),
        "sample_path": str(sample_path) if sample_path else None,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
    }


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Feature Engineering Module")
    parser.add_argument("--sample", action="store_true",
                        help="Also save small sample outputs")
    parser.add_argument("--sample-rows", type=int, default=50,
                        help="Number of rows for sample output")
    args = parser.parse_args()

    print("=" * 70)
    print(" FEATURE ENGINEERING MODULE - T1.5")
    print("=" * 70)
    print(f"📁 Input:   {INPUT_DIR}")
    print(f"📁 Output:  {OUTPUT_DIR}")
    print(f"📁 Schema:  {SCHEMA_DIR}")

    if not INPUT_DIR.exists():
        print(f"\n❌ Input directory does not exist: {INPUT_DIR}")
        return

    files = get_cleaned_files(INPUT_DIR)
    print(f"\n📁 Found {len(files)} datasets to process\n")

    if not files:
        print("❌ No cleaned CSV files found!")
        return

    successful = 0
    failed = 0
    failed_files = []

    for file_path in files:
        try:
            result = process_file(
                file_path,
                sample_mode=args.sample,
                sample_rows=args.sample_rows
            )
            if result is not None:
                successful += 1
            else:
                failed += 1
                failed_files.append(file_path.name)
        except Exception as e:
            print(f"\n   ❌ UNEXPECTED ERROR processing {file_path.name}: {e}")
            traceback.print_exc()
            failed += 1
            failed_files.append(file_path.name)

    # --- Summary ---
    print("\n" + "=" * 70)
    print(" FEATURE ENGINEERING COMPLETE!")
    print("=" * 70)
    print(f"   ✅ Successful: {successful}/{len(files)}")
    print(f"   ❌ Failed:     {failed}/{len(files)}")

    if failed_files:
        print(f"\n   Failed files:")
        for f in failed_files:
            print(f"      - {f}")

    print(f"\n📁 Output folder: {OUTPUT_DIR}")

    # --- Verification (pandas 2/3 safe) ---
    print("\n📊 FEATURE FILES CREATED:")
    print("-" * 50)
    for f in sorted(OUTPUT_DIR.glob("*_features.csv")):
        try:
            full_df = pd.read_csv(f)
            # Pandas 2/3 safe: use api.types per column
            all_numeric = all(
                pd.api.types.is_numeric_dtype(full_df[col])
                for col in full_df.columns
            )
            no_nulls = not full_df.isnull().any().any()
            print(f"   ✅ {f.name}")
            print(f"      → {full_df.shape[0]} rows, {full_df.shape[1]} columns")
            print(f"      → All numeric: {all_numeric}")
            print(f"      → No nulls:    {no_nulls}")
        except Exception as e:
            print(f"   ❌ Could not verify {f.name}: {e}")

    print("\n" + "=" * 70)
    print("🎉 T1.5 COMPLETE — READY FOR MACHINE LEARNING!")
    print("=" * 70)


if __name__ == "__main__":
    main()