"""
Feature Engineering Module - T1.5
Author: Maha Qaddoumi
Refactored for team-friendly GitHub workflow

This module:
1. Reads curated CSV files from data/curated/
2. Drops ID columns and constant columns
3. Encodes categorical columns to numbers
4. Scales numeric columns
5. Saves full feature files to data/features/
6. Optionally saves small sample outputs for GitHub-safe sharing
"""

from pathlib import Path
import argparse
import traceback

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "curated"
OUTPUT_DIR = PROJECT_ROOT / "data" / "features"
SAMPLE_OUTPUT_DIR = PROJECT_ROOT / "data" / "features_samples"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_cleaned_files(input_dir: Path):
    return sorted(input_dir.glob("*_cleaned.csv"))


def detect_id_columns(df: pd.DataFrame):
    id_patterns = [
        "id", "customerid", "employeenumber", "index", "imbd title id"
    ]
    id_cols = []

    for col in df.columns:
        normalized = str(col).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
        if any(pattern in normalized for pattern in id_patterns):
            id_cols.append(col)
        elif df[col].nunique() == len(df) and len(df) > 100:
            id_cols.append(col)

    return id_cols


def handle_missing_values(df: pd.DataFrame):
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0
            df[col] = df[col].fillna(median_val)

    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna("unknown")

    return df


def replace_infinite_values(df: pd.DataFrame):
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(0)
    return df


def split_column_types(df: pd.DataFrame):
    categorical_cols = []
    numeric_cols = []

    for col in df.columns:
        if df[col].dtype == "object":
            categorical_cols.append(col)
        elif df[col].nunique() <= 10:
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)

    return categorical_cols, numeric_cols


def encode_categorical_columns(df: pd.DataFrame, categorical_cols):
    encoding_log = {}

    for col in categorical_cols:
        df[col] = df[col].astype(str).fillna("unknown")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoding_log[col] = dict(zip(le.classes_, range(len(le.classes_))))

    return df, encoding_log


def scale_numeric_columns(df: pd.DataFrame, numeric_cols):
    if numeric_cols:
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(0)

        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df


def process_file(file_path: Path, sample_mode: bool = False, sample_rows: int = 50):
    print(f"\n{'=' * 70}")
    print(f"📊 Processing: {file_path.name}")
    print(f"{'=' * 70}")

    df = pd.read_csv(file_path)
    print(f"   📂 Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    id_cols = detect_id_columns(df)
    if id_cols:
        df = df.drop(columns=id_cols)
        print(f"   🗑️ Dropped ID columns: {id_cols}")

    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)
        print(f"   🗑️ Dropped constant columns: {constant_cols}")

    df = handle_missing_values(df)
    df = replace_infinite_values(df)

    categorical_cols, numeric_cols = split_column_types(df)
    print(f"   🏷️ Categorical columns: {len(categorical_cols)}")
    print(f"   🔢 Numeric columns: {len(numeric_cols)}")

    df, encoding_log = encode_categorical_columns(df, categorical_cols)
    for col, mapping in encoding_log.items():
        short_mapping = dict(list(mapping.items())[:5])
        print(f"      🔄 Encoded '{col}': {short_mapping}")

    df = scale_numeric_columns(df, numeric_cols)
    if numeric_cols:
        print(f"   📏 Scaled: {len(numeric_cols)} numeric columns")

    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)

    output_name = file_path.name.replace("_cleaned.csv", "_features.csv")
    output_path = OUTPUT_DIR / output_name
    df.to_csv(output_path, index=False)

    print(f"\n   ✅ SAVED FULL: {output_name}")
    print(f"      📊 Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"      🔍 Nulls: {df.isnull().any().any()}")
    print(f"      🔍 Infs: {np.isinf(df).any().any()}")

    sample_path = None
    if sample_mode:
        sample_df = df.head(sample_rows)
        sample_name = output_name.replace("_features.csv", f"_sample_{sample_rows}_features.csv")
        sample_path = SAMPLE_OUTPUT_DIR / sample_name
        sample_df.to_csv(sample_path, index=False)
        print(f"   ✅ SAVED SAMPLE: {sample_name}")

    return {
        "output_path": str(output_path),
        "sample_path": str(sample_path) if sample_path else None,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
    }


def main():
    parser = argparse.ArgumentParser(description="Feature Engineering Module")
    parser.add_argument("--sample", action="store_true", help="Also save small sample outputs for GitHub-safe sharing")
    parser.add_argument("--sample-rows", type=int, default=50, help="Number of rows for sample output")
    args = parser.parse_args()

    print("=" * 70)
    print(" FEATURE ENGINEERING MODULE - T1.5")
    print("=" * 70)
    print(f"📁 Input:  {INPUT_DIR}")
    print(f"📁 Output: {OUTPUT_DIR}")
    if args.sample:
        print(f"📁 Sample Output: {SAMPLE_OUTPUT_DIR}")

    if not INPUT_DIR.exists():
        print(f"\n❌ Input directory does not exist: {INPUT_DIR}")
        return

    files = get_cleaned_files(INPUT_DIR)
    print(f"\n📁 Found {len(files)} datasets to process\n")

    successful = 0
    failed = 0

    for file_path in files:
        try:
            process_file(file_path, sample_mode=args.sample, sample_rows=args.sample_rows)
            successful += 1
        except Exception as e:
            print(f"\n   ❌ ERROR: {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(" FEATURE ENGINEERING COMPLETE!")
    print("=" * 70)
    print(f"   ✅ Successful: {successful}/{len(files)}")
    print(f"   ❌ Failed: {failed}/{len(files)}")
    print(f"   📁 Output folder: {OUTPUT_DIR}")
    if args.sample:
        print(f"   📁 Sample output folder: {SAMPLE_OUTPUT_DIR}")
    print("=" * 70)

    print("\n📊 FEATURE FILES CREATED:")
    print("-" * 50)

    for f in sorted(OUTPUT_DIR.glob("*_features.csv")):
        try:
            df_check = pd.read_csv(f)
            print(f"   ✅ {f.name}")
            print(f"      → {df_check.shape[0]} rows, {df_check.shape[1]} columns")
            print(f"      → All numeric: {all(df_check.dtypes.isin(['float64', 'int64']))}")
            print(f"      → No nulls: {not df_check.isnull().any().any()}")
        except Exception as e:
            print(f"   ❌ Could not verify {f.name}: {e}")

    print("\n" + "=" * 70)
    print("🎉 T1.5 IS COMPLETE! READY FOR MACHINE LEARNING!")
    print("=" * 70)


if __name__ == "__main__":
    main()