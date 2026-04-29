import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
import json
import traceback
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ========== SUPABASE IMPORTS ==========
from registry.supabase_client import log_run, update_run, upsert_dataset

# ======================================


# ============================================
# CONFIGURATION
# ============================================
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR = PROJECT_ROOT / "data/curated"
OUTPUT_DIR = PROJECT_ROOT / "data/features"
SCHEMA_DIR = PROJECT_ROOT / "data/schema"

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SCHEMA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# HELPER FUNCTIONS
# ============================================
def extract_original_name(file_path):
    """Extract original dataset name from curated filename"""
    stem = file_path.stem.replace('_cleaned', '')
    parts = stem.split('_')

    # Skip timestamp (2 parts) and hash (1 part)
    if len(parts) >= 4:
        original_name = '_'.join(parts[3:])
    else:
        original_name = parts[-1]

    return original_name


def get_cleaned_files(input_dir):
    """Return list of cleaned CSV files to process"""
    return list(input_dir.glob("*_cleaned.csv"))


def load_schema(dataset_name):
    """Load schema JSON for a dataset if it exists"""
    schema_path = SCHEMA_DIR / f"{dataset_name}_schema.json"
    if schema_path.exists():
        with open(schema_path, 'r') as f:
            return json.load(f)
    return None


def drop_id_columns(df, df_name=""):
    """Drop common ID-like columns"""
    id_patterns = ['id', 'Id', 'ID', 'customerID', 'EmployeeNumber', 'index', 'IMBD title ID', 'Id']
    id_cols = []
    for col in df.columns:
        if any(pattern in col for pattern in id_patterns):
            id_cols.append(col)
        elif df[col].nunique() == len(df) and len(df) > 100:
            id_cols.append(col)

    if id_cols:
        df = df.drop(columns=id_cols)
        print(f"   🗑️ Dropped ID columns: {id_cols}")
    return df, id_cols


def drop_constant_columns(df):
    """Drop columns that have only one unique value"""
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)

    if constant_cols:
        df = df.drop(columns=constant_cols)
        print(f"   🗑️ Dropped constant columns: {constant_cols}")
    return df, constant_cols


def handle_missing_values(df):
    """Fill missing values - median for numeric, 'unknown' for categorical"""
    # Fill numeric NaN with median
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0
            df[col] = df[col].fillna(median_val)

    # Fill categorical NaN with 'unknown'
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna('unknown')

    return df


def replace_infinite_values(df):
    """Replace inf and -inf with NaN then fill with 0"""
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(0)
    return df


def detect_task_type(df, target_column=None):
    """Detect if task is classification or regression"""
    if target_column and target_column in df.columns:
        unique_count = df[target_column].nunique()
        if unique_count <= 10:
            return "classification"
        else:
            return "regression"
    return "classification"


def encode_categorical(df):
    """Encode categorical columns using LabelEncoder"""
    categorical_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            categorical_cols.append(col)
        elif df[col].nunique() <= 10:
            categorical_cols.append(col)

    for col in categorical_cols:
        df[col] = df[col].astype(str).fillna('unknown')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        print(f"      🔄 Encoded '{col}'")

    return df


def scale_numeric(df):
    """Scale numeric columns using StandardScaler"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        # Final check for NaN
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(0)

        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        print(f"   📏 Scaled: {len(numeric_cols)} numeric columns")

    return df


def process_file(file_path, sample_mode=False, sample_rows=50):
    """Process a single cleaned CSV file into features"""

    print(f"\n{'=' * 70}")
    print(f"📊 Processing: {file_path.name}")
    print(f"{'=' * 70}")

    # Load data
    df = pd.read_csv(file_path)
    print(f"   📂 Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Extract original dataset name
    original_name = extract_original_name(file_path)
    print(f"   🏷️ Dataset ID: {original_name}")

    # Detect task type
    task_type = detect_task_type(df)

    # ========== INSERT/UPDATE DATASET IN SUPABASE ==========
    try:
        upsert_dataset(
            dataset_id=original_name,
            rows=df.shape[0],
            columns=df.shape[1],
            task_type=task_type
        )
    except Exception as e:
        print(f"   ⚠️ Supabase dataset error: {e}")
    # =======================================================

    # ========== LOG RUN START ==========
    try:
        run_id = log_run(dataset_id=original_name, status="running")
    except Exception as e:
        print(f"   ⚠️ Supabase log run error: {e}")
        run_id = None
    # ===================================

    try:
        # Drop ID columns
        df, dropped_ids = drop_id_columns(df, original_name)

        # Drop constant columns
        df, dropped_constants = drop_constant_columns(df)

        # Handle missing values
        df = handle_missing_values(df)

        # Replace infinite values
        df = replace_infinite_values(df)

        # Encode categorical columns
        df = encode_categorical(df)

        # Scale numeric columns
        df = scale_numeric(df)

        # Final cleanup
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)

        # Save sample if requested
        if sample_mode:
            sample_path = OUTPUT_DIR / f"{file_path.stem}_sample.csv"
            df.head(sample_rows).to_csv(sample_path, index=False)
            print(f"   📁 Sample saved: {sample_path.name} ({sample_rows} rows)")

        # Save full features
        output_name = file_path.name.replace('_cleaned.csv', '_features.csv')
        output_path = OUTPUT_DIR / output_name
        df.to_csv(output_path, index=False)

        print(f"\n   ✅ SAVED: {output_name}")
        print(f"      📊 Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"      🔍 Nulls: {df.isnull().any().any()}")

        # Only check numeric columns for infinite values
        numeric_cols_for_check = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols_for_check) > 0:
            has_infs = np.isinf(df[numeric_cols_for_check]).any().any()
        else:
            has_infs = False
        print(f"      🔍 Infs: {has_infs}")

        # ========== LOG RUN SUCCESS ==========
        if run_id:
            try:
                update_run(run_id, "completed")
            except Exception as e:
                print(f"   ⚠️ Supabase update error: {e}")
        # =====================================

        return df

    except Exception as e:
        # ========== LOG RUN FAILURE ==========
        if run_id:
            try:
                update_run(run_id, "failed")
            except Exception as e2:
                print(f"   ⚠️ Supabase update error: {e2}")
        # =====================================
        print(f"\n   ❌ ERROR: {e}")
        traceback.print_exc()
        return None


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
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
        exit(1)

    files = get_cleaned_files(INPUT_DIR)
    print(f"\n📁 Found {len(files)} datasets to process\n")

    if not files:
        print("❌ No cleaned CSV files found!")
        exit(1)

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

    # --- Verification ---
    print("\n📊 FEATURE FILES CREATED:")
    print("-" * 50)
    for f in sorted(OUTPUT_DIR.glob("*_features.csv")):
        try:
            full_df = pd.read_csv(f)
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