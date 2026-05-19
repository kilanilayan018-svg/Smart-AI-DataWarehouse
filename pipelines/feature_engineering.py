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
from registry.supabase_client import log_run, update_run, upsert_dataset, supabase
# ======================================


# ============================================
# CONFIGURATION
# ============================================
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR = PROJECT_ROOT / "data/curated"
RAW_DIR = PROJECT_ROOT / "data/raw"
OUTPUT_DIR = PROJECT_ROOT / "data/features"
SCHEMA_DIR = PROJECT_ROOT / "data/schema"

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SCHEMA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# HELPER FUNCTIONS
# ============================================

def auto_detect_and_fix_delimiter(file_path):
    """ULTIMATE CSV READER - Works for ANY dataset"""
    print(f"   🔧 Reading: {file_path.name}")

    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'mac_roman']
    delimiters = [',', ';', '\t', '|', ' ']

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            if df.shape[1] > 1 and len(df) > 0:
                print(f"   ✅ Read with encoding: {encoding}, shape: {df.shape}")
                return df
        except:
            continue

    for delim in delimiters:
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, sep=delim, encoding=encoding)
                if df.shape[1] > 1 and len(df) > 0:
                    print(f"   ✅ Read with delimiter='{delim}', encoding: {encoding}")
                    return df
            except:
                continue

    try:
        df = pd.read_csv(file_path, encoding='latin-1', on_bad_lines='skip')
        print(f"   ✅ Read with fallback (latin-1, skipping bad lines)")
        return df
    except Exception as e:
        print(f"   ❌ Failed to read: {e}")
        return None


def get_source_data(file_path):
    """Load data from curated or fallback to raw"""
    if file_path.exists():
        try:
            df = pd.read_csv(file_path)
            if df.shape[1] >= 2:
                return df
        except:
            pass

    raw_filename = file_path.name.replace('_cleaned', '')
    raw_path = RAW_DIR / raw_filename
    if not raw_path.exists():
        raw_filename = file_path.name.replace('_cleaned.csv', '.csv')
        raw_path = RAW_DIR / raw_filename

    if raw_path.exists():
        return auto_detect_and_fix_delimiter(raw_path)

    return None


def extract_clean_dataset_id(file_path):
    """Extract clean dataset name from filename"""
    stem = file_path.stem.replace('_cleaned', '')
    parts = stem.split('_')

    for i, part in enumerate(parts):
        if part.isdigit() and len(part) == 8:
            continue
        if part.isdigit() and len(part) == 6:
            continue
        if len(part) >= 8 and all(c in '0123456789abcdef' for c in part.lower()):
            continue
        return '_'.join(parts[i:]).lower().replace(' ', '_')

    return stem.lower().replace(' ', '_')


def get_cleaned_files(input_dir):
    """Return list of cleaned CSV files to process"""
    return list(input_dir.glob("*_cleaned.csv"))


def drop_id_columns(df):
    """Drop ID columns but keep at least one column"""
    id_patterns = ['id', 'Id', 'ID', 'customerID', 'EmployeeNumber',
                   'index', 'IMBD title ID', 'PassengerId']
    id_cols = []

    for col in df.columns:
        if any(pattern in col for pattern in id_patterns):
            id_cols.append(col)

    if len(id_cols) == len(df.columns):
        return df, []

    if id_cols:
        df = df.drop(columns=id_cols)
        print(f"   🗑️ Dropped ID columns: {id_cols}")
    return df, id_cols


def drop_constant_columns(df):
    """Drop constant columns but keep at least one"""
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)

    if len(constant_cols) == len(df.columns):
        return df, []

    if constant_cols:
        df = df.drop(columns=constant_cols)
        print(f"   🗑️ Dropped constant columns: {constant_cols}")
    return df, constant_cols


def handle_missing_values(df):
    """Fill missing values"""
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0
            df[col] = df[col].fillna(median_val)

    for col in df.select_dtypes(include=['object']).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna('unknown')

    return df


def replace_infinite_values(df):
    """Replace inf values"""
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(0)
    return df


def identify_target_column(df, dataset_name=None):
    """
    Automatically identify the target column.
    Uses statistical properties, not manual mappings.
    """

    # Columns to NEVER use as target
    bad_patterns = ['id', 'Id', 'ID', 'customerID', 'EmployeeNumber', 'PassengerId',
                    'index', 'timestamp', 'date', 'time', '202', 'created_at', 'updated_at']

    # Score each column as potential target
    scores = {}

    for col in df.columns:
        col_str = str(col)

        # Skip bad columns
        skip = False
        for pattern in bad_patterns:
            if pattern in col_str:
                scores[col] = -100
                skip = True
                break
        if skip:
            continue

        score = 0
        col_data = df[col].dropna()

        if len(col_data) == 0:
            scores[col] = -100
            continue

        # +50 points: column name suggests target
        target_words = ['target', 'label', 'class', 'output', 'result', 'outcome',
                        'predict', 'response', 'survived', 'churn', 'attrition',
                        'price', 'sales', 'mpg', 'delay', 'species', 'cardio']
        for word in target_words:
            if word in col_str.lower():
                score += 50
                break

        # +30 points: last column (common convention)
        if col == df.columns[-1]:
            score += 30

        # +20 points: low cardinality (2-10 unique values) - good for classification
        unique_count = col_data.nunique()
        if 2 <= unique_count <= 10:
            score += 20

        # -50 points: very high cardinality (likely continuous feature, not target)
        if unique_count > 100:
            score -= 50

        # +10 points: has string values (often target in classification)
        if col_data.dtype == 'object':
            score += 10

        # -20 points: column has spaces or special chars (less likely to be target)
        if ' ' in col_str or any(c in col_str for c in ['$', '%', '@']):
            score -= 20

        scores[col] = score

    # Get column with highest score
    if scores:
        best_col = max(scores, key=lambda x: scores[x])
        best_score = scores[best_col]

        if best_score > 0:
            print(f"   🎯 Auto-detected target: '{best_col}' (score: {best_score})")
            return best_col
        else:
            print(f"   ⚠️ No good target column found (best score: {best_score})")
            return None

    return None


def detect_task_type(df, dataset_name=None, unique_ratio_threshold=0.05, max_integer_classes=20):
    """Detect classification vs regression purely from target column statistics."""

    # Auto-identify target column with dataset name
    target_col = identify_target_column(df, dataset_name)

    print(f"   🎯 Target column: '{target_col}'")

    # Check if target column exists
    if target_col not in df.columns:
        print(f"   ⚠️ Target column '{target_col}' not found in dataframe")
        return "classification", None

    col = df[target_col].dropna()

    # ========== FIX: Check if column is empty ==========
    if len(col) == 0:
        print(f"   ⚠️ Target column '{target_col}' has no valid values")
        return "classification", target_col
    # ===================================================

    # Signal 1: non-numeric dtype (string, bool, category) → classification
    if not pd.api.types.is_numeric_dtype(col):
        print(f"   📊 Detected CLASSIFICATION (non-numeric dtype: {col.dtype})")
        return "classification", target_col

    # Signal 2: high cardinality ratio → continuous → regression
    unique_ratio = col.nunique() / len(col)
    if unique_ratio > unique_ratio_threshold:
        print(f"   📊 Detected REGRESSION (unique ratio: {unique_ratio:.3f})")
        return "regression", target_col

    # Signal 3: contains floats → continuous values → regression
    try:
        if not np.array_equal(col.values, col.values.astype(int)):
            print(f"   📊 Detected REGRESSION (non-integer floats)")
            return "regression", target_col
    except (ValueError, TypeError):
        print(f"   📊 Detected CLASSIFICATION (float cast failed → likely labels)")
        return "classification", target_col

    # Signal 4: small number of distinct integers → class labels → classification
    if col.nunique() <= max_integer_classes:
        print(f"   📊 Detected CLASSIFICATION ({col.nunique()} distinct integer values)")
        return "classification", target_col

    # Fallback: many distinct integers → regression
    print(f"   📊 Detected REGRESSION ({col.nunique()} distinct integers)")
    return "regression", target_col


def encode_categorical(df):
    """Encode categorical columns"""
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
    """Scale numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
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

    # --- Start timing ---
    start_time = datetime.now()

    # Load curated data
    df = get_source_data(file_path)
    if df is None:
        print(f"   ❌ Could not load data")
        return None

    print(f"   📊 Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Extract clean dataset name
    clean_dataset_name = extract_clean_dataset_id(file_path)
    print(f"   🏷️ Dataset name: {clean_dataset_name}")

    # ✅ Detect task type on CURATED data — before any transformation
    task_type, target_col = detect_task_type(df, clean_dataset_name)

    # ========== UPSERT DATASET TO SUPABASE ==========
    try:
        upsert_dataset(
            dataset_name=clean_dataset_name,
            original_filename=file_path.name,
            rows=df.shape[0],
            columns=df.shape[1],
            task_type=task_type,
            target_column=target_col
        )
    except Exception as e:
        print(f"   ⚠️ Supabase upsert error: {e}")

    # ========== LOG RUN START ==========
    run_id = None
    try:
        run_id = log_run(dataset_name=clean_dataset_name, status="running")
    except Exception as e:
        print(f"   ⚠️ Supabase log error: {e}")
        print(f"   🔍 DEBUG: run_id before update = {run_id}")

    try:
        # ✅ Now transform — AFTER task type detection
        df, _ = drop_id_columns(df)
        df, _ = drop_constant_columns(df)
        df = handle_missing_values(df)
        df = replace_infinite_values(df)
        df = encode_categorical(df)
        df = scale_numeric(df)
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)

        if df.shape[1] == 0:
            print(f"   ⚠️ No features remaining, adding placeholder")
            df['placeholder'] = 0

        # Save sample if requested
        if sample_mode:
            sample_path = OUTPUT_DIR / f"{file_path.stem}_sample.csv"
            df.head(sample_rows).to_csv(sample_path, index=False)
            print(f"   📁 Sample saved: {sample_path.name}")

        # Save full features
        output_name = file_path.name.replace('_cleaned.csv', '_features.csv')
        output_path = OUTPUT_DIR / output_name
        df.to_csv(output_path, index=False)

        print(f"\n   ✅ SAVED: {output_name}")
        print(f"      📊 Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"      🎯 Task: {task_type} | Target: {target_col}")

        # --- Update run with success details ---
        if run_id:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            supabase.table("runs").update({
                "status": "completed",
                "completed_at": end_time.isoformat(),
                "execution_time_seconds": duration,
                "rows_processed": df.shape[0],
                "error_message": "No error"  # <-- add this line
            }).eq("run_id", run_id).execute()
            print(f"   ✅ Run {run_id} completed in {duration:.2f}s, {df.shape[0]} rows processed")

        return df

    except Exception as e:
        # --- Update run with failure details ---
        if run_id:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            supabase.table("runs").update({
                "status": "failed",
                "completed_at": end_time.isoformat(),
                "execution_time_seconds": duration,
                "rows_processed": df.shape[0] if 'df' in locals() else 0,
                "error_message": str(e)[:500]
            }).eq("run_id", run_id).execute()
            print(f"   ❌ Run {run_id} failed after {duration:.2f}s: {str(e)[:100]}")
        print(f"\n   ❌ ERROR: {e}")
        traceback.print_exc()
        return None


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Engineering Module")
    parser.add_argument("--sample", action="store_true", help="Save sample outputs")
    parser.add_argument("--sample-rows", type=int, default=50)
    parser.add_argument("--force-reprocess", action="store_true", help="Force reprocessing")
    args = parser.parse_args()

    print("=" * 70)
    print(" FEATURE ENGINEERING MODULE - T1.5")
    print("=" * 70)
    print(f"📁 Input:  {INPUT_DIR}")
    print(f"📁 Output: {OUTPUT_DIR}")
    print(f"🔧 Force reprocess: {args.force_reprocess}")

    if not INPUT_DIR.exists():
        print(f"\n❌ Input directory does not exist: {INPUT_DIR}")
        exit(1)

    files = get_cleaned_files(INPUT_DIR)
    print(f"\n📁 Found {len(files)} datasets\n")

    successful = 0
    failed = 0

    for file_path in files:
        result = process_file(
            file_path,
            sample_mode=args.sample,
            sample_rows=args.sample_rows
        )
        if result is not None:
            successful += 1
        else:
            failed += 1

    print("\n" + "=" * 70)
    print(f" FEATURE ENGINEERING COMPLETE!")
    print(f"   ✅ Successful: {successful}/{len(files)}")
    print(f"   ❌ Failed:     {failed}/{len(files)}")
    print("=" * 70)