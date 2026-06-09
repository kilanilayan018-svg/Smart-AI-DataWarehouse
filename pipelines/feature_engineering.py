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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# ========== OPTIONAL SUPABASE IMPORTS ==========
try:
    from registry.supabase_client import log_run, update_run, upsert_dataset, supabase
    SUPABASE_AVAILABLE = True
except Exception as e:
    supabase = None
    SUPABASE_AVAILABLE = False
    SUPABASE_IMPORT_ERROR = str(e)

    def log_run(*args, **kwargs):
        print(f"   ⚠️ Supabase not available – skipping run log.")
        return None

    def update_run(*args, **kwargs):
        print(f"   ⚠️ Supabase not available – skipping run update.")
        return False

    def upsert_dataset(*args, **kwargs):
        print(f"   ⚠️ Supabase not available – skipping dataset upsert.")
        return False
# ===============================================


# ============================================
# CONFIGURATION
# ============================================
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR    = PROJECT_ROOT / "data/curated"
RAW_DIR      = PROJECT_ROOT / "data/raw"
OUTPUT_DIR   = PROJECT_ROOT / "data/features"
SCHEMA_DIR   = PROJECT_ROOT / "data/schema"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SCHEMA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# FILE READING HELPERS
# ============================================

def auto_detect_and_fix_delimiter(file_path):
    """ULTIMATE CSV READER - Works for ANY dataset"""
    print(f"   🔧 Reading: {file_path.name}")
    encodings  = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'mac_roman']
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


# ============================================
# TIME SERIES DETECTION & FEATURE EXTRACTION
# ============================================

def detect_time_series(df):
    """
    Detect if dataset has a time/date column.
    Returns (is_time_series: bool, time_col: str or None)
    """
    time_patterns = [
        'date', 'time', 'timestamp', 'created_at', 'updated_at',
        'datetime', 'week', 'hour', 'minute'
        # NOTE: 'year', 'month', 'day' intentionally excluded —
        # these are often numeric feature columns, not datetime columns
    ]

    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in time_patterns):
            try:
                parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
                if parsed.notna().sum() > len(df) * 0.5:
                    print(f"   🕐 Time series detected: '{col}'")
                    return True, col
            except:
                continue

    return False, None


def extract_time_features(df, time_col):
    """
    Extract useful features from a datetime column.
    Sorts the dataframe by time — critical for correct splitting.
    """
    try:
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], infer_datetime_format=True, errors='coerce')

        # Sort by time — must happen before splitting
        df = df.sort_values(time_col).reset_index(drop=True)
        print(f"   📅 Sorted dataset by '{time_col}'")

        # Extract calendar components
        df[f'{time_col}_year']       = df[time_col].dt.year
        df[f'{time_col}_month']      = df[time_col].dt.month
        df[f'{time_col}_day']        = df[time_col].dt.day
        df[f'{time_col}_weekday']    = df[time_col].dt.weekday
        df[f'{time_col}_quarter']    = df[time_col].dt.quarter
        df[f'{time_col}_is_weekend'] = (df[time_col].dt.weekday >= 5).astype(int)

        # Hour / minute only if time component is present
        if df[time_col].dt.hour.nunique() > 1:
            df[f'{time_col}_hour']   = df[time_col].dt.hour
            df[f'{time_col}_minute'] = df[time_col].dt.minute

        df = df.drop(columns=[time_col])
        print(f"   ✅ Extracted time features from '{time_col}'")

    except Exception as e:
        print(f"   ⚠️ Could not extract time features from '{time_col}': {e}")
        if time_col in df.columns:
            df = df.drop(columns=[time_col])

    return df


# ============================================
# CLEANING HELPERS
# ============================================

def drop_id_columns(df, protect_col=None):
    """
    Drop ID columns but keep at least one column.
    Never drops protect_col (the target column).
    """
    id_patterns = [
        'id', 'Id', 'ID', 'customerID', 'EmployeeNumber',
        'index', 'IMBD title ID', 'PassengerId'
    ]
    id_cols = []

    for col in df.columns:
        if col == protect_col:
            continue
        if any(pattern in col for pattern in id_patterns):
            id_cols.append(col)

    if len(id_cols) == len(df.columns):
        return df, []

    if id_cols:
        df = df.drop(columns=id_cols)
        print(f"   🗑️ Dropped ID columns: {id_cols}")

    return df, id_cols


def drop_constant_columns(df, protect_col=None):
    """
    Drop constant columns but keep at least one.
    Never drops protect_col (the target column).
    """
    constant_cols = [
        col for col in df.columns
        if df[col].nunique() <= 1 and col != protect_col
    ]

    if len(constant_cols) == len(df.columns):
        return df, []

    if constant_cols:
        df = df.drop(columns=constant_cols)
        print(f"   🗑️ Dropped constant columns: {constant_cols}")

    return df, constant_cols


def handle_missing_values(df):
    """Fill missing values — median for numeric, 'unknown' for categorical"""
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(0 if pd.isna(median_val) else median_val)

    # Fix pandas 4 deprecation warning — use 'str' instead of 'object'
    for col in df.select_dtypes(include=['object', 'str']).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna('unknown')

    return df


def replace_infinite_values(df):
    """Replace inf/-inf with NaN then fill with 0"""
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(0)
    return df


# ============================================
# TARGET & TASK DETECTION (FALLBACK)
# ============================================

def identify_target_column(df, dataset_name=None):
    """
    Fallback: Automatically identify the target column using statistical scoring.
    Only called when model plan is not available.
    """
    bad_patterns = [
        'id', 'Id', 'ID', 'customerID', 'EmployeeNumber', 'PassengerId',
        'index', 'timestamp', 'date', 'time', '202', 'created_at', 'updated_at'
    ]

    scores = {}

    for col in df.columns:
        col_str = str(col)

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

        target_words = [
            'target', 'label', 'class', 'output', 'result', 'outcome',
            'predict', 'response', 'survived', 'churn', 'attrition',
            'price', 'charges', 'sales', 'mpg', 'delay', 'species',
            'cardio', 'admit', 'income', 'salary'
        ]
        for word in target_words:
            if word in col_str.lower():
                score += 50
                break

        if col == df.columns[-1]:
            score += 30

        unique_count = col_data.nunique()
        if 2 <= unique_count <= 10:
            score += 20
        if unique_count > 100:
            score -= 50

        if col_data.dtype == 'object':
            score += 10

        if ' ' in col_str or any(c in col_str for c in ['$', '%', '@']):
            score -= 20

        scores[col] = score

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


def detect_task_type(df, dataset_name=None,
                     unique_ratio_threshold=0.05, max_integer_classes=20):
    """
    Fallback: Detect classification vs regression from target column statistics.
    Only called when model plan is not available.
    """
    target_col = identify_target_column(df, dataset_name)
    print(f"   🎯 Target column: '{target_col}'")

    if not target_col or target_col not in df.columns:
        print(f"   ⚠️ Target column not found — defaulting to classification")
        return "classification", None

    col = df[target_col].dropna()

    if len(col) == 0:
        return "classification", target_col

    if not pd.api.types.is_numeric_dtype(col):
        print(f"   📊 Detected CLASSIFICATION (non-numeric dtype: {col.dtype})")
        return "classification", target_col

    unique_ratio = col.nunique() / len(col)
    if unique_ratio > unique_ratio_threshold:
        print(f"   📊 Detected REGRESSION (unique ratio: {unique_ratio:.3f})")
        return "regression", target_col

    try:
        if not np.array_equal(col.values, col.values.astype(int)):
            print(f"   📊 Detected REGRESSION (non-integer floats)")
            return "regression", target_col
    except (ValueError, TypeError):
        return "classification", target_col

    if col.nunique() <= max_integer_classes:
        print(f"   📊 Detected CLASSIFICATION ({col.nunique()} distinct values)")
        return "classification", target_col

    print(f"   📊 Detected REGRESSION ({col.nunique()} distinct integers)")
    return "regression", target_col


# ============================================
# ENCODING & SCALING
# ============================================

def encode_categorical(df, target_col=None):
    """
    Encode categorical columns using label encoding.
    Never encodes the target column.
    """
    categorical_cols = []
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].dtype == 'object' or df[col].dtype.name == 'string':
            categorical_cols.append(col)
        elif df[col].nunique() <= 10:
            categorical_cols.append(col)

    for col in categorical_cols:
        df[col] = df[col].astype(str).fillna('unknown')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        print(f"      🔄 Encoded '{col}'")

    return df


def scale_numeric(df, target_col=None, strategy='standard'):
    """
    Scale numeric columns.
    Never scales the target column.
    Supports 'standard' and 'minmax'.
    """
    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col != target_col
    ]

    if not numeric_cols:
        return df

    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(0)

    scaler = MinMaxScaler() if strategy == 'minmax' else StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print(f"   📏 Scaled {len(numeric_cols)} numeric columns using '{strategy}'")

    return df


# ============================================
# TRAIN / TEST SPLIT
# ============================================

def split_data(df, target_col, task_type, is_time_series=False):
    """
    Split dataset into train/test.
    - Time series  → chronological split (no shuffle)
    - Regular      → random stratified split
    """
    if not target_col or target_col not in df.columns:
        print(f"   ⚠️ Target '{target_col}' not in dataframe — skipping split")
        return None, None, None, None

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if is_time_series:
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        print(f"   ✅ Time-based split: {len(X_train)} train, {len(X_test)} test")
    else:
        from sklearn.model_selection import train_test_split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y if task_type == "classification" else None
            )
        except ValueError:
            print(f"   ⚠️ Stratified split failed — falling back to random split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        print(f"   ✅ Random split: {len(X_train)} train, {len(X_test)} test")

    return X_train, X_test, y_train, y_test


# ============================================
# MAIN PROCESSING FUNCTION
# ============================================

def process_file(file_path, plan=None, sample_mode=False, sample_rows=50):
    """
    Process a single cleaned CSV file into features.

    Args:
        file_path:   Path to the cleaned CSV file
        plan:        Optional model plan dict from DeepSeek+LoRA dispatcher.
                     If provided, target_column and task_type are taken from it.
                     If None or invalid, pipeline falls back to rule-based detection.
        sample_mode: Save a small sample output file
        sample_rows: Number of rows in the sample
    """

    print(f"\n{'=' * 70}")
    print(f"📊 Processing: {file_path.name}")
    print(f"{'=' * 70}")

    start_time = datetime.now()

    # Load data
    df = get_source_data(file_path)
    if df is None:
        print(f"   ❌ Could not load data")
        return None

    print(f"   📊 Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    clean_dataset_name = extract_clean_dataset_id(file_path)
    print(f"   🏷️ Dataset name: {clean_dataset_name}")

    # ── Step 1: Detect time series BEFORE any column dropping ──────────────
    is_time_series, time_col = detect_time_series(df)
    if is_time_series:
        df = extract_time_features(df, time_col)
    # ────────────────────────────────────────────────────────────────────────

    # ── Step 2: Get target & task type — MODEL FIRST, pipeline fallback ─────
    if plan and plan.get("target_column"):
        # Model plan available — use it
        target_col = plan.get("target_column")
        task_type  = plan.get("task_type", "classification")
        print(f"   🤖 Target from model:    '{target_col}'")
        print(f"   🤖 Task type from model: '{task_type}'")

        # Validate model's target actually exists in the dataframe
        if target_col not in df.columns:
            print(f"   ⚠️ Model target '{target_col}' not in dataframe — falling back to pipeline")
            task_type, target_col = detect_task_type(df, clean_dataset_name)
            print(f"   🔧 Fallback target: '{target_col}'")
    else:
        # No plan — use rule-based detection
        print(f"   🔧 No model plan — using pipeline fallback")
        task_type, target_col = detect_task_type(df, clean_dataset_name)
    # ────────────────────────────────────────────────────────────────────────

    # ── Step 3: Upsert dataset metadata to Supabase ─────────────────────────
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
    # ────────────────────────────────────────────────────────────────────────

    # ── Step 4: Log run start ───────────────────────────────────────────────
    run_id = None
    try:
        run_id = log_run(dataset_name=clean_dataset_name, status="running")
    except Exception as e:
        print(f"   ⚠️ Supabase log error: {e}")
    # ────────────────────────────────────────────────────────────────────────

    try:
        # ── Step 5: Drop ID & constant columns — protect target ─────────────
        df, _ = drop_id_columns(df, protect_col=target_col)
        df, _ = drop_constant_columns(df, protect_col=target_col)
        # ────────────────────────────────────────────────────────────────────

        # ── Step 6: Clean ────────────────────────────────────────────────────
        df = handle_missing_values(df)
        df = replace_infinite_values(df)
        # ────────────────────────────────────────────────────────────────────

        # ── Step 7: Encode — never touch target column ───────────────────────
        df = encode_categorical(df, target_col=target_col)
        # ────────────────────────────────────────────────────────────────────

        # ── Step 8: Scale — never touch target column ────────────────────────
        # Get scaling strategy from model plan if available
        scaling_strategy = 'standard'
        if plan:
            preprocessing = plan.get("preprocessing_plan", {}).get("preprocessing", {})
            scaling_strategy = preprocessing.get("scaling", {}).get("strategy", "standard")

        df = scale_numeric(df, target_col=target_col, strategy=scaling_strategy)
        # ────────────────────────────────────────────────────────────────────

        # Final safety cleanup
        df = df.fillna(0).replace([np.inf, -np.inf], 0)

        if df.shape[1] == 0:
            print(f"   ⚠️ No features remaining — adding placeholder")
            df['placeholder'] = 0

        # ── Step 9: Smart train/test split ───────────────────────────────────
        if target_col and target_col in df.columns:
            X_train, X_test, y_train, y_test = split_data(
                df, target_col, task_type, is_time_series=is_time_series
            )
        else:
            X_train = X_test = y_train = y_test = None
            print(f"   ⚠️ Skipping split — target column not available")
        # ────────────────────────────────────────────────────────────────────

        # ── Step 10: Save outputs ────────────────────────────────────────────
        output_name = file_path.name.replace('_cleaned.csv', '_features.csv')
        output_path = OUTPUT_DIR / output_name
        df.to_csv(output_path, index=False)
        print(f"\n   ✅ SAVED: {output_name}")
        print(f"      📊 Shape:       {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"      🎯 Task:        {task_type} | Target: {target_col}")
        print(f"      🕐 Time series: {is_time_series}")
        print(f"      🤖 Plan source: {'Model (DeepSeek+LoRA)' if plan and plan.get('target_column') else 'Pipeline fallback'}")

        if sample_mode:
            sample_path = OUTPUT_DIR / f"{file_path.stem}_sample.csv"
            df.head(sample_rows).to_csv(sample_path, index=False)
            print(f"   📁 Sample saved: {sample_path.name}")

        if X_train is not None:
            split_dir = OUTPUT_DIR / "splits" / clean_dataset_name
            split_dir.mkdir(parents=True, exist_ok=True)
            X_train.to_csv(split_dir / "X_train.csv", index=False)
            X_test.to_csv(split_dir  / "X_test.csv",  index=False)
            y_train.to_csv(split_dir / "y_train.csv", index=False)
            y_test.to_csv(split_dir  / "y_test.csv",  index=False)
            print(f"      💾 Split files saved to: splits/{clean_dataset_name}/")
        # ────────────────────────────────────────────────────────────────────

        # ── Step 11: Update Supabase run ─────────────────────────────────────
        if run_id and SUPABASE_AVAILABLE:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            supabase.table("runs").update({
                "status": "completed",
                "completed_at": end_time.isoformat(),
                "execution_time_seconds": duration,
                "rows_processed": df.shape[0],
                "error_message": "No error"
            }).eq("run_id", run_id).execute()
            print(f"   ✅ Run {run_id} completed in {duration:.2f}s")
        # ────────────────────────────────────────────────────────────────────

        return df

    except Exception as e:
        if run_id and SUPABASE_AVAILABLE:
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
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Engineering Module")
    parser.add_argument("--sample",          action="store_true", help="Save sample outputs")
    parser.add_argument("--sample-rows",     type=int, default=50)
    parser.add_argument("--force-reprocess", action="store_true", help="Force reprocessing all files")
    args = parser.parse_args()

    print("=" * 70)
    print(" FEATURE ENGINEERING MODULE")
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
    failed     = 0

    for file_path in files:
        # When running standalone (no model plan), pipeline fallback is used for all datasets
        result = process_file(
            file_path,
            plan=None,           # Pass plan from dispatcher when running in full pipeline
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