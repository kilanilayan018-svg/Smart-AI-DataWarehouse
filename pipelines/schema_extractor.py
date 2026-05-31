import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

print(f"Working directory: {os.getcwd()}")

# Always resolve paths relative to project root (one level up from pipelines/)
PROJECT_ROOT = Path(__file__).parent.parent


# ============================================
# FILE READING
# ============================================

def read_csv_auto_delimiter(file_path):
    """
    Auto-detect comma or semicolon delimiter by analyzing first line.
    Handles multiple encodings.
    """
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'mac_roman']

    first_line = None
    used_encoding = None

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                first_line = f.readline()
                used_encoding = encoding
                break
        except:
            continue

    if first_line is None:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
            used_encoding = 'utf-8 (with errors ignored)'

    print(f"   📖 Encoding: {used_encoding}")

    comma_count     = first_line.count(',')
    semicolon_count = first_line.count(';')

    print(f"   📊 Delimiter analysis: {comma_count} commas, {semicolon_count} semicolons")

    if semicolon_count > comma_count:
        print(f"   🔧 Using semicolon delimiter")
        try:
            df = pd.read_csv(file_path, sep=';', encoding=used_encoding)
        except:
            df = pd.read_csv(file_path, sep=';', encoding='latin-1')
    else:
        print(f"   🔧 Using comma delimiter")
        try:
            df = pd.read_csv(file_path, encoding=used_encoding)
        except:
            df = pd.read_csv(file_path, encoding='latin-1')

    if df.shape[1] == 1:
        print(f"   🔧 WARNING: Only 1 column, forcing semicolon...")
        df = pd.read_csv(file_path, sep=';', encoding='latin-1')

    return df


# ============================================
# TARGET DETECTION
# ============================================

def detect_target_column(df):
    """
    Auto-detect the target column using a priority-based heuristic.
    Consistent with PlanGenerator._resolve_target() logic.

    Priority order:
      1. Strong keyword match in column name
      2. Last non-ID column that is non-numeric dtype
      3. Last non-ID column with low cardinality (<= 10 unique values)
      4. Last non-ID column as fallback
    """
    strong_targets = [
        "target", "label", "class", "species", "price",
        "output", "y", "churn", "attrition", "cardio", "diagnosis",
        "survived", "charges", "salary", "income", "admit"
    ]

    numeric_dtypes = {"int64", "float64", "int32", "float32", "int", "float"}

    def looks_like_id(col):
        lowered = col.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
        return lowered == "id" or lowered.endswith("id")

    columns = list(df.columns)

    # Priority 1: strong keyword match
    for col in columns:
        if col.strip().lower() in strong_targets:
            return col, "keyword_match"

    # Priority 2 & 3: last non-ID column with non-numeric dtype or low cardinality
    non_id_columns = [col for col in columns if not looks_like_id(col)]

    for col in reversed(non_id_columns):
        dtype        = str(df[col].dtype).lower()
        unique_count = df[col].nunique()

        if dtype not in numeric_dtypes:
            return col, "non_numeric_dtype"

        if unique_count <= 10:
            return col, "low_cardinality"

    # Priority 4: last non-ID column as final fallback
    if non_id_columns:
        return non_id_columns[-1], "last_column_fallback"

    return None, None


# ============================================
# TIME SERIES DETECTION
# ============================================

def detect_time_series(df):
    """
    Detect if the dataset is time-series by looking for parseable datetime columns.

    Returns:
        is_time_series (bool)
        time_col (str or None) — name of the detected datetime column
        frequency (str or None) — detected frequency: 'daily', 'monthly', 'yearly', or None
    """
    time_patterns = [
        'date', 'time', 'timestamp', 'created_at', 'updated_at',
        'datetime', 'week', 'hour', 'minute'
    ]

    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in time_patterns):
            try:
                parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
                valid_ratio = parsed.notna().sum() / len(df)

                if valid_ratio > 0.5:
                    # Detect frequency
                    parsed_sorted = parsed.dropna().sort_values()
                    frequency = None

                    if len(parsed_sorted) > 1:
                        diffs = parsed_sorted.diff().dropna()
                        median_diff = diffs.median()

                        if median_diff <= pd.Timedelta(days=1):
                            frequency = "daily"
                        elif median_diff <= pd.Timedelta(days=7):
                            frequency = "weekly"
                        elif median_diff <= pd.Timedelta(days=31):
                            frequency = "monthly"
                        else:
                            frequency = "yearly"

                    print(f"   🕐 Time series detected: '{col}' (frequency: {frequency})")
                    return True, col, frequency

            except:
                continue

    return False, None, None


# ============================================
# SKEWNESS DETECTION
# ============================================

def compute_skewness(col_data):
    """
    Compute skewness for a numeric column.
    Returns skewness value and a flag if high skew (abs > 1.0).
    """
    try:
        clean = col_data.dropna()
        if len(clean) < 3:
            return None, False
        skew_val = float(clean.skew())
        is_high_skew = abs(skew_val) > 1.0
        return round(skew_val, 4), is_high_skew
    except:
        return None, False


# ============================================
# MAIN SCHEMA EXTRACTION
# ============================================

def extract_schema(file_path):
    """
    Extract schema from CSV file.

    Detects:
    - Target column
    - Time series presence
    - Skewness per numeric column
    - Outlier counts (IQR method)

    Returns a schema dict with per-column profiles and a _meta block.
    """
    df = read_csv_auto_delimiter(file_path)

    print(f"   📊 Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # ── Target detection ────────────────────────────────────────────────────
    target_column, detection_method = detect_target_column(df)

    if target_column:
        target_data  = df[target_column]
        value_counts = target_data.value_counts()

        print(f"\n   🎯 Target column: '{target_column}'")
        print(f"   🔍 Detection method: {detection_method}")
        print(f"   📌 dtype: {target_data.dtype}")
        print(f"   📌 unique values ({target_data.nunique()}):")
        for val, count in value_counts.head(5).items():
            pct = (count / len(df)) * 100
            print(f"      {val}: {count} ({pct:.1f}%)")
    else:
        print(f"\n   ⚠️  Could not detect target column")
    # ────────────────────────────────────────────────────────────────────────

    # ── Time series detection ────────────────────────────────────────────────
    is_time_series, time_col, frequency = detect_time_series(df)
    # ────────────────────────────────────────────────────────────────────────

    # ── Build per-column schema ──────────────────────────────────────────────
    schema         = {}
    skewed_columns = []  # columns with abs(skew) > 1.0

    for column in df.columns:
        col_data = df[column]

        col_info = {
            "name":               column,
            "dtype":              str(col_data.dtype),
            "missing_count":      int(col_data.isnull().sum()),
            "missing_percentage": round((col_data.isnull().sum() / len(df)) * 100, 2),
            "unique_count":       int(col_data.nunique()),
            "is_target":          column == target_column,
            "is_datetime":        column == time_col
        }

        if pd.api.types.is_numeric_dtype(col_data):
            clean = col_data.dropna()

            if len(clean) > 0:
                col_info["min"]  = float(clean.min())
                col_info["max"]  = float(clean.max())
                col_info["mean"] = float(clean.mean())
                col_info["std"]  = float(clean.std())

                # ── Skewness ─────────────────────────────────────────────
                skew_val, is_high_skew = compute_skewness(col_data)
                col_info["skewness"]      = skew_val
                col_info["high_skew"]     = is_high_skew
                col_info["log_transform_recommended"] = is_high_skew and column != target_column

                if is_high_skew and column != target_column:
                    skewed_columns.append(column)
                    print(f"   📐 High skew detected: '{column}' (skew={skew_val})")
                # ─────────────────────────────────────────────────────────

                # ── Outlier detection (IQR) ───────────────────────────────
                Q1  = float(clean.quantile(0.25))
                Q3  = float(clean.quantile(0.75))
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outlier_count = int(((clean < lower_bound) | (clean > upper_bound)).sum())
                col_info["Q1"]            = round(Q1, 4)
                col_info["Q3"]            = round(Q3, 4)
                col_info["IQR"]           = round(IQR, 4)
                col_info["outlier_count"] = outlier_count
                col_info["outlier_lower"] = round(lower_bound, 4)
                col_info["outlier_upper"] = round(upper_bound, 4)

                if outlier_count > 0:
                    print(f"   ⚠️  Outliers detected: '{column}' ({outlier_count} IQR outliers)")
                # ─────────────────────────────────────────────────────────

        elif pd.api.types.is_object_dtype(col_data):
            samples = col_data.dropna().unique()[:3]
            col_info["sample_values"] = samples.tolist()

            # Try to detect if object column is actually datetime
            if column == time_col:
                col_info["is_datetime"] = True

        schema[column] = col_info
    # ────────────────────────────────────────────────────────────────────────

    # ── Meta block ───────────────────────────────────────────────────────────
    schema["_meta"] = {
        "target_column":           target_column,
        "target_detection_method": detection_method,
        "row_count":               int(df.shape[0]),
        "column_count":            int(df.shape[1]),

        # Time series info
        "is_time_series":          is_time_series,
        "time_column":             time_col,
        "time_frequency":          frequency,

        # Skewness summary
        "skewed_columns":          skewed_columns,
        "skew_threshold":          1.0,

        # Outlier summary
        "columns_with_outliers": [
            col for col in schema
            if col != "_meta" and isinstance(schema[col].get("outlier_count"), int)
            and schema[col]["outlier_count"] > 0
        ]
    }
    # ────────────────────────────────────────────────────────────────────────

    return schema


# ============================================
# SAVE
# ============================================

def save_schema(schema, file_name, output_folder=None):
    """Save schema as JSON."""
    if output_folder is None:
        output_folder = PROJECT_ROOT / "data/schema/"
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    base_name   = file_name.replace('.csv', '')
    output_path = os.path.join(output_folder, f"{base_name}_schema.json")

    with open(output_path, 'w') as f:
        json.dump(schema, f, indent=2)

    print(f"   ✅ Saved: {base_name}_schema.json")
    return output_path


# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SCHEMA EXTRACTOR")
    print("Reading from: data/raw/")
    print("Detects: delimiter, encoding, target, time-series, skewness, outliers")
    print("=" * 60 + "\n")

    raw_folder    = PROJECT_ROOT / "data/raw/"
    schema_folder = PROJECT_ROOT / "data/schema/"

    Path(schema_folder).mkdir(parents=True, exist_ok=True)

    if not raw_folder.exists():
        print(f"❌ ERROR: {raw_folder} not found!")
        exit(1)

    csv_files = list(raw_folder.glob("*.csv"))

    print(f"📁 Found {len(csv_files)} CSV files in raw folder:\n")

    for i, file_path in enumerate(csv_files, 1):
        csv_file = file_path.name
        print(f"{i}. Processing: {csv_file}")

        try:
            schema = extract_schema(file_path)
            save_schema(schema, csv_file)

            meta = schema.get("_meta", {})
            print(f"   📊 {meta.get('column_count')} columns, {meta.get('row_count')} rows")
            print(f"   🎯 Target: '{meta.get('target_column')}' (via {meta.get('target_detection_method')})")
            print(f"   🕐 Time series: {meta.get('is_time_series')} "
                  f"(col: {meta.get('time_column')}, freq: {meta.get('time_frequency')})")
            print(f"   📐 Skewed columns: {meta.get('skewed_columns')}")
            print(f"   ⚠️  Columns with outliers: {meta.get('columns_with_outliers')}\n")

        except Exception as e:
            print(f"   ❌ Error: {e}\n")

    print("=" * 60)
    print("✅ Schema extraction complete!")
    print(f"📁 Schemas saved to: {schema_folder}")
    print("=" * 60)