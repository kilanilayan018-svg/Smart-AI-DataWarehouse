import pandas as pd
import json
import os
import warnings
from pathlib import Path

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')

# ============================================
# PROJECT PATHS (Fixed - works from anywhere)
# ============================================
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data/raw"
SCHEMA_DIR = PROJECT_ROOT / "data/schema"

print(f"Project root: {PROJECT_ROOT}")
print(f"Raw directory: {RAW_DIR}")
print(f"Schema directory: {SCHEMA_DIR}")
print(f"Working directory: {os.getcwd()}")


# ============================================
# HELPER FUNCTIONS
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

    comma_count = first_line.count(',')
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


def detect_target_column(df):
    """
    Auto-detect the target column with confidence score.
    Returns: (target_column, detection_method, confidence_score)
    """
    strong_targets = [
        "target", "label", "class", "species", "price", "output", "y",
        "churn", "attrition", "cardio", "diagnosis", "survived", "mpg",
        "sales", "expenses", "score", "rating", "grade", "income"
    ]

    def looks_like_id(col):
        lowered = col.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
        return lowered == "id" or lowered.endswith("id") or "passengerid" in lowered

    columns = list(df.columns)
    best_match = None
    best_score = 0
    best_method = "unknown"

    for col in columns:
        score = 0
        method = "unknown"
        col_lower = col.lower()

        # Exact keyword match (highest confidence)
        if col_lower in strong_targets:
            score += 100
            method = "exact_keyword"
        else:
            # Partial keyword match
            for kw in strong_targets:
                if kw in col_lower:
                    score += 50
                    method = "partial_keyword"
                    break

        # Last column bonus (common dataset convention)
        if col == columns[-1]:
            score += 30
            if method == "unknown":
                method = "last_column"

        # Low cardinality bonus (2-10 unique values - good for classification)
        unique_count = df[col].nunique()
        if 2 <= unique_count <= 10:
            score += 20
            if method == "unknown":
                method = "low_cardinality"

        # Non-numeric dtype bonus (often target in classification)
        if df[col].dtype == 'object':
            score += 10
            if method == "unknown":
                method = "non_numeric"

        # Penalty for ID-like columns
        if looks_like_id(col):
            score -= 100

        # Penalty for very high cardinality (likely not target)
        if unique_count > 100:
            score -= 30

        if score > best_score:
            best_score = score
            best_match = col
            best_method = method

    if best_match and best_score > 20:
        return best_match, best_method, best_score
    else:
        return None, "none", 0


def detect_time_series(df):
    """
    Detect if dataset is time-series and identify the date column.
    Only flags columns that are ACTUAL dates (not random numbers).
    """
    date_columns = []

    for col in df.columns:
        col_lower = col.lower()

        # First check: column name must contain date/time indicators
        is_date_name = any(keyword in col_lower for keyword in
                           ['date', 'time', 'timestamp', 'year', 'month', 'day',
                            'datetime', 'created', 'updated', 'recorded'])

        if not is_date_name:
            continue

        # Second check: try to convert to datetime
        try:
            sample = df[col].dropna().head(10)
            dates = pd.to_datetime(sample, errors='coerce')

            # Check if conversion worked (not all NaN)
            if dates.isna().all():
                continue

            # Check if values look like real dates (year between 1900 and 2100)
            sample_year = dates.dt.year.dropna()
            if len(sample_year) > 0:
                if sample_year.min() < 1900 or sample_year.max() > 2100:
                    continue

            date_columns.append(col)
        except:
            continue

    if not date_columns:
        return None, None

    # Use the first detected date column
    date_col = date_columns[0]

    try:
        dates = pd.to_datetime(df[date_col])

        # Check if dates are monotonic (increasing or decreasing)
        is_monotonic = dates.is_monotonic_increasing or dates.is_monotonic_decreasing

        # Detect frequency
        if len(dates) > 1:
            diff = dates.diff().dropna()
            if len(diff) > 0:
                most_common_diff = diff.mode()[0]
                if most_common_diff == pd.Timedelta(days=1):
                    freq = "daily"
                elif most_common_diff == pd.Timedelta(days=7):
                    freq = "weekly"
                elif most_common_diff == pd.Timedelta(days=30):
                    freq = "monthly"
                elif most_common_diff == pd.Timedelta(days=365):
                    freq = "yearly"
                else:
                    freq = "irregular"
            else:
                freq = "unknown"
        else:
            freq = "unknown"

        if is_monotonic:
            return date_col, freq
        else:
            return date_col, "non_sequential"

    except:
        return date_col, "invalid_format"


def extract_schema(file_path):
    """
    Extract schema from CSV file with time-series, skewness, and outlier detection.
    """
    df = read_csv_auto_delimiter(file_path)

    print(f"   📊 Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Target detection with confidence
    target_column, detection_method, confidence = detect_target_column(df)

    # Time-series detection
    time_series_col, time_series_freq = detect_time_series(df)

    if target_column:
        target_data = df[target_column]
        value_counts = target_data.value_counts()

        print(f"\n   🎯 Target column: '{target_column}'")
        print(f"   🔍 Detection method: {detection_method} (confidence: {confidence}%)")
        print(f"   📌 dtype: {target_data.dtype}")
        print(f"   📌 unique values ({target_data.nunique()}):")
        for val, count in list(value_counts.items())[:5]:
            pct = (count / len(df)) * 100
            print(f"      {val}: {count} ({pct:.1f}%)")
    else:
        print(f"\n   ⚠️  Could not detect target column (confidence too low)")

    if time_series_col:
        print(f"\n   🕐 Time-series detected: column='{time_series_col}', frequency='{time_series_freq}'")
    else:
        print(f"\n   🕐 Time-series: False (no date column detected)")

    # Build schema
    schema = {}
    skewed_cols = []
    outlier_cols = []
    total_missing = 0
    total_outliers = 0

    for column in df.columns:
        col_data = df[column]

        col_info = {
            "name": column,
            "dtype": str(col_data.dtype),
            "missing_count": int(col_data.isnull().sum()),
            "missing_percentage": round((col_data.isnull().sum() / len(df)) * 100, 2),
            "unique_count": int(col_data.nunique()),
            "is_target": column == target_column
        }

        total_missing += col_info["missing_count"]

        if pd.api.types.is_numeric_dtype(col_data):
            clean = col_data.dropna()
            if len(clean) > 0:
                col_info["min"] = float(clean.min())
                col_info["max"] = float(clean.max())
                col_info["mean"] = float(clean.mean())

                # Skewness detection
                try:
                    skewness = col_data.skew()
                    if not pd.isna(skewness):
                        col_info["skewness"] = round(float(skewness), 4)
                        if abs(skewness) > 1.0:
                            skewed_cols.append(column)
                            print(f"   📐 High skew detected: '{column}' (skew={skewness:.4f})")
                except:
                    pass

                # Outlier detection (IQR method)
                try:
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    outlier_count = len(outliers)
                    if outlier_count > 0:
                        outlier_cols.append(column)
                        total_outliers += outlier_count
                        col_info["outlier_count"] = int(outlier_count)
                        col_info["outlier_percentage"] = round((outlier_count / len(df)) * 100, 2)
                        print(f"   ⚠️ Outliers detected: '{column}' ({outlier_count} IQR outliers)")
                except:
                    pass

        elif pd.api.types.is_object_dtype(col_data):
            samples = col_data.dropna().unique()[:3]
            col_info["sample_values"] = samples.tolist()

        schema[column] = col_info

    # Calculate data quality score (0-100)
    quality_score = 100

    # Penalty for missing values
    missing_percentage = (total_missing / (df.shape[0] * df.shape[1])) * 100 if (df.shape[0] * df.shape[1]) > 0 else 0
    if missing_percentage > 0:
        quality_score -= min(30, missing_percentage)

    # Penalty for outliers
    outlier_percentage = (total_outliers / df.shape[0]) * 100 if df.shape[0] > 0 else 0
    if outlier_percentage > 0:
        quality_score -= min(20, outlier_percentage / 2)

    # Penalty for no target detected
    if target_column is None:
        quality_score -= 25

    quality_score = round(max(0, quality_score), 1)

    # Meta block
    schema["_meta"] = {
        "target_column": target_column,
        "target_detection_method": detection_method,
        "target_confidence": confidence,
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "time_series_column": time_series_col,
        "time_series_frequency": time_series_freq,
        "skewed_columns": skewed_cols,
        "columns_with_outliers": outlier_cols,
        "total_missing_values": total_missing,
        "missing_percentage": round(missing_percentage, 2),
        "total_outliers": total_outliers,
        "quality_score": quality_score
    }

    return schema


def save_schema(schema, file_name, output_folder=None):
    """Save schema as JSON."""
    if output_folder is None:
        output_folder = str(SCHEMA_DIR)

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    base_name = file_name.replace('.csv', '')
    output_path = os.path.join(output_folder, f"{base_name}_schema.json")

    with open(output_path, 'w') as f:
        json.dump(schema, f, indent=2)

    print(f"   ✅ Saved: {base_name}_schema.json")
    return output_path


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SCHEMA EXTRACTOR")
    print(f"Reading from: {RAW_DIR}")
    print("Detects: delimiter, encoding, target, time-series, skewness, outliers")
    print("=" * 60 + "\n")

    # Create schema directory if it doesn't exist
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DIR.exists():
        print(f"❌ ERROR: {RAW_DIR} not found!")
        print("   Please make sure raw data exists in data/raw/")
        exit(1)

    # Get all CSV files
    csv_files = list(RAW_DIR.glob("*.csv"))

    print(f"📁 Found {len(csv_files)} CSV files in raw folder:\n")

    for i, file_path in enumerate(csv_files, 1):
        print(f"{i}. Processing: {file_path.name}")

        try:
            schema = extract_schema(str(file_path))
            save_schema(schema, file_path.name)

            meta = schema.get("_meta", {})
            print(f"   📊 {meta.get('column_count')} columns, {meta.get('row_count')} rows")

            target = meta.get('target_column')
            method = meta.get('target_detection_method')
            conf = meta.get('target_confidence')
            if target:
                print(f"   🎯 Target: '{target}' (via {method}, confidence: {conf}%)")
            else:
                print(f"   🎯 Target: None")

            print(f"   📊 Quality score: {meta.get('quality_score')}/100")

            if meta.get('time_series_column'):
                print(f"   🕐 Time series: {meta.get('time_series_column')} (freq: {meta.get('time_series_frequency')})")
            else:
                print(f"   🕐 Time series: False")

            if meta.get('skewed_columns'):
                print(f"   📐 Skewed columns: {len(meta.get('skewed_columns'))} columns")

            if meta.get('columns_with_outliers'):
                print(f"   ⚠️  Columns with outliers: {len(meta.get('columns_with_outliers'))} columns")

            print(f"   📊 Missing values: {meta.get('total_missing_values')} ({meta.get('missing_percentage')}%)")
            print()

        except Exception as e:
            print(f"   ❌ Error: {e}\n")
            import traceback

            traceback.print_exc()

    print("=" * 60)
    print("✅ Schema extraction complete!")
    print(f"📁 Schemas saved to: {SCHEMA_DIR}")
    print("=" * 60)