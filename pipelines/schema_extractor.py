import pandas as pd
import json
import os
from pathlib import Path

print(f"Working directory: {os.getcwd()}")


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
        "output", "y", "churn", "attrition", "cardio", "diagnosis"
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
        dtype = str(df[col].dtype).lower()
        unique_count = df[col].nunique()

        if dtype not in numeric_dtypes:
            return col, "non_numeric_dtype"

        if unique_count <= 10:
            return col, "low_cardinality"

    # Priority 4: last non-ID column as final fallback
    if non_id_columns:
        return non_id_columns[-1], "last_column_fallback"

    return None, None


def extract_schema(file_path):
    """
    Extract schema from CSV file.
    Detects target column and marks it in schema.
    Adds _meta block with dataset-level information.
    """
    df = read_csv_auto_delimiter(file_path)

    print(f"   📊 Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # --- Target detection ---
    target_column, detection_method = detect_target_column(df)

    if target_column:
        target_data = df[target_column]
        value_counts = target_data.value_counts()

        print(f"\n   🎯 Target column: '{target_column}'")
        print(f"   🔍 Detection method: {detection_method}")
        print(f"   📌 dtype: {target_data.dtype}")
        print(f"   📌 unique values ({target_data.nunique()}):")
        for val, count in value_counts.items():
            pct = (count / len(df)) * 100
            print(f"      {val}: {count} ({pct:.1f}%)")
    else:
        print(f"\n   ⚠️  Could not detect target column")

    # --- Build schema ---
    schema = {}

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

        if pd.api.types.is_numeric_dtype(col_data):
            clean = col_data.dropna()
            if len(clean) > 0:
                col_info["min"] = float(clean.min())
                col_info["max"] = float(clean.max())
                col_info["mean"] = float(clean.mean())

        elif pd.api.types.is_object_dtype(col_data):
            samples = col_data.dropna().unique()[:3]
            col_info["sample_values"] = samples.tolist()

        schema[column] = col_info

    # --- Meta block ---
    schema["_meta"] = {
        "target_column": target_column,
        "target_detection_method": detection_method,
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1])
    }

    return schema


def save_schema(schema, file_name, output_folder="data/schema/"):
    """Save schema as JSON."""
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    base_name = file_name.replace('.csv', '')
    output_path = os.path.join(output_folder, f"{base_name}_schema.json")

    with open(output_path, 'w') as f:
        json.dump(schema, f, indent=2)

    print(f"   ✅ Saved: {base_name}_schema.json")
    return output_path


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SCHEMA EXTRACTOR - T1.2")
    print("Reading from: data/raw/")
    print("Auto-detecting delimiters, encodings & target column")
    print("=" * 60 + "\n")

    raw_folder = "data/raw/"
    schema_folder = "data/schema/"

    Path(schema_folder).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(raw_folder):
        print(f"❌ ERROR: {raw_folder} not found!")
        exit(1)

    csv_files = [f for f in os.listdir(raw_folder) if f.endswith('.csv')]

    print(f"📁 Found {len(csv_files)} CSV files in raw folder:\n")

    for i, csv_file in enumerate(csv_files, 1):
        print(f"{i}. Processing: {csv_file}")
        file_path = os.path.join(raw_folder, csv_file)

        try:
            schema = extract_schema(file_path)
            save_schema(schema, csv_file)

            # Summary print
            meta = schema.get("_meta", {})
            print(f"   📊 {meta.get('column_count')} columns, "
                  f"{meta.get('row_count')} rows")
            print(f"   🎯 Target: '{meta.get('target_column')}' "
                  f"(via {meta.get('target_detection_method')})\n")

        except Exception as e:
            print(f"   ❌ Error: {e}\n")

    print("=" * 60)
    print("✅ Schema extraction complete!")
    print(f"📁 Schemas saved to: {schema_folder}")
    print("=" * 60)