"""
Schema Extractor - T1.2
Author: Maha Qaddoumi
Reads from: data/raw/
Outputs to: data/schema/
Auto-detects comma OR semicolon delimiters
"""

import pandas as pd
import json
import os
from pathlib import Path

import os
import pandas as pd
import json
from pathlib import Path

# ========== FORCE CORRECT WORKING DIRECTORY ==========
project_root = "/Users/mahaqaddoumi/PycharmProjects/Smart-AI-DataWarehouse1"
if os.getcwd() != project_root:
    print(f"Changing directory from {os.getcwd()} to {project_root}")
    os.chdir(project_root)
print(f"Working directory: {os.getcwd()}")
# =====================================================

# ... rest of your code
def read_csv_auto_delimiter(file_path):
    """
    Auto-detect comma or semicolon delimiter when reading from RAW.
    Returns dataframe with correct columns.
    """
    # Try comma first
    df = pd.read_csv(file_path)

    # If only 1 column and contains semicolon, use semicolon delimiter
    if df.shape[1] == 1 and ';' in str(df.columns[0]):
        print(f"   🔧 Detected semicolon delimiter - converting...")
        df = pd.read_csv(file_path, sep=';')

    return df


def extract_schema(file_path):
    """
    Analyze CSV from RAW folder and return schema dictionary.
    """

    # Auto-detect delimiter and read
    df = read_csv_auto_delimiter(file_path)

    print(f"   📊 {df.shape[0]} rows, {df.shape[1]} columns")

    schema = {}

    for column in df.columns:
        col_data = df[column]

        col_info = {
            "name": column,
            "dtype": str(col_data.dtype),
            "missing_count": int(col_data.isnull().sum()),
            "missing_percentage": round((col_data.isnull().sum() / len(df)) * 100, 2),
            "unique_count": int(col_data.nunique())
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

    return schema


def save_schema(schema, file_name, output_folder="data/schema/"):
    """
    Save schema JSON to schema folder.
    Overwrites existing files (no duplicates).
    """

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
    print("Auto-detecting comma or semicolon delimiters")
    print("=" * 60 + "\n")

    raw_folder = "data/raw/"
    schema_folder = "data/schema/"

    Path(schema_folder).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(raw_folder):
        print(f"❌ ERROR: {raw_folder} folder not found!")
        exit(1)

    csv_files = [f for f in os.listdir(raw_folder) if f.endswith('.csv')]

    if not csv_files:
        print(f"❌ No CSV files found in {raw_folder}")
        exit(1)

    print(f"📁 Found {len(csv_files)} CSV files in raw folder:\n")

    for i, csv_file in enumerate(csv_files, 1):
        print(f"{i}. Processing: {csv_file}")
        file_path = os.path.join(raw_folder, csv_file)

        try:
            schema = extract_schema(file_path)
            save_schema(schema, csv_file)
            print(f"   📊 {len(schema)} columns\n")
        except Exception as e:
            print(f"   ❌ Error: {e}\n")

    print("=" * 60)
    print("✅ Schema extraction complete!")
    print(f"📁 Schemas saved to: {schema_folder}")
    print("=" * 60)