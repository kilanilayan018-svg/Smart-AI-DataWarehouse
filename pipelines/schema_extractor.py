import pandas as pd
import json
import os
from pathlib import Path

# Force correct working directory
project_root = "/Users/mahaqaddoumi/PycharmProjects/Smart-AI-DataWarehouse1"
os.chdir(project_root)
print(f"Working directory: {os.getcwd()}")


def read_csv_auto_delimiter(file_path):
    """
    Auto-detect comma or semicolon delimiter by analyzing first line.
    Handles multiple encodings.
    """

    # Try different encodings in order
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'mac_roman']

    first_line = None
    used_encoding = None

    # Find encoding that works for first line
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                first_line = f.readline()
                used_encoding = encoding
                break
        except:
            continue

    if first_line is None:
        # Last resort - ignore errors
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
            used_encoding = 'utf-8 (with errors ignored)'

    print(f"   📖 Encoding: {used_encoding}")

    # Count delimiters
    comma_count = first_line.count(',')
    semicolon_count = first_line.count(';')

    print(f"   📊 Delimiter analysis: {comma_count} commas, {semicolon_count} semicolons")

    # Choose delimiter and read with the working encoding
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

    # Final check - if still 1 column, force semicolon
    if df.shape[1] == 1:
        print(f"   🔧 WARNING: Only 1 column, forcing semicolon...")
        df = pd.read_csv(file_path, sep=';', encoding='latin-1')

    return df


def extract_schema(file_path):
    """Extract schema from CSV file"""

    df = read_csv_auto_delimiter(file_path)

    print(f"   📊 Shape: {df.shape[0]} rows, {df.shape[1]} columns")

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
    """Save schema as JSON"""

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
    print("Auto-detecting delimiters & encodings")
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
            print(f"   📊 {len(schema)} columns\n")
        except Exception as e:
            print(f"   ❌ Error: {e}\n")

    print("=" * 60)
    print("✅ Schema extraction complete!")
    print(f"📁 Schemas saved to: {schema_folder}")
    print("=" * 60)