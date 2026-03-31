"""
Schema Extractor - T1.2
Author: Maha Qaddoumi
Description: Analyzes CSV files in data/raw/ and generates schema JSON files in data/schema/
"""

import pandas as pd
import json
import os
from pathlib import Path


def extract_schema(file_path):
    """
    Analyze a CSV file and return a schema dictionary.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        dict: Schema information for each column
    """

    # Try different encodings to handle various file formats
    # Non-ISO extended-ASCII text usually works with 'mac_roman' or 'cp1252'
    encodings = ['mac_roman', 'cp1252', 'latin-1', 'utf-8', 'iso-8859-1', 'utf-16']
    df = None
    used_encoding = None

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            used_encoding = encoding
            break
        except:
            continue

    # If all encodings fail, try with error handling
    if df is None:
        try:
            df = pd.read_csv(file_path, encoding='latin-1', on_bad_lines='skip')
            used_encoding = 'latin-1 (with bad lines skipped)'
        except Exception as e:
            raise Exception(f"Could not read file: {file_path} - {e}")

    print(f"   📖 Encoding used: {used_encoding}")

    schema = {}

    for column in df.columns:
        col_data = df[column]

        # Basic column information
        col_info = {
            "name": column,
            "dtype": str(col_data.dtype),
            "missing_count": int(col_data.isnull().sum()),
            "missing_percentage": round((col_data.isnull().sum() / len(df)) * 100, 2),
            "unique_count": int(col_data.nunique())
        }

        # Numeric column statistics
        if pd.api.types.is_numeric_dtype(col_data):
            clean = col_data.dropna()
            if len(clean) > 0:
                col_info["min"] = float(clean.min())
                col_info["max"] = float(clean.max())
                col_info["mean"] = float(clean.mean())

        # Text/Categorical column statistics
        elif pd.api.types.is_object_dtype(col_data):
            samples = col_data.dropna().unique()[:3]
            col_info["sample_values"] = samples.tolist()

        schema[column] = col_info

    return schema

def save_schema(schema, file_name, output_folder="data/schema/"):
    """
    Save schema dictionary as JSON file.

    Args:
        schema (dict): Schema information
        file_name (str): Original CSV file name
        output_folder (str): Where to save the JSON

    Returns:
        str: Path to saved file
    """

    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Create JSON filename from CSV filename
    base_name = file_name.replace('.csv', '')
    output_path = os.path.join(output_folder, f"{base_name}_schema.json")

    # Save to file
    with open(output_path, 'w') as f:
        json.dump(schema, f, indent=2)

    print(f"   ✅ Saved: {base_name}_schema.json")
    return output_path


if __name__ == "__main__":
    # Main execution
    print("\n" + "=" * 60)
    print("SCHEMA EXTRACTOR - T1.2")
    print("Author: Maha Qaddoumi")
    print("Processing all datasets in data/raw/")
    print("=" * 60 + "\n")

    # Define folders
    raw_folder = "data/raw/"
    schema_folder = "data/schema/"

    # Create schema folder if it doesn't exist
    Path(schema_folder).mkdir(parents=True, exist_ok=True)

    # Get all CSV files from raw folder
    csv_files = [f for f in os.listdir(raw_folder) if f.endswith('.csv')]

    print(f"📁 Found {len(csv_files)} CSV files to process:\n")

    # Process each CSV file
    for i, csv_file in enumerate(csv_files, 1):
        print(f"{i}. Processing: {csv_file}")
        file_path = os.path.join(raw_folder, csv_file)

        try:
            schema = extract_schema(file_path)
            save_schema(schema, csv_file)
            print(f"   📊 Columns: {len(schema)}\n")
        except Exception as e:
            print(f"   ❌ Error: {e}\n")

    print("=" * 60)
    print("✅ Schema extraction complete!")
    print(f"📁 Check: {schema_folder}")
    print("=" * 60)