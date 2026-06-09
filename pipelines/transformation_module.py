# pipelines/transformation_module.py

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CURATED_DIR = PROJECT_ROOT / "data" / "curated"
LOG_DIR = PROJECT_ROOT / "logs" / "transform"


class TransformationModule:
    def __init__(self, raw_dir=None, curated_dir=None, log_dir=None):
        self.raw_dir = Path(raw_dir) if raw_dir else RAW_DIR
        self.curated_dir = Path(curated_dir) if curated_dir else CURATED_DIR
        self.log_dir = Path(log_dir) if log_dir else LOG_DIR

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.curated_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def process_file(self, input_path, output_filename=None):
        input_path = Path(input_path)
        if not input_path.exists():
            print(f"   ❌ File not found: {input_path}")
            return None

        if output_filename is None:
            output_filename = input_path.stem + "_cleaned.csv"
        output_path = self.curated_dir / output_filename

        print(f"\n   📂 Processing: {input_path.name}")

        # Try multiple encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'mac_roman']
        df = None
        used_encoding = None

        for encoding in encodings:
            try:
                df = pd.read_csv(input_path, encoding=encoding)
                used_encoding = encoding
                print(f"   ✅ Read with encoding: {encoding}")
                print(f"   📊 Shape: {df.shape[0]} rows, {df.shape[1]} columns")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                continue

        if df is None:
            try:
                df = pd.read_csv(input_path, encoding='latin-1', on_bad_lines='skip')
                used_encoding = 'latin-1 (with errors skipped)'
                print(f"   ✅ Read with fallback encoding")
                print(f"   📊 Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                print(f"   ❌ Failed to read: {e}")
                return None

        # ========== SPECIAL FIX ONLY FOR IMDB ==========
        if 'imdb' in input_path.name.lower():
            print(f"   🔧 IMDB dataset detected - checking header...")
            first_row = df.iloc[0].astype(str)

            # Check if first row looks like column names (not numeric data)
            if not first_row.str.isnumeric().all():
                new_columns = [str(col).strip() for col in first_row]
                df.columns = new_columns
                df = df.iloc[1:].reset_index(drop=True)
                print(f"   ✅ Fixed IMDB header")
                print(f"   📊 New shape: {df.shape[0]} rows, {df.shape[1]} columns")
        # ================================================

        # Save log
        log_data = {
            "name": input_path.stem,
            "encoding_used": used_encoding,
            "original_shape": list(df.shape),
            "final_shape": list(df.shape),
            "columns": list(df.columns),
            "status": "success"
        }

        # Save to curated folder
        df.to_csv(output_path, index=False)
        print(f"   ✅ Saved: {output_path.name}")

        # Save log
        safe_name = input_path.stem.replace('.csv', '').replace(' ', '_')
        log_path = self.log_dir / f"{safe_name}_transformation.json"
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)

        return df


if __name__ == "__main__":
    print("=" * 70)
    print("TRANSFORMATION MODULE - WITH ENCODING FALLBACK")
    print("=" * 70)

    transformer = TransformationModule()

    if not transformer.raw_dir.exists():
        print(f"\n❌ Raw directory not found: {transformer.raw_dir}")
        exit(1)

    csv_files = list(transformer.raw_dir.glob("*.csv"))
    print(f"\n📁 Found {len(csv_files)} files\n")

    success = 0
    for raw_file in csv_files:
        print(f"\n{'=' * 50}")
        print(f"Processing: {raw_file.name}")
        print(f"{'=' * 50}")
        result = transformer.process_file(raw_file)
        if result is not None:
            success += 1

    print("\n" + "=" * 70)
    print(f"✅ COMPLETE! Processed {success}/{len(csv_files)} files")
    print("=" * 70)