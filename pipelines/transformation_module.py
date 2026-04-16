# pipelines/transformation_module.py

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIGURATION - EDIT THESE PATHS IF NEEDED
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CURATED_DIR = PROJECT_ROOT / "data" / "curated"
LOG_FILE = PROJECT_ROOT / "logs" / "transformation_log.json"


# ============================================================

class TransformationModule:
    """
    Cleans raw datasets: median imputation, type casting, duplicate removal.
    Saves curated files to data/curated/ with comma separator.
    Writes detailed logs to logs/transformation_log.json.
    """

    def __init__(self, raw_dir=None, curated_dir=None, log_file=None):
        self.raw_dir = Path(raw_dir) if raw_dir else RAW_DIR
        self.curated_dir = Path(curated_dir) if curated_dir else CURATED_DIR
        self.log_file = Path(log_file) if log_file else LOG_FILE

        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.curated_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"📂 Raw directory: {self.raw_dir}")
        print(f"📂 Curated directory: {self.curated_dir}")
        print(f"📝 Log file: {self.log_file}")

    def _read_with_auto_delimiter(self, file_path):
        """
        Auto-detect comma or semicolon delimiter with multiple encoding attempts.
        Returns DataFrame or None if reading fails.
        """
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'mac_roman']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    first_line = f.readline()

                comma_count = first_line.count(',')
                semicolon_count = first_line.count(';')

                if semicolon_count > comma_count:
                    print(f"   🔧 Detected semicolon delimiter (encoding: {encoding})")
                    df = pd.read_csv(file_path, sep=';', encoding=encoding)
                else:
                    print(f"   🔧 Detected comma delimiter (encoding: {encoding})")
                    df = pd.read_csv(file_path, encoding=encoding)

                if df.shape[1] > 1:
                    return df

            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
            except Exception:
                continue

        # Last resort
        print(f"   🔧 Last resort: trying with error handling")
        try:
            return pd.read_csv(file_path, encoding='latin-1', on_bad_lines='skip')
        except:
            try:
                return pd.read_csv(file_path, sep=';', encoding='latin-1', on_bad_lines='skip')
            except:
                return None

    def _append_log_entry(self, entry):
        """Append a log entry to the JSON log file."""
        log_entries = []
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    log_entries = json.load(f)
            except:
                log_entries = []

        entry["timestamp"] = datetime.now().isoformat()
        log_entries.append(entry)

        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(log_entries, f, indent=2)

    def _median_imputation(self, df):
        """Replace NaN in numeric columns with median. Handles NA values safely."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()

                # Skip if median is NA (all nulls or cannot compute)
                if pd.isna(median_val):
                    print(f"   ⚠️  Skipping {col}: median is NA (all values may be null)")
                    continue

                try:
                    df[col] = df[col].fillna(median_val)
                    self._append_log_entry({
                        "action": "median_imputation",
                        "file": str(self.current_file) if hasattr(self, 'current_file') else "unknown",
                        "column": col,
                        "median_value": float(median_val),
                        "rows_imputed": int(df[col].isnull().sum())
                    })
                except Exception as e:
                    print(f"   ⚠️  Could not impute {col}: {e}")

        return df

    def _type_casting(self, df):
        """Cast columns to appropriate types."""
        original_dtypes = df.dtypes.astype(str).to_dict()

        # Convert object/string columns that look like numbers
        for col in df.select_dtypes(include=['object', 'string']).columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                pass

        # Ensure integer columns are Int64 (nullable)
        for col in df.select_dtypes(include=['float']).columns:
            try:
                if len(df[col].dropna()) > 0:
                    # Check if all non-null values are integers
                    if (df[col].dropna() % 1 == 0).all():
                        df[col] = df[col].astype('Int64')
            except:
                pass

        new_dtypes = df.dtypes.astype(str).to_dict()

        self._append_log_entry({
            "action": "type_casting",
            "file": str(self.current_file) if hasattr(self, 'current_file') else "unknown",
            "original_dtypes": original_dtypes,
            "new_dtypes": new_dtypes
        })

        return df

    def _remove_duplicates(self, df):
        """Drop duplicate rows based on all columns."""
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)

        if before != after:
            self._append_log_entry({
                "action": "remove_duplicates",
                "file": str(self.current_file) if hasattr(self, 'current_file') else "unknown",
                "rows_before": before,
                "rows_after": after,
                "duplicates_removed": before - after
            })

        return df

    def process_file(self, input_path, output_filename=None):
        """
        Full transformation pipeline for one raw file.

        Args:
            input_path: Path to raw CSV file
            output_filename: Optional custom output filename

        Returns:
            DataFrame if successful, None if failed
        """
        input_path = Path(input_path)
        self.current_file = input_path.name

        if not input_path.exists():
            print(f"   ❌ File not found: {input_path}")
            return None

        # Generate output filename
        if output_filename is None:
            output_filename = input_path.stem + "_cleaned.csv"
        output_path = self.curated_dir / output_filename

        # Read the file
        try:
            df = self._read_with_auto_delimiter(input_path)
            if df is None:
                raise Exception("Could not read file with any encoding/delimiter")
        except Exception as e:
            print(f"   ❌ Could not read file: {e}")
            self._append_log_entry({
                "action": "read_error",
                "file": input_path.name,
                "error": str(e)
            })
            return None

        print(f"   Shape: {df.shape[0]} rows, {df.shape[1]} cols")

        # Log start
        self._append_log_entry({
            "action": "start_transformation",
            "file": input_path.name,
            "output_file": output_filename,
            "original_shape": list(df.shape)
        })

        # Apply transformations
        df = self._remove_duplicates(df)
        df = self._type_casting(df)
        df = self._median_imputation(df)

        # Save to curated folder
        try:
            df.to_csv(output_path, index=False)
            print(f"   ✅ Saved to: {output_path}")
        except Exception as e:
            print(f"   ❌ Could not save file: {e}")
            self._append_log_entry({
                "action": "save_error",
                "file": input_path.name,
                "output_file": output_filename,
                "error": str(e)
            })
            return None

        # Log completion
        self._append_log_entry({
            "action": "complete_transformation",
            "file": input_path.name,
            "output_file": output_filename,
            "final_shape": list(df.shape),
            "columns": list(df.columns)
        })

        return df


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TRANSFORMATION MODULE - T1.4")
    print("=" * 70)
    print(f"Project Root: {PROJECT_ROOT}")
    print("=" * 70)

    # Initialize transformer
    transformer = TransformationModule()

    # Check if raw directory exists and has files
    if not transformer.raw_dir.exists():
        print(f"\n❌ ERROR: Raw directory not found: {transformer.raw_dir}")
        print("   Please make sure raw data exists in data/raw/")
        exit(1)

    # Get all CSV files
    csv_files = list(transformer.raw_dir.glob("*.csv"))
    print(f"\n📁 Found {len(csv_files)} CSV files in {transformer.raw_dir}\n")

    if len(csv_files) == 0:
        print("❌ No CSV files found to process!")
        exit(1)

    # Process each file
    success_count = 0
    error_count = 0
    results = []

    for i, raw_file in enumerate(csv_files, 1):
        print(f"[{i}/{len(csv_files)}] 📊 Processing: {raw_file.name}")
        result = transformer.process_file(raw_file)

        if result is not None:
            success_count += 1
            results.append({"file": raw_file.name, "status": "success", "shape": result.shape})
        else:
            error_count += 1
            results.append({"file": raw_file.name, "status": "failed"})

        print()  # Empty line for readability

    # Final summary
    print("=" * 70)
    print("✅ TRANSFORMATION COMPLETE!")
    print("=" * 70)
    print(f"   Successfully processed: {success_count} files")
    print(f"   Errors: {error_count} files")
    print(f"📁 Curated files saved to: {transformer.curated_dir}")
    print(f"📝 Log saved to: {transformer.log_file}")
    print("=" * 70)

    # List curated files
    curated_files = list(transformer.curated_dir.glob("*.csv"))
    if curated_files:
        print(f"\n📂 CURATED FILES ({len(curated_files)} files):")
        for f in sorted(curated_files):
            size_kb = f.stat().st_size / 1024
            print(f"   - {f.name} ({size_kb:.1f} KB)")
    else:
        print("\n⚠️  No curated files were created!")

    print("\n" + "=" * 70)