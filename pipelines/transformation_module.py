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
LOG_DIR = PROJECT_ROOT / "logs" / "transform"  # Individual logs per dataset


# ============================================================

class TransformationModule:
    """
    Cleans raw datasets: median imputation, type casting, duplicate removal.
    PRESERVES all useful columns while removing only truly constant ones.
    Saves curated files to data/curated/ with comma separator.
    Creates individual log files for each dataset in logs/transform/
    """

    def __init__(self, raw_dir=None, curated_dir=None, log_dir=None):
        self.raw_dir = Path(raw_dir) if raw_dir else RAW_DIR
        self.curated_dir = Path(curated_dir) if curated_dir else CURATED_DIR
        self.log_dir = Path(log_dir) if log_dir else LOG_DIR

        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.curated_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        print(f"📂 Raw directory: {self.raw_dir}")
        print(f"📂 Curated directory: {self.curated_dir}")
        print(f"📝 Log directory: {self.log_dir}")

    def _make_json_serializable(self, obj):
        """Convert non-JSON-serializable objects to JSON-serializable types."""
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj

    def _save_individual_log(self, dataset_name, log_data):
        """Save individual log for a dataset."""
        # Create safe filename from dataset name
        safe_name = dataset_name.replace('.csv', '').replace(' ', '_')
        log_path = self.log_dir / f"{safe_name}_transformation.json"

        # Add timestamp
        log_data["timestamp"] = datetime.now().isoformat()

        # Make JSON serializable
        log_data = self._make_json_serializable(log_data)

        # Save individual log
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2)

        print(f"   📝 Individual log saved: {log_path.name}")

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

    def _remove_duplicates(self, df):
        """Drop duplicate rows based on all columns."""
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        duplicates_removed = before - after

        if duplicates_removed > 0:
            print(f"   🗑️  Removed {duplicates_removed} duplicate rows")

        return df, duplicates_removed

    def _drop_truly_constant_columns(self, df):
        """
        Drop only columns that are TRULY constant (all values identical).
        This is GOOD practice - constant columns provide no predictive power.
        """
        before_cols = len(df.columns)
        constant_cols = []

        for col in df.columns:
            try:
                # Check unique values excluding NaN
                n_unique = df[col].nunique(dropna=False)

                if n_unique == 0:
                    # All NaN values
                    constant_cols.append(col)
                    print(f"   🔍 Found all-NaN column: '{col}'")
                elif n_unique == 1:
                    # All same value (could be NaN or a constant)
                    unique_val = df[col].iloc[0] if len(df) > 0 else "empty"
                    constant_cols.append(col)
                    print(f"   🔍 Found constant column: '{col}' (value: {unique_val})")
            except Exception as e:
                print(f"   ⚠️  Could not check column {col}: {e}")

        if constant_cols:
            df = df.drop(columns=constant_cols)
            print(f"   📊 Dropped {len(constant_cols)} constant/all-NaN columns")
        else:
            print(f"   ✅ No constant columns found")

        return df, constant_cols

    def _safe_type_casting(self, df):
        """
        Safely cast columns to appropriate types WITHOUT losing data.
        Preserves original values when conversion fails.
        """
        type_changes = []

        # Try to convert object columns to numeric, but keep original if fails
        for col in df.select_dtypes(include=['object', 'string']).columns:
            # Skip if column looks like an identifier (has 'id' in name)
            if 'id' in col.lower() or col.lower() in ['customerid', 'userid']:
                print(f"   🔒 Preserving ID column as string: {col}")
                continue

            try:
                # Try conversion to numeric
                converted = pd.to_numeric(df[col], errors='coerce')
                original_dtype = df[col].dtype

                # Only apply if we didn't lose too many values (>90% converted)
                conversion_rate = converted.notna().sum() / len(df)
                if conversion_rate > 0.9:
                    df[col] = converted
                    print(f"   🔄 Converted {col} to numeric (success rate: {conversion_rate:.1%})")
                    type_changes.append({
                        "column": col,
                        "original_type": str(original_dtype),
                        "new_type": "numeric",
                        "conversion_rate": float(conversion_rate)
                    })
                else:
                    print(f"   ⚠️  Keeping {col} as object (low conversion rate: {conversion_rate:.1%})")
            except Exception as e:
                print(f"   ⚠️  Could not convert {col}: {e}")

        # Handle integer columns safely
        for col in df.select_dtypes(include=['float']).columns:
            try:
                # Check if all non-null values are integers
                non_null = df[col].dropna()
                if len(non_null) > 0 and (non_null % 1 == 0).all():
                    original_dtype = df[col].dtype
                    df[col] = df[col].astype('Int64')  # Nullable integer type
                    print(f"   🔄 Converted {col} to integer")
                    type_changes.append({
                        "column": col,
                        "original_type": str(original_dtype),
                        "new_type": "Int64"
                    })
            except:
                pass

        return df, type_changes

    def _smart_median_imputation(self, df):
        """
        Replace NaN in numeric columns with median.
        Only imputes columns that are mostly numeric.
        """
        imputed_columns = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            if col in df.columns and df[col].isnull().any():
                null_count = df[col].isnull().sum()
                median_val = df[col].median()

                # Skip if median is NA (all nulls or cannot compute)
                if pd.isna(median_val):
                    print(f"   ⚠️  Skipping {col}: median is NA ({null_count} nulls)")
                    continue

                try:
                    df[col] = df[col].fillna(median_val)
                    print(f"   📊 Imputed {null_count} nulls in {col} with median={median_val:.2f}")
                    imputed_columns[col] = {
                        "nulls_imputed": int(null_count),
                        "median_value": float(median_val)
                    }
                except Exception as e:
                    print(f"   ⚠️  Could not impute {col}: {e}")

        return df, imputed_columns

    def _handle_missing_values_categorical(self, df):
        """
        Handle missing values in categorical columns by filling with mode.
        """
        categorical_imputations = {}
        categorical_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

        for col in categorical_cols:
            if df[col].isnull().any():
                null_count = df[col].isnull().sum()
                mode_val = df[col].mode()

                if len(mode_val) > 0:
                    fill_value = mode_val[0]
                    df[col] = df[col].fillna(fill_value)
                    print(f"   📊 Filled {null_count} nulls in {col} with mode='{fill_value}'")
                    categorical_imputations[col] = {
                        "nulls_imputed": int(null_count),
                        "mode_value": str(fill_value)
                    }
                else:
                    # If no mode, fill with "Unknown"
                    df[col] = df[col].fillna("Unknown")
                    print(f"   📊 Filled {null_count} nulls in {col} with 'Unknown'")
                    categorical_imputations[col] = {
                        "nulls_imputed": int(null_count),
                        "mode_value": "Unknown"
                    }

        return df, categorical_imputations

    def process_file(self, input_path, output_filename=None):
        """
        Full transformation pipeline for one raw file.
        Preserves all useful columns while cleaning data properly.
        Creates individual log file for the dataset.
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

        # Initialize log data for this file
        log_data = {
            "name": input_path.stem,
            "version": "unknown",
            "raw_file_path": str(input_path),
            "steps": [],
            "original_shape": None,
            "duplicates_removed": 0,
            "constant_columns_dropped": [],
            "columns_type_cast": [],
            "categorical_imputations": {},
            "numeric_imputations": {},
            "curated_path": str(output_path),
            "final_shape": None,
            "status": "pending"
        }

        # Extract version from filename if present
        import re
        version_match = re.search(r'v[a-f0-9]{8}', input_path.stem)
        if version_match:
            log_data["version"] = version_match.group()

        # Read the file
        try:
            df = self._read_with_auto_delimiter(input_path)
            if df is None:
                raise Exception("Could not read file with any encoding/delimiter")
        except Exception as e:
            print(f"   ❌ Could not read file: {e}")
            log_data["status"] = "failed"
            log_data["error"] = str(e)
            self._save_individual_log(input_path.stem, log_data)
            return None

        log_data["original_shape"] = list(df.shape)
        log_data["steps"].append(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} cols")
        print(f"\n   📊 INITIAL: {df.shape[0]} rows, {df.shape[1]} columns")

        # STEP 1: Remove duplicates
        print(f"\n   🔄 STEP 1: Removing duplicates...")
        df, duplicates_removed = self._remove_duplicates(df)
        log_data["duplicates_removed"] = duplicates_removed
        if duplicates_removed > 0:
            log_data["steps"].append(f"Removed {duplicates_removed} duplicate rows")
        else:
            log_data["steps"].append("No duplicate rows found")

        # STEP 2: Remove constant columns
        print(f"\n   🔄 STEP 2: Removing truly constant columns...")
        df, constant_cols = self._drop_truly_constant_columns(df)
        log_data["constant_columns_dropped"] = constant_cols
        if constant_cols:
            log_data["steps"].append(f"Dropped {len(constant_cols)} constant columns: {constant_cols}")
        else:
            log_data["steps"].append("No constant columns found")

        # STEP 3: Handle missing values in categorical columns
        print(f"\n   🔄 STEP 3: Handling missing values in categorical columns...")
        df, categorical_imputations = self._handle_missing_values_categorical(df)
        log_data["categorical_imputations"] = categorical_imputations
        if categorical_imputations:
            log_data["steps"].append(f"Imputed {len(categorical_imputations)} categorical columns")
        else:
            log_data["steps"].append("No missing values in categorical columns")

        # STEP 4: Safe type casting
        print(f"\n   🔄 STEP 4: Safe type casting...")
        df, type_changes = self._safe_type_casting(df)
        log_data["columns_type_cast"] = type_changes
        if type_changes:
            log_data["steps"].append(f"Type cast {len(type_changes)} columns")
        else:
            log_data["steps"].append("No type casting performed")

        # STEP 5: Median imputation for numeric columns
        print(f"\n   🔄 STEP 5: Median imputation for numeric columns...")
        df, numeric_imputations = self._smart_median_imputation(df)
        log_data["numeric_imputations"] = numeric_imputations
        if numeric_imputations:
            log_data["steps"].append(f"Imputed {len(numeric_imputations)} numeric columns with median")
        else:
            log_data["steps"].append("No missing values in numeric columns")

        log_data["final_shape"] = list(df.shape)
        log_data["status"] = "success"

        print(f"\n   📊 FINAL: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"   📊 Columns preserved: {df.shape[1]} out of original {log_data['original_shape'][1]}")

        # Save to curated folder
        try:
            df.to_csv(output_path, index=False)
            print(f"\n   ✅ Saved to: {output_path}")
            log_data["steps"].append(f"Saved to {output_path.name}")
        except Exception as e:
            print(f"   ❌ Could not save file: {e}")
            log_data["status"] = "failed"
            log_data["error"] = str(e)
            self._save_individual_log(input_path.stem, log_data)
            return None

        # Save individual log
        self._save_individual_log(input_path.stem, log_data)

        # Also add columns list to log (for reference)
        log_data["columns"] = list(df.columns)

        # Re-save with columns
        safe_name = input_path.stem.replace('.csv', '').replace(' ', '_')
        log_path = self.log_dir / f"{safe_name}_transformation.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2)

        return df


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TRANSFORMATION MODULE - T1.4 (WITH INDIVIDUAL LOGS)")
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
        print(f"\n{'=' * 70}")
        print(f"[{i}/{len(csv_files)}] 📊 Processing: {raw_file.name}")
        print(f"{'=' * 70}")
        try:
            result = transformer.process_file(raw_file)
            if result is not None:
                success_count += 1
                results.append({"file": raw_file.name, "status": "success", "shape": result.shape})
            else:
                error_count += 1
                results.append({"file": raw_file.name, "status": "failed"})
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            error_count += 1
            results.append({"file": raw_file.name, "status": "failed", "error": str(e)})

    # Final summary
    print("\n" + "=" * 70)
    print("✅ TRANSFORMATION COMPLETE!")
    print("=" * 70)
    print(f"   Successfully processed: {success_count} files")
    print(f"   Errors: {error_count} files")
    print(f"📁 Curated files saved to: {transformer.curated_dir}")
    print(f"📝 Individual logs saved to: {transformer.log_dir}")
    print("=" * 70)

    # List curated files
    curated_files = list(transformer.curated_dir.glob("*.csv"))
    if curated_files:
        print(f"\n📂 CURATED FILES ({len(curated_files)} files):")
        for f in sorted(curated_files):
            size_kb = f.stat().st_size / 1024
            print(f"   - {f.name} ({size_kb:.1f} KB)")

            # Show column count for each file
            try:
                df_sample = pd.read_csv(f, nrows=0)
                print(f"     → {len(df_sample.columns)} columns")
            except:
                pass
    else:
        print("\n⚠️  No curated files were created!")

    # List log files
    log_files = list(transformer.log_dir.glob("*.json"))
    if log_files:
        print(f"\n📝 LOG FILES ({len(log_files)} files):")
        for f in sorted(log_files):
            size_kb = f.stat().st_size / 1024
            print(f"   - {f.name} ({size_kb:.1f} KB)")

    print("\n" + "=" * 70)