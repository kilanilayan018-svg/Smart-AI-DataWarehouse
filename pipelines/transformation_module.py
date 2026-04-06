# pipelines/transformation_module.py

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime

class TransformationModule:
    """
    Cleans raw datasets: median imputation, type casting, duplicate removal.
    Saves curated files to data/curated/ with comma separator.
    Also fixes the known cardio_train delimiter issue post-hoc.
    """

    def __init__(self, curated_dir="data/curated", log_file="logs/transformation_log.json"):
        self.curated_dir = Path(curated_dir)
        self.log_file = Path(log_file)
        self.curated_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _fix_cardio_delimiter(self, file_path):
        """
        Specifically corrects the cardio_train file if it was saved with ';' separator.
        Only runs when the file exists and is malformed (only one column).
        """
        if not file_path.exists():
            return

        # Try reading with comma – if it gives multiple columns, file is already correct
        try:
            test_df = pd.read_csv(file_path, nrows=1)
            if test_df.shape[1] > 1:
                return  # Already fine
        except Exception:
            pass

        # Malformed: read with semicolon, rewrite with comma
        df_fixed = pd.read_csv(file_path, sep=';')
        df_fixed.to_csv(file_path, index=False)
        self._append_log_entry({
            "action": "delimiter_fix",
            "file": file_path.name,
            "original_sep": ";",
            "new_sep": ",",
            "rows": len(df_fixed),
            "columns": list(df_fixed.columns)
        })

    def _append_log_entry(self, entry):
        """Append a log entry to the JSON log file."""
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                log = json.load(f)
        else:
            log = []
        entry["timestamp"] = datetime.now().isoformat()
        log.append(entry)
        with open(self.log_file, 'w') as f:
            json.dump(log, f, indent=2)

    def _median_imputation(self, df, numeric_cols):
        """Replace NaN in numeric columns with median."""
        for col in numeric_cols:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                self._append_log_entry({
                    "action": "median_imputation",
                    "column": col,
                    "median_value": float(median_val),
                    "rows_imputed": int(df[col].isnull().sum())
                })
        return df

    def _type_casting(self, df):
        """Cast columns to appropriate types."""
        original_dtypes = df.dtypes.astype(str).to_dict()
        # Convert object columns that look like numbers
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except Exception:
                pass
        # Ensure integer columns are int (but keep NaN as float)
        for col in df.select_dtypes(include=['float']).columns:
            if df[col].dropna().apply(float.is_integer).all():
                df[col] = df[col].astype('Int64')  # nullable integer
        new_dtypes = df.dtypes.astype(str).to_dict()
        self._append_log_entry({
            "action": "type_casting",
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
                "rows_before": before,
                "rows_after": after,
                "duplicates_removed": before - after
            })
        return df

    def process_file(self, input_path, output_filename=None):
        """
        Full transformation pipeline for one raw file.
        input_path: path to raw file (CSV, Excel, etc.)
        output_filename: optional name for curated file (default = input stem + '_cleaned.csv')
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Determine output name
        if output_filename is None:
            output_filename = input_path.stem + "_cleaned.csv"
        output_path = self.curated_dir / output_filename

        # Read raw file (auto-detect delimiter? we assume comma, but handle if needed)
        try:
            df = pd.read_csv(input_path)
        except Exception:
            # Fallback for semicolon-delimited raw files
            df = pd.read_csv(input_path, sep=';')

        # Log start
        self._append_log_entry({
            "action": "start_transformation",
            "input_file": str(input_path),
            "output_file": str(output_path),
            "original_shape": list(df.shape)
        })

        # 1. Remove duplicates
        df = self._remove_duplicates(df)

        # 2. Type casting
        df = self._type_casting(df)

        # 3. Median imputation for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df = self._median_imputation(df, numeric_cols)

        # 4. Save curated file with comma separator
        df.to_csv(output_path, index=False)

        # 5. SPECIAL FIX: cardio_train delimiter issue (if this file is the problematic one)
        if "cardio_train_v48cc5b8b_cleaned.csv" in str(output_path):
            self._fix_cardio_delimiter(output_path)

        # Log completion
        self._append_log_entry({
            "action": "complete_transformation",
            "output_file": str(output_path),
            "final_shape": list(df.shape),
            "columns": list(df.columns)
        })

        return df

# ------------------------------
# Example usage (for T1.4 owner: Layan)
if __name__ == "__main__":
    transformer = TransformationModule()
    # Process all raw files in data/raw_input/
    raw_dir = Path("data/raw_input")
    for raw_file in raw_dir.glob("*"):
        if raw_file.suffix in ['.csv', '.xlsx']:
            print(f"Processing {raw_file.name}...")
            transformer.process_file(raw_file)
    print("All transformations complete. Check data/curated/ and logs/transformation_log.json")