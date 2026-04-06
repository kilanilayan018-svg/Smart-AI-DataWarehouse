"""
Feature Engineering Module - T1.5 (FULL WORKING CODE)
Author: Maha Qaddoumi
Date: April 6, 2026

This module:
1. Reads curated CSV files from pipelines/data/curated/
2. Drops ID columns and constant columns (only 1 unique value)
3. Encodes categorical columns to numbers
4. Scales numeric columns to mean=0, std=1
5. Saves feature files to data/features/
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ============================================
# CONFIGURATION
# ============================================

INPUT_DIR = "/Users/mahaqaddoumi/PycharmProjects/Smart-AI-DataWarehouse2/pipelines/data/curated/"
OUTPUT_DIR = "/Users/mahaqaddoumi/PycharmProjects/Smart-AI-DataWarehouse2/data/features/"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print(" FEATURE ENGINEERING MODULE - T1.5")
print("=" * 70)
print(f"📁 Input:  {INPUT_DIR}")
print(f"📁 Output: {OUTPUT_DIR}")

# Get all cleaned CSV files
files = [f for f in os.listdir(INPUT_DIR) if f.endswith('_cleaned.csv')]
print(f"\n📁 Found {len(files)} datasets to process\n")

# ============================================
# MAIN PROCESSING LOOP
# ============================================

successful = 0
failed = 0

for file in files:
    print(f"\n{'=' * 70}")
    print(f"📊 Processing: {file}")
    print(f"{'=' * 70}")

    try:
        # --------------------------------------------------
        # 1. LOAD DATA
        # --------------------------------------------------
        df = pd.read_csv(os.path.join(INPUT_DIR, file))
        print(f"   📂 Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # --------------------------------------------------
        # 2. DROP ID COLUMNS
        # --------------------------------------------------
        id_patterns = ['id', 'Id', 'ID', 'customerID', 'EmployeeNumber', 'index', 'IMBD title ID', 'Id']
        id_cols = []
        for col in df.columns:
            if any(pattern in col for pattern in id_patterns):
                id_cols.append(col)
            elif df[col].nunique() == len(df) and len(df) > 100:
                id_cols.append(col)

        if id_cols:
            df = df.drop(columns=id_cols)
            print(f"   🗑️ Dropped ID columns: {id_cols}")

        # --------------------------------------------------
        # 3. DROP CONSTANT COLUMNS (only 1 unique value)
        # --------------------------------------------------
        constant_cols = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_cols.append(col)

        if constant_cols:
            df = df.drop(columns=constant_cols)
            print(f"   🗑️ Dropped constant columns: {constant_cols}")

        # --------------------------------------------------
        # 4. HANDLE MISSING VALUES
        # --------------------------------------------------
        # Fill numeric NaN with median
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().any():
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)

        # Fill categorical NaN with 'unknown'
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna('unknown')

        # --------------------------------------------------
        # 5. REPLACE INFINITE VALUES
        # --------------------------------------------------
        df = df.replace([np.inf, -np.inf], np.nan)
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(0)

        # --------------------------------------------------
        # 6. IDENTIFY CATEGORICAL VS NUMERIC COLUMNS
        # --------------------------------------------------
        categorical_cols = []
        numeric_cols = []

        for col in df.columns:
            # Object type = categorical
            if df[col].dtype == 'object':
                categorical_cols.append(col)
            # Low cardinality numeric = categorical
            elif df[col].nunique() <= 10:
                categorical_cols.append(col)
            # Everything else = numeric
            else:
                numeric_cols.append(col)

        print(f"   🏷️ Categorical columns: {len(categorical_cols)}")
        print(f"   🔢 Numeric columns: {len(numeric_cols)}")

        # --------------------------------------------------
        # 7. ENCODE CATEGORICAL COLUMNS
        # --------------------------------------------------
        encoding_log = {}
        for col in categorical_cols:
            # Convert to string and fill NaN
            df[col] = df[col].astype(str).fillna('unknown')
            # Label encode
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoding_log[col] = dict(zip(le.classes_, range(len(le.classes_))))
            print(f"      🔄 Encoded '{col}': {encoding_log[col]}")

        # --------------------------------------------------
        # 8. SCALE NUMERIC COLUMNS
        # --------------------------------------------------
        if numeric_cols:
            # Final check for NaN
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(0)

            # Standard scaling
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            print(f"   📏 Scaled: {len(numeric_cols)} numeric columns")

        # --------------------------------------------------
        # 9. FINAL CLEANUP
        # --------------------------------------------------
        # Remove any remaining NaN or Inf
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)

        # --------------------------------------------------
        # 10. SAVE FEATURE FILE
        # --------------------------------------------------
        output_name = file.replace('_cleaned.csv', '_features.csv')
        output_path = os.path.join(OUTPUT_DIR, output_name)
        df.to_csv(output_path, index=False)

        # --------------------------------------------------
        # 11. VERIFICATION
        # --------------------------------------------------
        print(f"\n   ✅ SAVED: {output_name}")
        print(f"      📊 Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"      🔍 Nulls: {df.isnull().any().any()}")
        print(f"      🔍 Infs: {np.isinf(df).any().any()}")

        successful += 1

    except Exception as e:
        print(f"\n   ❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        failed += 1

# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 70)
print(" FEATURE ENGINEERING COMPLETE!")
print("=" * 70)
print(f"   ✅ Successful: {successful}/{len(files)}")
print(f"   ❌ Failed: {failed}/{len(files)}")
print(f"   📁 Output folder: {OUTPUT_DIR}")
print("=" * 70)

# ============================================
# DISPLAY ALL CREATED FEATURE FILES
# ============================================

print("\n📊 FEATURE FILES CREATED:")
print("-" * 50)

for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith('_features.csv'):
        df_check = pd.read_csv(os.path.join(OUTPUT_DIR, f))
        print(f"   ✅ {f}")
        print(f"      → {df_check.shape[0]} rows, {df_check.shape[1]} columns")
        print(f"      → All numeric: {all(df_check.dtypes.isin(['float64', 'int64']))}")
        print(f"      → No nulls: {not df_check.isnull().any().any()}")

print("\n" + "=" * 70)
print("🎉 T1.5 IS COMPLETE! READY FOR MACHINE LEARNING!")
print("=" * 70)
