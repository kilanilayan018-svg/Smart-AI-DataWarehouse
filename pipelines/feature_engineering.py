"""
Feature Engineering Module - T1.5
Author: Maha Qaddoumi
Reads from: pipelines/data/curated/
Writes to: data/features/
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
# Create output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# MAIN PROCESSING
# ============================================

print("=" * 60)
print("FEATURE ENGINEERING MODULE - T1.5")
print("=" * 60)

# Get all cleaned CSV files
files = [f for f in os.listdir(INPUT_DIR) if f.endswith('_cleaned.csv')]
print(f"\n📁 Found {len(files)} datasets to process:\n")

for file in files:
    print(f"📊 Processing: {file}")

    # Load the data
    file_path = os.path.join(INPUT_DIR, file)
    df = pd.read_csv(file_path)
    print(f"   Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # 1. Drop ID columns
    id_cols = [c for c in df.columns if 'id' in c.lower() or c == 'Id' or c == 'customerID']
    if id_cols:
        df = df.drop(columns=id_cols)
        print(f"   Dropped IDs: {id_cols}")

    # 2. Handle infinite values and NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill NaN with median for numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0
            df[col].fillna(median_val, inplace=True)

    # 3. Encode categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"   Encoded: {col}")

    # 4. Scale numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) > 0:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        print(f"   Scaled: {len(num_cols)} numeric columns")

    # 5. Save feature file
    output_name = file.replace('_cleaned.csv', '_features.csv')
    output_path = os.path.join(OUTPUT_DIR, output_name)
    df.to_csv(output_path, index=False)
    print(f"   ✅ Saved: {output_name} ({df.shape[0]} rows, {df.shape[1]} features)\n")

# ============================================
# SUMMARY
# ============================================
print("=" * 60)
print("✅ FEATURE ENGINEERING COMPLETE!")
print(f"📁 Output folder: {OUTPUT_DIR}")
print("=" * 60)