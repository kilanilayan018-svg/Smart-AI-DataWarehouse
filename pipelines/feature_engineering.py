import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ============================================
# CONFIGURATION
# ============================================
INPUT_DIR = "data/curated/"
OUTPUT_DIR = "data/features/"
LOG_FILE = "logs/feature_engineering_log.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("logs/", exist_ok=True)


# ============================================
# AUTO-DETECT DELIMITER FUNCTION
# ============================================

def read_with_auto_delimiter(filepath):
    """Automatically detect if file uses comma or semicolon as delimiter"""

    # Try comma first
    try:
        df = pd.read_csv(filepath)
        # If only 1 column, try semicolon
        if df.shape[1] == 1:
            print(f"    Only 1 column detected, trying semicolon separator...")
            df = pd.read_csv(filepath, sep=';')
            print(f"    ✅ Using semicolon separator - found {df.shape[1]} columns")
        else:
            print(f"    ✅ Using comma separator - found {df.shape[1]} columns")
    except:
        # If comma fails, try semicolon
        df = pd.read_csv(filepath, sep=';')
        print(f"    ✅ Using semicolon separator - found {df.shape[1]} columns")

    return df


# ============================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================

def drop_id_columns(df):
    """Remove ID columns"""
    id_cols = [c for c in df.columns if 'id' in c.lower() or c == 'Id' or c == 'customerID']
    if id_cols:
        df = df.drop(columns=id_cols)
        print(f"    Dropped IDs: {id_cols}")
    return df


def encode_categorical(df):
    """Convert text columns to numbers"""
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in categorical_cols:
        if df[col].nunique() <= 10:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])
            print(f"    One-hot encoded: {col}")
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            print(f"    Label encoded: {col}")

    return df


def scale_numeric(df, target_column):
    """Scale numeric columns to mean=0, std=1"""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if target_column in numeric_cols:
        numeric_cols.remove(target_column)

    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        print(f"    Scaled {len(numeric_cols)} numeric columns")

    return df


def fix_skew(df):
    """Apply log transform to skewed numeric columns"""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    for col in numeric_cols:
        if df[col].min() >= 0:
            skewness = df[col].skew()
            if abs(skewness) > 1.0:
                df[col] = np.log1p(df[col])
                print(f"    Log transformed: {col} (skew={skewness:.2f})")

    return df


def process_one_dataset(filename, target_column):
    """Process a single dataset"""

    print(f"\n📊 Processing: {filename}")

    file_path = os.path.join(INPUT_DIR, filename)

    # Use auto-delimiter detection
    df = read_with_auto_delimiter(file_path)
    print(f"    Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    df = drop_id_columns(df)
    df = fix_skew(df)
    df = encode_categorical(df)
    df = scale_numeric(df, target_column)

    if target_column in df.columns:
        X = df.drop(columns=[target_column])
        print(f"    Target column: {target_column}")
    else:
        X = df
        print(f"    Warning: Target '{target_column}' not found - using all columns")

    output_name = filename.replace('_cleaned.csv', '_features.csv')
    output_path = os.path.join(OUTPUT_DIR, output_name)
    X.to_csv(output_path, index=False)
    print(f"    ✅ Saved: {output_name} ({X.shape[0]} rows, {X.shape[1]} features)")

    return X.shape


# ============================================
# MAIN - PROCESS ALL DATASETS
# ============================================

print("=" * 60)
print("T1.5 - FEATURE ENGINEERING MODULE")
print("Auto-detecting delimiters (comma or semicolon)")
print("=" * 60)

# Auto-detect all cleaned files
all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('_cleaned.csv')]

print(f"\n📁 Found {len(all_files)} datasets in {INPUT_DIR}:")
for f in all_files:
    print(f"    - {f}")

# Target mapping (from your earlier verification)
target_mapping = {
    'Uncleaned_DS_jobs_vc1793e5d_cleaned.csv': 'Competitors',
    'Iris 1 _v0a43131b_cleaned.csv': 'Species',
    'metadata_vd017e4f7_cleaned.csv': 'task_type',
    'raw_data (2)_v04fa9782_cleaned.csv': ' <=50K',
    'automobile_dataset_v1acb0a5a_cleaned.csv': 'symboling',
    'customer churn_v5fd469a5_cleaned.csv': 'Churn',
    'cardio_train_v48cc5b8b_cleaned.csv': 'cardio',  # Will auto-detect semicolon
}

results = []

for filename in all_files:
    target = target_mapping.get(filename)

    if target is None:
        print(f"\n⚠️  No target mapping for {filename} - skipping")
        continue

    shape = process_one_dataset(filename, target)
    results.append({
        "filename": filename,
        "target": target,
        "rows": shape[0],
        "features": shape[1],
        "status": "success"
    })

# Save log
log_data = {
    "timestamp": pd.Timestamp.now().isoformat(),
    "module": "FeatureEngineeringModule",
    "total_datasets_found": len(all_files),
    "datasets_processed": len(results),
    "results": results
}

with open(LOG_FILE, 'w') as f:
    json.dump(log_data, f, indent=2)

print("\n" + "=" * 60)
print(f"✅ COMPLETE!")
print(f"   Processed: {len(results)}/{len(all_files)} datasets")
print(f"   Output folder: {OUTPUT_DIR}")
print(f"   Log: {LOG_FILE}")
print("=" * 60)

# Print summary of what was processed
print("\n📊 PROCESSING SUMMARY:")
for r in results:
    print(f"   ✅ {r['filename']} → {r['rows']} rows, {r['features']} features")