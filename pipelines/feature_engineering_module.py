import pandas as pd
import json
import os
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler


class FeatureEngineeringModule:
    def __init__(self, curated_file_path, name, version):
        self.curated_file_path = curated_file_path
        self.name = name
        self.version = version
        self.df = None
        self.log = {
            "name": name,
            "version": version,
            "curated_file_path": curated_file_path,
            "steps": []
        }

    def load(self):
        self.df = pd.read_csv(self.curated_file_path)
        self.log["original_shape"] = list(self.df.shape)
        self.log["steps"].append(f"Loaded curated dataset: {self.df.shape[0]} rows, {self.df.shape[1]} cols")

    def drop_id_columns(self):
        id_keywords = ['id', 'ID', 'Id', 'index', 'Index']
        dropped = []
        for col in self.df.columns:
            if any(keyword == col or col.lower().endswith('_id') for keyword in id_keywords):
                self.df.drop(columns=[col], inplace=True)
                dropped.append(col)
        self.log["id_columns_dropped"] = dropped
        self.log["steps"].append(f"Dropped ID columns: {dropped}")

    def encode_categoricals(self):
        encoded = []
        label_encoded = []
        for col in self.df.select_dtypes(include='object').columns:
            unique_vals = self.df[col].nunique()
            if unique_vals <= 10:
                # One-hot encode low cardinality
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df.drop(columns=[col]), dummies], axis=1)
                encoded.append(col)
            else:
                # Label encode high cardinality
                self.df[col] = self.df[col].astype('category').cat.codes
                label_encoded.append(col)
        self.log["one_hot_encoded"] = encoded
        self.log["label_encoded"] = label_encoded
        self.log["steps"].append(f"One-hot encoded: {encoded} | Label encoded: {label_encoded}")

    def log_transform_skewed(self):
        transformed = []
        for col in self.df.select_dtypes(include=['float64', 'int64']).columns:
            skewness = self.df[col].skew()
            if abs(skewness) > 1 and self.df[col].min() >= 0:
                self.df[col] = np.log1p(self.df[col])
                transformed.append({"column": col, "skewness_before": round(skewness, 4)})
        self.log["log_transformed"] = transformed
        self.log["steps"].append(f"Log-transformed {len(transformed)} skewed columns")

    def scale_features(self):
        scaled = []
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if numeric_cols:
            scaler = StandardScaler()
            self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
            scaled = numeric_cols
        self.log["scaled_columns"] = list(scaled)
        self.log["steps"].append(f"StandardScaler applied to {len(scaled)} numeric columns")

    def extract_datetime_features(self):
        extracted = []
        for col in self.df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                    self.df[f"{col}_year"] = self.df[col].dt.year
                    self.df[f"{col}_month"] = self.df[col].dt.month
                    self.df[f"{col}_day"] = self.df[col].dt.day
                    self.df.drop(columns=[col], inplace=True)
                    extracted.append(col)
                except Exception:
                    pass
        self.log["datetime_extracted"] = extracted
        self.log["steps"].append(f"Extracted datetime features from: {extracted}")

    def save(self, output_dir="../data/features", log_dir="../logs/features"):
        # Resolve output directories relative to script location
        script_dir = Path(__file__).parent.absolute()
        output_dir_abs = (script_dir / output_dir).resolve()
        log_dir_abs = (script_dir / log_dir).resolve()

        os.makedirs(output_dir_abs, exist_ok=True)
        os.makedirs(log_dir_abs, exist_ok=True)

        out_path = output_dir_abs / f"{self.name}_v{self.version}_features.csv"
        self.df.to_csv(out_path, index=False)

        self.log["feature_path"] = str(out_path)
        self.log["final_shape"] = list(self.df.shape)
        self.log["timestamp"] = datetime.now().isoformat()
        log_path = log_dir_abs / f"{self.name}_v{self.version}_features.json"
        with open(log_path, "w") as f:
            json.dump(self.log, f, indent=2)

        print(f"✅ {self.name}: saved to {out_path}")
        return str(out_path), str(log_path)

    def engineer(self):
        self.load()
        self.drop_id_columns()
        self.extract_datetime_features()
        self.encode_categoricals()
        self.log_transform_skewed()
        self.scale_features()
        return self.save()


# ── Runner with robust path resolution ────────────────────────────────────────

def run_feature_engineering(summary_path="../logs/transform/transform_summary.json"):
    # Get the directory where this script lives
    script_dir = Path(__file__).parent.absolute()

    # Resolve the summary path (relative to script location)
    full_summary_path = (script_dir / summary_path).resolve()
    if not full_summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {full_summary_path}")

    with open(full_summary_path) as f:
        summary = json.load(f)

    curated_datasets = summary["curated_datasets"]
    print(f"Found {len(curated_datasets)} curated datasets to process.\n")

    # Determine project root (one level up from script dir, adjust if needed)
    project_root = script_dir.parent

    results = []
    for dataset in curated_datasets:
        original_path = dataset["curated_path"]  # e.g., "data/curated/xxx_cleaned.csv"
        # Try multiple possible absolute paths
        possible_abs_paths = [
            (project_root / original_path).resolve(),          # project_root/data/curated/...
            (script_dir / original_path).resolve(),            # src/data/curated/... (unlikely)
            Path(original_path).resolve()                      # as-is, from cwd
        ]
        # Also try with "data/curated/" replaced if original_path is already relative
        # (some JSON may store just the filename; we handle by checking existence)
        found_path = None
        for p in possible_abs_paths:
            if p.exists():
                found_path = p
                break

        if found_path is None:
            print(f"❌ Cannot find curated file for {dataset['name']}: tried {[str(p) for p in possible_abs_paths]}")
            results.append({
                "name": dataset["name"],
                "version": dataset["version"],
                "status": "feature_failed",
                "error": f"File not found: {original_path}"
            })
            continue

        print(f"Processing: {dataset['name']} (using {found_path})...")
        try:
            module = FeatureEngineeringModule(
                curated_file_path=str(found_path),
                name=dataset["name"],
                version=dataset["version"]
            )
            feature_path, log_path = module.engineer()
            results.append({
                "name": dataset["name"],
                "version": dataset["version"],
                "feature_path": feature_path,
                "feature_log_path": log_path,
                "status": "featured"
            })
        except Exception as e:
            print(f"❌ Failed on {dataset['name']}: {e}")
            results.append({
                "name": dataset["name"],
                "version": dataset["version"],
                "status": "feature_failed",
                "error": str(e)
            })

    # Save summary of featured datasets
    logs_dir = script_dir / "../logs/features"
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary_out_path = logs_dir / "feature_summary.json"
    with open(summary_out_path, "w") as f:
        json.dump({"featured_datasets": [r for r in results if r["status"] == "featured"]}, f, indent=2)

    success_count = sum(1 for r in results if r["status"] == "featured")
    print(f"\nDone. {success_count}/{len(curated_datasets)} datasets feature engineered successfully.")


if __name__ == "__main__":
    run_feature_engineering()