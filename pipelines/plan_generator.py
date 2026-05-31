

import json
import sys
import os
from pathlib import Path
from typing import Dict, Optional, List, Tuple

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from registry.supabase_client import supabase

    SUPABASE_AVAILABLE = True
except Exception as e:
    supabase = None
    SUPABASE_AVAILABLE = False
    SUPABASE_IMPORT_ERROR = str(e)

from datetime import datetime
import pandas as pd
import numpy as np

# ============================================
# CONFIGURATION
# ============================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = PROJECT_ROOT / "data" / "schema"
PLAN_DIR = PROJECT_ROOT / "metadata" / "plans"
CURATED_DIR = PROJECT_ROOT / "data" / "curated"

PLAN_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# HELPER FUNCTIONS
# ============================================

def extract_short_name(long_name: str) -> str:
    """Convert long filename to short dataset name."""
    parts = long_name.split('_')
    meaningful = []
    for part in parts:
        if part.isdigit() and len(part) in (8, 6):
            continue
        if len(part) >= 8 and all(c in '0123456789abcdef' for c in part.lower()):
            continue
        meaningful.append(part)
    name = '_'.join(meaningful) if meaningful else long_name
    return name.lower().replace(' ', '_')


def get_target_from_datasets_table(dataset_name: str) -> Optional[str]:
    """Fetch target column from Supabase datasets table as fallback."""
    if supabase is None:
        return None
    try:
        result = supabase.table("datasets").select("target_column").eq("dataset_name", dataset_name).execute()
        if result.data and result.data[0].get("target_column"):
            return result.data[0]["target_column"]
    except Exception as e:
        print(f"   ⚠️ Could not fetch target from datasets: {e}")
    return None


def save_plan_to_supabase(dataset_name: str, plan_json: str) -> bool:
    """Save plan to Supabase - upsert (no duplicates)."""
    if supabase is None:
        print("   ⚠️ Supabase not available – skipping database save.")
        return False
    try:
        ds_result = supabase.table("datasets").select("dataset_id").eq("dataset_name", dataset_name).execute()
        if not ds_result.data:
            print(f"   ⚠️ Dataset '{dataset_name}' not found – plan not saved.")
            return False
        dataset_id = ds_result.data[0]["dataset_id"]

        now = datetime.now().isoformat()
        data = {
            "dataset_id": dataset_id,
            "plan_json": plan_json,
            "updated_at": now,
        }

        existing = supabase.table("plans").select("plan_id").eq("dataset_id", dataset_id).execute()

        if existing.data:
            plan_id = existing.data[0]["plan_id"]
            supabase.table("plans").update(data).eq("plan_id", plan_id).execute()
            print(f"   📋 Updated plan for '{dataset_name}' (plan_id={plan_id})")
        else:
            data["created_at"] = now
            supabase.table("plans").insert(data).execute()
            print(f"   📋 Inserted plan for '{dataset_name}' (dataset_id={dataset_id})")
        return True
    except Exception as e:
        print(f"   ❌ Supabase error: {e}")
        return False


class PlanGenerator:
    def __init__(self):
        self.numeric_dtypes = {"int64", "float64", "int32", "float32", "int", "float"}

    # ============================================
    # FIX 4.3: Low correlation with target (|corr| < 0.05)
    # ============================================
    def detect_low_correlation_columns(self, df: pd.DataFrame, target_col: str, threshold: float = 0.05) -> List[str]:
        """Find columns with low correlation to target (|corr| < threshold)."""
        low_corr_cols = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if col != target_col:
                try:
                    corr = df[col].corr(df[target_col])
                    if abs(corr) < threshold:
                        low_corr_cols.append(col)
                        print(f"   📉 Low correlation: '{col}' (corr={corr:.3f})")
                except:
                    pass
        return low_corr_cols

    # ============================================
    # FIX 4.4: Near-duplicate column detection (pairwise corr > 0.9)
    # ============================================
    def detect_high_correlation_columns(self, df: pd.DataFrame, threshold: float = 0.9) -> List[Dict]:
        """Find pairs of columns with correlation > threshold."""
        high_corr_pairs = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1:]:
                try:
                    corr = df[col1].corr(df[col2])
                    if abs(corr) > threshold:
                        high_corr_pairs.append({
                            "col1": col1,
                            "col2": col2,
                            "correlation": round(corr, 3)
                        })
                        print(f"   🔗 High correlation: '{col1}' ↔ '{col2}' (corr={corr:.3f})")
                except:
                    pass
        return high_corr_pairs

    # ============================================
    # FIX 4.5: SMOTE auto-enable (class ratio > 3:1)
    # ============================================
    def should_enable_smote(self, df: pd.DataFrame, target_col: str) -> Tuple[bool, str]:
        """Auto-enable SMOTE when class ratio > 3:1."""
        if target_col not in df.columns:
            return False, "Target column not found in dataframe"

        value_counts = df[target_col].value_counts()
        if len(value_counts) < 2:
            return False, f"Only {len(value_counts)} class(es) found (need at least 2)"

        majority = value_counts.max()
        minority = value_counts.min()
        ratio = majority / minority

        if ratio > 3:
            return True, f"Class imbalance detected: majority/minority ratio = {ratio:.2f}"
        return False, f"Class ratio = {ratio:.2f} (below 3:1 threshold)"

    # ============================================
    # MAIN PLAN GENERATION
    # ============================================
    def generate_plan(self, schema: Dict, dataset_name: str) -> Dict:
        """Generate complete preprocessing plan with all fixes."""

        # ============================================
        # STEP 1: Resolve target column
        # ============================================
        target_col = self._resolve_target(schema, dataset_name)
        if not target_col:
            raise ValueError(f"No target column for '{dataset_name}'")

        # ============================================
        # STEP 2: Load curated data for correlation analysis
        # ============================================
        df = None
        curated_path = CURATED_DIR / f"{dataset_name}_cleaned.csv"
        if curated_path.exists():
            try:
                df = pd.read_csv(curated_path)
                print(f"   📊 Loaded curated data: {df.shape}")
            except Exception as e:
                print(f"   ⚠️ Could not load curated data: {e}")

        # ============================================
        # STEP 3: Analyze columns from schema
        # ============================================
        numeric = []
        categorical = []
        impute_num = []
        impute_cat = []
        one_hot = []
        label_enc = []
        scale_cols = []
        log_transform = []
        drop_cols = []
        outlier_cap_cols = []
        skewed_columns_by_skewness = []

        for col, info in schema.items():
            if col in ("_meta", target_col):
                continue

            dtype = str(info.get("dtype", "")).lower()
            miss_pct = float(info.get("missing_percentage", 0) or 0)
            unique_cnt = int(info.get("unique_count", 0) or 0)
            skewness = info.get("skewness", 0)
            outlier_count = info.get("outlier_count", 0)
            min_val = info.get("min")

            # Track skewed columns using skewness (FIX 4.1)
            if abs(skewness) > 1.0:
                skewed_columns_by_skewness.append(col)

            # Track columns with outliers for capping (FIX 4.2)
            if outlier_count > 0:
                outlier_cap_cols.append(col)

            # Drop ID columns
            if self._is_id_column(col):
                drop_cols.append(col)
                continue

            if self._is_numeric(dtype):
                numeric.append(col)
                scale_cols.append(col)
                if miss_pct > 0:
                    impute_num.append(col)
                # FIX 4.1: Log transform based on skewness (not max/mean ratio)
                if abs(skewness) > 1.0 and min_val is not None and min_val >= 0:
                    log_transform.append(col)
            else:
                categorical.append(col)
                if miss_pct > 0:
                    impute_cat.append(col)
                # FIX 5.2: Unordered categorical -> one-hot (not label encode)
                if unique_cnt <= 15:
                    one_hot.append(col)
                else:
                    label_enc.append(col)

        # ============================================
        # STEP 4: Run correlation analysis (if data available)
        # ============================================
        low_correlation_columns = []
        high_correlation_pairs = []
        smote_enabled = False
        smote_reason = ""

        if df is not None and target_col in df.columns:
            # FIX 4.3: Low correlation detection
            low_correlation_columns = self.detect_low_correlation_columns(df, target_col, threshold=0.05)

            # FIX 4.4: High correlation detection
            high_correlation_pairs = self.detect_high_correlation_columns(df, threshold=0.9)

            # FIX 4.4: Add high-correlation columns to drop list (drop the second one)
            for pair in high_correlation_pairs:
                if pair["col2"] not in drop_cols:
                    drop_cols.append(pair["col2"])
                    print(f"   🗑️ Dropping '{pair['col2']}' (correlated {pair['correlation']} with '{pair['col1']}')")

            # FIX 4.5: SMOTE auto-enable
            smote_enabled, smote_reason = self.should_enable_smote(df, target_col)

        # ============================================
        # STEP 5: Determine task type and class count
        # ============================================
        task_type = self._infer_task_type(schema, target_col)
        class_count = None
        if task_type == "classification":
            class_count = int(schema.get(target_col, {}).get("unique_count", 0) or 0)
            if class_count == 0 and df is not None and target_col in df.columns:
                class_count = df[target_col].nunique()

        # ============================================
        # STEP 6: Build the plan
        # ============================================
        plan = {
            "dataset_name": dataset_name,
            "target_column": target_col,
            "task_type": task_type,
            "class_count": class_count,
            "preprocessing": {
                "drop_columns": drop_cols,
                "numeric_columns": numeric,
                "categorical_columns": categorical,
                "impute_numeric": impute_num,
                "impute_categorical": impute_cat,
                "one_hot_encode": one_hot,
                "label_encode": label_enc,
                "scale_columns": scale_cols,
                "log_transform": log_transform,
                # FIX 4.2: Outlier capping rule
                "outlier_capping": {
                    "enabled": True,
                    "method": "iqr",
                    "multiplier": 1.5,
                    "columns_with_outliers": outlier_cap_cols
                },
                # FIX 4.3: Low correlation drop rule
                "drop_low_correlation": {
                    "enabled": True,
                    "threshold": 0.05,
                    "columns": low_correlation_columns
                }
            },
            "split_strategy": {
                "test_size": 0.2,
                "random_state": 42,
                "stratify": task_type == "classification",
            },
            # FIX 4.5: SMOTE configuration
            "smote": {
                "enabled": smote_enabled,
                "reason": smote_reason,
                "would_apply_if": class_count and class_count < 1000
            },
            # Quality metrics for debugging and reporting
            "quality_metrics": {
                "skewed_columns_by_skewness": skewed_columns_by_skewness,
                "high_correlation_pairs": high_correlation_pairs,
                "outlier_columns_count": len(outlier_cap_cols),
                "low_correlation_columns": low_correlation_columns,
                "total_numeric_columns": len(numeric),
                "total_categorical_columns": len(categorical),
                "columns_dropped": len(drop_cols)
            }
        }

        return plan

    # ============================================
    # HELPER METHODS
    # ============================================

    def _resolve_target(self, schema: dict, dataset_name: str) -> Optional[str]:
        """Resolve target column using priority-based detection."""
        schema_keys = [k for k in schema if k != "_meta"]

        # Priority 1: From _meta
        meta = schema.get("_meta", {})
        if meta.get("target_column") in schema_keys:
            print(f"   🎯 Target from _meta: '{meta['target_column']}'")
            return meta["target_column"]

        # Priority 2: Keyword match
        keywords = ["target", "label", "class", "species", "price", "churn",
                    "attrition", "cardio", "survived", "mpg", "sales", "expenses"]
        for col in schema_keys:
            if col.lower().strip() in keywords:
                print(f"   🎯 Target by keyword: '{col}'")
                return col

        # Priority 3: Last non-ID column
        non_id = [c for c in schema_keys if not self._is_id_column(c)]
        if non_id:
            print(f"   🎯 Target fallback (last non-ID): '{non_id[-1]}'")
            return non_id[-1]

        # Priority 4: Fallback to datasets table
        db_target = get_target_from_datasets_table(dataset_name)
        if db_target and db_target in schema_keys:
            print(f"   🎯 Target from datasets table: '{db_target}'")
            return db_target

        print(f"   ⚠️ Could not resolve target column for {dataset_name}")
        return None

    def _infer_task_type(self, schema: Dict, target_col: str) -> str:
        """Infer classification vs regression from target column."""
        info = schema.get(target_col, {})
        dtype = str(info.get("dtype", "")).lower()
        uniq = int(info.get("unique_count", 0) or 0)

        if not self._is_numeric(dtype) or uniq <= 10:
            return "classification"
        return "regression"

    def _is_numeric(self, dtype: str) -> bool:
        return dtype in self.numeric_dtypes

    def _is_id_column(self, col: str) -> bool:
        low = col.strip().lower().replace(" ", "").replace("_", "")
        return low == "id" or low.endswith("id") or "passengerid" in low or "unnamed" in low

    def save_plan_local(self, plan: dict, dataset_name: str) -> str:
        """Save plan to local JSON file."""
        output_path = PLAN_DIR / f"{dataset_name}_plan.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2)
        return str(output_path)


# ============================================
# MAIN EXECUTION
# ============================================
def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PLAN GENERATOR - COMPLETE (ALL FIXES)")
    print("=" * 70)
    print("  ✓ Target detection (priority-based)")
    print("  ✓ Skewness-based log transform (skew > 1.0)")
    print("  ✓ Outlier capping (IQR, enabled by default)")
    print("  ✓ Low-correlation drop (|corr| < 0.05)")
    print("  ✓ Near-duplicate detection (corr > 0.9)")
    print("  ✓ SMOTE auto-enable (class ratio > 3:1)")
    print("=" * 70)

    if not SCHEMA_DIR.exists():
        print(f"\n❌ Schema directory not found: {SCHEMA_DIR}")
        raise SystemExit(1)

    schema_files = sorted(SCHEMA_DIR.glob("*_schema.json"))
    print(f"\n📁 Found {len(schema_files)} schema files.\n")

    generator = PlanGenerator()
    success = 0

    for sf in schema_files:
        print(f"📄 {sf.name}")
        try:
            long_name = sf.stem.replace("_schema", "")
            dataset_name = extract_short_name(long_name)
            schema = load_json(sf)

            # Generate plan
            plan = generator.generate_plan(schema, dataset_name)

            # Save locally
            local_path = generator.save_plan_local(plan, dataset_name)

            # Save to Supabase
            if save_plan_to_supabase(dataset_name, json.dumps(plan, indent=2)):
                success += 1
                print(f"   ✅ Target: {plan['target_column']} | Task: {plan['task_type']}")
                if plan.get('class_count'):
                    print(f"   ✅ Classes: {plan['class_count']}")
                if plan['preprocessing'].get('log_transform'):
                    print(f"   ✅ Log transform: {len(plan['preprocessing']['log_transform'])} columns")
                if plan['preprocessing'].get('outlier_capping', {}).get('enabled'):
                    print(f"   ✅ Outlier capping: ENABLED")
                if plan['preprocessing'].get('drop_low_correlation', {}).get('columns'):
                    print(
                        f"   ✅ Low correlation drops: {len(plan['preprocessing']['drop_low_correlation']['columns'])}")
                if plan.get('smote', {}).get('enabled'):
                    print(f"   ✅ SMOTE: {plan['smote']['reason']}")
            else:
                print(f"   ❌ Supabase save failed")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"✅ Complete: {success}/{len(schema_files)} plans processed")
    print(f"📁 Local plans: {PLAN_DIR}")
    print("📊 Supabase: plans table (one row per dataset, no duplicates)")
    print("=" * 70)
