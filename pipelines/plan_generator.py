"""
Plan Generator - T1.6 (NO DUPLICATES)
- Uses plan_id as primary key
- Unique constraint on dataset_id ensures one plan per dataset
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from registry.supabase_client import supabase
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = PROJECT_ROOT / "data" / "schema"
PLAN_DIR = PROJECT_ROOT / "metadata" / "plans"
PLAN_DIR.mkdir(parents=True, exist_ok=True)

def get_target_from_datasets_table(dataset_name: str) -> Optional[str]:
    try:
        result = supabase.table("datasets").select("target_column").eq("dataset_name", dataset_name).execute()
        if result.data and result.data[0].get("target_column"):
            return result.data[0]["target_column"]
    except Exception as e:
        print(f"   ⚠️ Could not fetch target from datasets: {e}")
    return None

def extract_short_name(long_name: str) -> str:
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


def save_plan_to_supabase(dataset_name: str, plan_json: str) -> bool:
    try:
        # Get dataset_id from datasets table
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

        # Check existing plan by dataset_id
        existing = supabase.table("plans").select("plan_id").eq("dataset_id", dataset_id).execute()

        if existing.data:
            # Update existing plan using its plan_id
            plan_id = existing.data[0]["plan_id"]
            supabase.table("plans").update(data).eq("plan_id", plan_id).execute()
            print(f"   📋 Updated plan for '{dataset_name}' (plan_id={plan_id})")
        else:
            # Insert new plan
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

    def generate_plan(self, schema: Dict, dataset_name: str) -> Dict:
        target_col = self._resolve_target(schema, dataset_name)
        if not target_col:
            raise ValueError(f"No target column for '{dataset_name}'")

        numeric = []
        categorical = []
        impute_num = []
        impute_cat = []
        one_hot = []
        label_enc = []
        scale_cols = []
        log_transform = []
        drop_cols = []

        for col, info in schema.items():
            if col in ("_meta", target_col):
                continue

            dtype = str(info.get("dtype", "")).lower()
            miss_pct = float(info.get("missing_percentage", 0) or 0)
            unique_cnt = int(info.get("unique_count", 0) or 0)
            min_val = info.get("min")
            max_val = info.get("max")
            mean_val = info.get("mean")

            if self._is_id_column(col):
                drop_cols.append(col)
                continue

            if self._is_numeric(dtype):
                numeric.append(col)
                scale_cols.append(col)
                if miss_pct > 0:
                    impute_num.append(col)
                if self._should_log_transform(min_val, max_val, mean_val):
                    log_transform.append(col)
            else:
                categorical.append(col)
                if miss_pct > 0:
                    impute_cat.append(col)
                if unique_cnt <= 10:
                    one_hot.append(col)
                else:
                    label_enc.append(col)

        task_type = self._infer_task_type(schema, target_col)
        class_count = None
        if task_type == "classification":
            class_count = int(schema.get(target_col, {}).get("unique_count", 0) or 0)

        return {
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
            },
            "split_strategy": {
                "test_size": 0.2,
                "random_state": 42,
                "stratify": task_type == "classification",
            }
        }

    def _resolve_target(self, schema: dict, dataset_name: str) -> Optional[str]:
        schema_keys = [k for k in schema if k != "_meta"]

        # 1. From _meta
        meta = schema.get("_meta", {})
        if meta.get("target_column") in schema_keys:
            return meta["target_column"]

        # 2. Keyword match
        keywords = ["target", "label", "class", "species", "price", "churn",
                    "attrition", "cardio", "survived", "mpg", "sales"]
        for col in schema_keys:
            if col.lower().strip() in keywords:
                return col

        # 3. Last non-ID column
        non_id = [c for c in schema_keys if not self._is_id_column(c)]
        if non_id:
            return non_id[-1]

        # 4. **NEW: Fallback to datasets table**
        db_target = get_target_from_datasets_table(dataset_name)
        if db_target and db_target in schema_keys:
            print(f"   🎯 Target from datasets table: '{db_target}'")
            return db_target

        return None

    def _infer_task_type(self, schema: Dict, target_col: str) -> str:
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
        return low == "id" or low.endswith("id") or "passengerid" in low

    def _should_log_transform(self, min_val, max_val, mean_val) -> bool:
        try:
            if min_val is None or max_val is None or mean_val is None:
                return False
            min_v = float(min_val)
            max_v = float(max_val)
            mean_v = float(mean_val)
            if min_v < 0 or mean_v <= 0:
                return False
            return (max_v / mean_v) >= 3
        except:
            return False


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PLAN GENERATOR - NO DUPLICATES (UPDATES EXISTING)")
    print("=" * 60)

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

            plan = generator.generate_plan(schema, dataset_name)

            # Save locally
            local_path = PLAN_DIR / f"{dataset_name}_plan.json"
            with open(local_path, "w") as f:
                json.dump(plan, f, indent=2)

            # Save to Supabase (upsert)
            if save_plan_to_supabase(dataset_name, json.dumps(plan, indent=2)):
                success += 1
                print(f"   ✅ Target: {plan['target_column']} | Task: {plan['task_type']}")
                if plan.get('class_count'):
                    print(f"   ✅ Classes: {plan['class_count']}")
            else:
                print(f"   ❌ Supabase save failed")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n" + "=" * 60)
    print(f"✅ Complete: {success}/{len(schema_files)} plans processed")
    print(f"📁 Local plans: {PLAN_DIR}")
    print("📊 Supabase: plans table (one row per dataset_id, no duplicates)")
    print("=" * 60)