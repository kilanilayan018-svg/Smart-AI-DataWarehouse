# upload_plans_simple.py
import json
from pathlib import Path
from registry.supabase_client import supabase
from datetime import datetime

print("=" * 60)
print("UPLOADING PLANS TO SUPABASE")
print("=" * 60)

plan_files = list(Path(".").rglob("*_plan.json"))
print(f"\n📁 Found {len(plan_files)} plan files")

uploaded = 0
failed = 0

for plan_file in plan_files:
    try:
        with open(plan_file, 'r') as f:
            plan_data = json.load(f)

        dataset_id = plan_file.stem.replace('_plan', '')

        record = {
            "dataset_id": dataset_id,
            "plan_json": json.dumps(plan_data),
            "created_at": datetime.now().isoformat()
        }

        # Simple insert without on_conflict
        supabase.table("plans").insert(record).execute()
        print(f"   ✅ Uploaded: {dataset_id}")
        uploaded += 1

    except Exception as e:
        print(f"   ❌ Failed: {plan_file.name} - {e}")
        failed += 1

print(f"\n📊 Summary: {uploaded} uploaded, {failed} failed")
print("=" * 60)