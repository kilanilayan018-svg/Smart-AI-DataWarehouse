# registry/supabase_client.py
import os
from supabase import create_client
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(URL, KEY)


def upsert_dataset(dataset_name, original_filename, rows, columns, task_type, target_column=None):
    """Insert or update dataset in Supabase"""

    # Check if exists using dataset_name
    existing = supabase.table("datasets").select("dataset_name").eq("dataset_name", dataset_name).execute()

    data = {
        "dataset_name": dataset_name,
        "original_filename": original_filename,
        "rows": rows,
        "columns": columns,
        "task_type": task_type,
        "updated_at": datetime.now().isoformat()
    }

    # Add target_column if provided
    if target_column:
        data["target_column"] = target_column

    if existing.data:
        # Update existing (don't update created_at)
        supabase.table("datasets").update(data).eq("dataset_name", dataset_name).execute()
        print(f"📊 Updated dataset: {dataset_name}")
    else:
        # Insert new
        data["created_at"] = datetime.now().isoformat()
        supabase.table("datasets").insert(data).execute()
        print(f"📊 Inserted dataset: {dataset_name}")

    return True


def log_run(dataset_name, status):
    """Log pipeline run - returns run_id"""
    data = {
        "dataset_name": dataset_name,
        "status": status,
        "started_at": datetime.now().isoformat()
    }
    result = supabase.table("runs").insert(data).execute()

    # IMPORTANT: Your table uses 'run_id' as primary key
    # The insert returns the value under 'run_id'
    run_id = result.data[0].get("run_id")

    if run_id is None:
        # Fallback: try 'id' if 'run_id' doesn't exist
        run_id = result.data[0].get("id")

    print(f"🔄 Run started: {dataset_name} (ID: {run_id})")
    return run_id


def update_run(run_id, status):
    """Update run status when done"""
    if run_id is None:
        print("⚠️ Cannot update: run_id is None")
        return False

    result = supabase.table("runs").update({
        "status": status,
        "completed_at": datetime.now().isoformat()
    }).eq("run_id", run_id).execute()

    print(f"✅ Run {run_id}: {status}")
    return True


def log_metrics(run_id, model_name, accuracy, f1_score):
    """Log model metrics"""
    data = {
        "run_id": run_id,
        "model_name": model_name,
        "accuracy": accuracy,
        "f1_score": f1_score
    }
    supabase.table("metrics").insert(data).execute()
    print(f"📈 Metrics logged: {model_name}")


def get_all_datasets():
    """Get all datasets"""
    result = supabase.table("datasets").select("*").execute()
    return result.data


def get_all_runs():
    """Get all runs"""
    result = supabase.table("runs").select("*").execute()
    return result.data

def get_dataset_id(dataset_name):
    """Convert dataset name to integer dataset_id"""
    result = supabase.table("datasets").select("dataset_id").eq("dataset_name", dataset_name).execute()
    return result.data[0]["dataset_id"] if result.data else None


def save_plan_to_supabase(dataset_name: str, plan_json: str) -> bool:
    try:
        dataset_id = get_dataset_id(dataset_name)
        if not dataset_id:
            print(f"   ⚠️ Dataset '{dataset_name}' not found")
            return False

        data = {
            "dataset_id": dataset_id,
            "plan_json": plan_json,
            "updated_at": datetime.now().isoformat()
        }

        existing = supabase.table("plans").select("id").eq("dataset_id", dataset_id).execute()

        if existing.data:
            supabase.table("plans").update(data).eq("dataset_id", dataset_id).execute()
            print(f"   📋 Updated plan for dataset_id: {dataset_id}")
        else:
            data["created_at"] = datetime.now().isoformat()
            supabase.table("plans").insert(data).execute()
            print(f"   📋 Inserted plan for dataset_id: {dataset_id}")

        return True
    except Exception as e:
        print(f"   ⚠️ Error: {e}")
        return False