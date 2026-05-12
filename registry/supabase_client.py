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
    run_id = result.data[0]["run_id"]
    print(f"🔄 Run started: {dataset_name} (ID: {run_id})")
    return run_id


def update_run(run_id, status):
    """Update run status when done"""
    supabase.table("runs").update({
        "status": status,
        "completed_at": datetime.now().isoformat()
    }).eq("run_id", run_id).execute()
    print(f"✅ Run {run_id}: {status}")


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