# registry/supabase_client.py
import os
from supabase import create_client
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(URL, KEY)


def upsert_dataset(dataset_id, rows, columns, task_type):
    """Insert or update dataset in Supabase"""

    data = {
        "dataset_id": dataset_id,
        "rows": rows,
        "columns": columns,
        "task_type": task_type,
        "created_at": datetime.now().isoformat()
    }

    # Check if exists
    existing = supabase.table("datasets").select("dataset_id").eq("dataset_id", dataset_id).execute()

    if existing.data:
        # Update existing
        update_data = {
            "rows": rows,
            "columns": columns,
            "task_type": task_type,
            "created_at": datetime.now().isoformat()
        }
        supabase.table("datasets").update(update_data).eq("dataset_id", dataset_id).execute()
        print(f"📊 Updated dataset: {dataset_id}")
    else:
        # Insert new
        supabase.table("datasets").insert(data).execute()
        print(f"📊 Inserted dataset: {dataset_id}")

    return True


def log_run(dataset_id, status):
    """Log pipeline run - returns run_id"""
    data = {
        "dataset_id": dataset_id,
        "status": status,
        "started_at": datetime.now().isoformat()
    }
    result = supabase.table("runs").insert(data).execute()
    run_id = result.data[0]["id"]
    print(f"🔄 Run started: {dataset_id} (ID: {run_id})")
    return run_id


def update_run(run_id, status):
    """Update run status when done"""
    supabase.table("runs").update({
        "status": status,
        "completed_at": datetime.now().isoformat()
    }).eq("id", run_id).execute()
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


def get_runs_for_dataset(dataset_id):
    """Get runs for a specific dataset"""
    result = supabase.table("runs").select("*").eq("dataset_id", dataset_id).execute()
    return result.data