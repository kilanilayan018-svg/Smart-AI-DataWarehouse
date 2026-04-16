import sqlite3
from pathlib import Path

DB_PATH = Path("registry/project_registry.db")


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_filename TEXT NOT NULL,
            stored_filename TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            version_id TEXT NOT NULL,
            file_type TEXT,
            rows_count INTEGER,
            columns_count INTEGER,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL,
            target_column TEXT,
            task_type TEXT,
            plan_source TEXT NOT NULL,
            plan_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (dataset_id) REFERENCES datasets(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL,
            plan_id INTEGER,
            status TEXT NOT NULL,
            step TEXT,
            message TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (dataset_id) REFERENCES datasets(id),
            FOREIGN KEY (plan_id) REFERENCES plans(id)
        )
    """)

    conn.commit()
    conn.close()


def insert_dataset(
    original_filename: str,
    stored_filename: str,
    stored_path: str,
    version_id: str,
    file_type: str,
    rows_count: int,
    columns_count: int,
    created_at: str,
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO datasets (
            original_filename,
            stored_filename,
            stored_path,
            version_id,
            file_type,
            rows_count,
            columns_count,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        original_filename,
        stored_filename,
        stored_path,
        version_id,
        file_type,
        rows_count,
        columns_count,
        created_at,
    ))

    conn.commit()
    dataset_id = cursor.lastrowid
    conn.close()
    return dataset_id


def insert_plan(
    dataset_id: int,
    target_column: str,
    task_type: str,
    plan_source: str,
    plan_path: str,
    created_at: str,
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO plans (
            dataset_id,
            target_column,
            task_type,
            plan_source,
            plan_path,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        dataset_id,
        target_column,
        task_type,
        plan_source,
        plan_path,
        created_at,
    ))

    conn.commit()
    plan_id = cursor.lastrowid
    conn.close()
    return plan_id


def insert_run(
    dataset_id: int,
    plan_id: int | None,
    status: str,
    step: str,
    message: str,
    created_at: str,
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO runs (
            dataset_id,
            plan_id,
            status,
            step,
            message,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        dataset_id,
        plan_id,
        status,
        step,
        message,
        created_at,
    ))

    conn.commit()
    run_id = cursor.lastrowid
    conn.close()
    return run_id


def fetch_recent_runs(limit: int = 10):
    conn = get_connection()

    rows = conn.execute("""
        SELECT
            runs.id,
            runs.status,
            runs.step,
            runs.message,
            runs.created_at,
            datasets.original_filename,
            datasets.stored_filename,
            plans.target_column,
            plans.task_type,
            plans.plan_source,
            plans.plan_path
        FROM runs
        JOIN datasets ON runs.dataset_id = datasets.id
        LEFT JOIN plans ON runs.plan_id = plans.id
        ORDER BY runs.id DESC
        LIMIT ?
    """, (limit,)).fetchall()

    conn.close()
    return [dict(row) for row in rows]