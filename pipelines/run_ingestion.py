import pandas as pd
from pathlib import Path
from pipelines.ingestion import DataIngestionModule

ingestion = DataIngestionModule()
input_dir = Path("data/raw_input")

success_count = 0
failure_count = 0

for file in input_dir.iterdir():
    if file.is_file() and file.suffix.lower() in [".csv", ".xlsx", ".xls"]:
        print(f"\nProcessing: {file.name}")
        try:
            result = ingestion.ingest(str(file))
            print("SUCCESS")
            print(result)
            success_count += 1
        except Exception as e:
            print("FAILED")
            print(f"Error: {e}")
            failure_count += 1

print("\n=== SUMMARY ===")
print(f"Successful ingestions: {success_count}")
print(f"Failed ingestions: {failure_count}")