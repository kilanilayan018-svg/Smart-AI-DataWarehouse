import time
from pathlib import Path

from pipelines.ingestion import DataIngestionModule
from utils.pipeline_logger import PipelineLogger


def main():
    ingestion = DataIngestionModule()
    logger = PipelineLogger()

    run_id = logger.new_run_id()

    print(f"\n🆔 Pipeline Run ID: {run_id}")

    input_dir = Path("data/raw_input")

    success_count = 0
    failure_count = 0

    for file in input_dir.iterdir():
        if file.is_file() and file.suffix.lower() in [".csv", ".xlsx", ".xls"]:
            print(f"\nProcessing: {file.name}")

            step_start_time = time.perf_counter()

            try:
                result = ingestion.ingest(str(file))

                logger.log_event(
                    run_id=run_id,
                    step="ingestion",
                    status="completed",
                    start_time=step_start_time,
                    dataset_name=file.stem,
                    version=result.get("version_id"),
                    error_detail=None,
                    details={
                        "original_filename": result.get("original_filename"),
                        "stored_filename": result.get("stored_filename"),
                        "stored_path": result.get("stored_path"),
                        "rows": result.get("rows"),
                        "columns_count": result.get("columns_count"),
                        "file_type": result.get("file_type")
                    }
                )

                print("SUCCESS")
                print(result)
                success_count += 1

            except Exception as e:
                logger.log_event(
                    run_id=run_id,
                    step="ingestion",
                    status="failed",
                    start_time=step_start_time,
                    dataset_name=file.stem,
                    version=None,
                    error_detail=str(e),
                    details={
                        "input_file": str(file).replace("\\", "/")
                    }
                )

                print("FAILED")
                print(f"Error: {e}")
                failure_count += 1

    print("\n=== SUMMARY ===")
    print(f"Successful ingestions: {success_count}")
    print(f"Failed ingestions: {failure_count}")
    print("📝 Pipeline events logged to: logs/pipeline_events.jsonl")


if __name__ == "__main__":
    main()