from pathlib import Path
from datetime import datetime
import json
import time
import uuid


class PipelineLogger:
    def __init__(self, log_file="logs/pipeline_events.jsonl"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def new_run_id(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:6]
        return f"run_{timestamp}_{short_id}"

    def log_event(
        self,
        run_id,
        step,
        status,
        start_time,
        dataset_name=None,
        version=None,
        error_detail=None,
        details=None
    ):
        duration_ms = int((time.perf_counter() - start_time) * 1000)

        event = {
            "run_id": run_id,
            "step": step,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "duration_ms": duration_ms,
            "error_detail": str(error_detail) if error_detail else None,
            "dataset_name": dataset_name,
            "version": version,
            "details": details or {}
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

        return event