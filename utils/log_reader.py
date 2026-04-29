import json
from pathlib import Path


class LogReader:
    def __init__(self, log_file="logs/pipeline_events.jsonl"):
        self.log_file = Path(log_file)

    def read_all(self):
        if not self.log_file.exists():
            return []

        records = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def filter(self, run_id=None, step=None, status=None):
        records = self.read_all()

        if run_id is not None:
            records = [r for r in records if r.get("run_id") == run_id]

        if step is not None:
            records = [r for r in records if r.get("step") == step]

        if status is not None:
            records = [r for r in records if r.get("status") == status]

        return records