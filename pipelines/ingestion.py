from pathlib import Path
from datetime import datetime
import shutil
import uuid
import json

import pandas as pd


class DataIngestionModule:
    def __init__(self, raw_dir: str = "data/raw", metadata_dir: str = "metadata"):
        self.raw_dir = Path(raw_dir)
        self.metadata_dir = Path(metadata_dir)

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _read_csv_with_fallbacks(self, file_path: Path) -> pd.DataFrame:
        encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

        last_error = None
        for encoding in encodings_to_try:
            try:
                return pd.read_csv(
                    file_path,
                    encoding=encoding,
                    sep=None,
                    engine="python",
                    on_bad_lines="skip"
                )
            except Exception as e:
                last_error = e

        raise ValueError(
            f"Could not read CSV file '{file_path.name}'. Last error: {last_error}"
        )

    def _read_file(self, file_path: Path) -> pd.DataFrame:
        extension = file_path.suffix.lower()

        if extension == ".csv":
            return self._read_csv_with_fallbacks(file_path)

        if extension in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)

        raise ValueError("Unsupported file type. Only CSV and Excel are allowed.")

    def ingest(self, file_path: str) -> dict:
        source_path = Path(file_path)

        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = source_path.suffix.lower()
        if extension not in [".csv", ".xlsx", ".xls"]:
            raise ValueError("Unsupported file type. Only CSV and Excel are allowed.")

        version_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        stored_filename = f"{timestamp}_{version_id}_{source_path.name}"
        stored_path = self.raw_dir / stored_filename

        shutil.copy2(source_path, stored_path)

        df = self._read_file(stored_path)

        metadata = {
            "original_filename": source_path.name,
            "stored_filename": stored_filename,
            "stored_path": str(stored_path),
            "version_id": version_id,
            "timestamp": timestamp,
            "file_type": extension,
            "rows": int(df.shape[0]),
            "columns_count": int(df.shape[1]),
            "columns": df.columns.tolist(),
        }

        # 🔥 SAVE METADATA AS FILE
        metadata_file = self.metadata_dir / f"{timestamp}_{version_id}_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        return metadata