Place this backend/ folder in your repo root.
Run:
uvicorn app.main:app --reload --app-dir backend

It expects these existing repo modules to already exist at repo root:
- pipelines/ingestion.py
- pipelines/schema_extractor.py
- pipelines/validation.py
- pipelines/plan_generator.py

SQLite file will be created at registry/project_registry.db
