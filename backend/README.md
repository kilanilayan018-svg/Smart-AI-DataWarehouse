# Smart AI DataWarehouse Backend

This backend is now wired for the intended architecture:

```text
Next.js website
  -> FastAPI backend (:8000)
  -> pipeline schema extraction + validation
  -> optional DeepSeek LoRA model server (:8001)
  -> Supabase tables/storage
```

## 1. Configure placeholders

Copy the example env file and replace the placeholder values locally:

```bash
cp backend/.env.example backend/.env
```

Required placeholders:

```env
SUPABASE_URL=https://YOUR_PROJECT_REF.supabase.co
SUPABASE_ANON_KEY=YOUR_SUPABASE_ANON_KEY
SUPABASE_SERVICE_ROLE_KEY=YOUR_SUPABASE_SERVICE_ROLE_KEY
SUPABASE_STORAGE_BUCKET=datasets
MODEL_API_URL=http://localhost:8001/generate-plan
```

Never put the service role key in the frontend.

## 2. Create Supabase tables

Run `backend/supabase/schema.sql` in the Supabase SQL editor.

## 3. Start the main backend

From the repo root:

```bash
pip install -r backend/requirements.txt
uvicorn app.main:app --reload --app-dir backend --port 8000
```

The backend expects these existing repo modules at repo root:

- `pipelines/ingestion.py`
- `pipelines/schema_extractor.py`
- `pipelines/validation.py`
- `pipelines/plan_generator.py`

## 4. Optional: start the DeepSeek model server

Put your `deepseek_stage2/` adapter folder beside the repo root or set `ADAPTER_PATH`.

```bash
pip install -r backend/model_server/requirements-model.txt
uvicorn backend.model_server.fastapi_model_server:app --host 0.0.0.0 --port 8001
```

The main backend calls `MODEL_API_URL`. If the model server is not configured or fails, the backend falls back to the rule-based `PlanGenerator`, so demos still work.

## 5. Upload flow

```text
/upload-and-plan
  1. receives CSV/Excel
  2. runs ingestion
  3. extracts schema
  4. validates data quality
  5. converts schema JSON into the DeepSeek prompt format
  6. calls MODEL_API_URL for the AI plan
  7. falls back to rule-based planning if needed
  8. saves dataset/plan/run metadata to Supabase or local SQLite
```
