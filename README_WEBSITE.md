# Smart AI Data Warehouse - Full Website Setup

This repo now contains a full-stack website around your existing data pipeline.

## Stack

- Frontend: Next.js 14 + React
- Backend: FastAPI
- Database: Supabase PostgreSQL
- Auth: Supabase Auth
- Storage: Supabase Storage bucket named `datasets`
- Pipeline: your existing `pipelines/` modules

## 1. Supabase setup

1. Create a Supabase project.
2. Open Supabase SQL Editor.
3. Run `backend/supabase/schema.sql`.
4. Copy your project URL, anon key, and service role key.

## 2. Backend setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `backend/.env` with your Supabase values.

Run from the repo root, not from `backend/`:

```bash
uvicorn backend.app.main:app --reload --port 8000
```

## 3. Frontend setup

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Open `http://localhost:3000`.

## Pages

- `/` landing page
- `/login` Supabase sign in / sign up
- `/dashboard` project dashboard
- `/workspace` upload dataset and generate/edit plan
- `/runs` pipeline run history

## Do not push generated data

The `.gitignore` excludes `.next`, `node_modules`, local DBs, generated datasets, metadata plans, and logs.
