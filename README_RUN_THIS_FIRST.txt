SMART AI DATAWAREHOUSE - RUN THIS FIRST
======================================

This rebuilt version is configured for your current best setup:

Frontend (Next.js)  ->  Backend (FastAPI)  ->  Public Colab/ngrok DeepSeek API
                                      ->  Local SQLite demo storage
                                      ->  Supabase later when you add real keys

IMPORTANT
---------
1. Keep your Colab/ngrok model API running:
   https://divisibly-wooing-smog.ngrok-free.dev/generate-plan

2. Supabase is optional right now.
   The placeholders in backend/.env will NOT crash the website.
   Until you add real Supabase keys, the app saves runs locally in SQLite.

3. You do NOT need to run the local model server on your 16GB RAM laptop.
   The backend calls the public ngrok API instead.

HOW TO RUN
----------
Double-click:

   run_website.bat

It opens two windows:

   Backend API:  http://127.0.0.1:8000/docs
   Frontend UI:  http://localhost:3000

OPTIONAL TESTS
--------------
After backend starts, double-click:

   check_backend.bat

To test the model API directly, double-click:

   test_model_api.bat

CONFIG FILES
------------
Backend config:

   backend/.env

Frontend config:

   frontend/.env.local

Current model API setting:

   MODEL_API_URL=https://divisibly-wooing-smog.ngrok-free.dev/generate-plan

When your ngrok URL changes, update backend/.env.

SUPABASE LATER
--------------
When you can access Supabase, replace these in backend/.env:

   SUPABASE_URL=https://your-real-project-ref.supabase.co
   SUPABASE_SERVICE_ROLE_KEY=your-real-service-role-key

And these in frontend/.env.local:

   NEXT_PUBLIC_SUPABASE_URL=https://your-real-project-ref.supabase.co
   NEXT_PUBLIC_SUPABASE_ANON_KEY=your-real-anon-key

Until then, local demo mode is intentional.
