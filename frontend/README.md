# Smart AI DataWarehouse Frontend

Copy the placeholder env file:

```bash
cp .env.local.example .env.local
```

Then fill:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
NEXT_PUBLIC_SUPABASE_URL=https://YOUR_PROJECT_REF.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=YOUR_SUPABASE_ANON_KEY
```

Run:

```bash
npm install
npm run dev
```

Pages included:

- `/` landing page
- `/login` Supabase auth page
- `/dashboard` run/dataset overview
- `/workspace` upload dataset, generate/edit plan, save plan
- `/runs` pipeline history
