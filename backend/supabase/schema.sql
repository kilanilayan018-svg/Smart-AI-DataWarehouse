create extension if not exists "uuid-ossp";

create table if not exists public.datasets (
  id uuid primary key default uuid_generate_v4(),
  owner_id uuid references auth.users(id) on delete cascade,
  original_filename text not null,
  stored_filename text,
  stored_path text,
  storage_bucket text default 'datasets',
  storage_path text,
  version_id text,
  file_type text,
  rows_count int default 0,
  columns_count int default 0,
  status text default 'uploaded',
  created_at timestamptz default now()
);

create table if not exists public.plans (
  id uuid primary key default uuid_generate_v4(),
  dataset_id uuid references public.datasets(id) on delete cascade,
  owner_id uuid references auth.users(id) on delete cascade,
  target_column text,
  task_type text,
  plan_source text default 'auto',
  plan jsonb not null default '{}'::jsonb,
  schema jsonb not null default '{}'::jsonb,
  validation jsonb not null default '{}'::jsonb,
  plan_path text,
  created_at timestamptz default now()
);

create table if not exists public.runs (
  id uuid primary key default uuid_generate_v4(),
  dataset_id uuid references public.datasets(id) on delete cascade,
  plan_id uuid references public.plans(id) on delete set null,
  owner_id uuid references auth.users(id) on delete cascade,
  status text not null default 'success',
  step text,
  message text,
  duration_ms int,
  created_at timestamptz default now()
);

alter table public.datasets enable row level security;
alter table public.plans enable row level security;
alter table public.runs enable row level security;

create policy "datasets select own" on public.datasets for select using (auth.uid() = owner_id);
create policy "datasets insert own" on public.datasets for insert with check (auth.uid() = owner_id);
create policy "datasets update own" on public.datasets for update using (auth.uid() = owner_id);
create policy "plans select own" on public.plans for select using (auth.uid() = owner_id);
create policy "plans insert own" on public.plans for insert with check (auth.uid() = owner_id);
create policy "runs select own" on public.runs for select using (auth.uid() = owner_id);
create policy "runs insert own" on public.runs for insert with check (auth.uid() = owner_id);

insert into storage.buckets (id, name, public) values ('datasets', 'datasets', false) on conflict (id) do nothing;
create policy "storage read own" on storage.objects for select using (bucket_id = 'datasets' and auth.uid()::text = (storage.foldername(name))[1]);
create policy "storage upload own" on storage.objects for insert with check (bucket_id = 'datasets' and auth.uid()::text = (storage.foldername(name))[1]);
