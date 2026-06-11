-- SEC Insight AI — user profiles, quotas, and Stripe linkage
-- Run in Supabase SQL Editor or via supabase db push

create table if not exists public.user_profiles (
  id uuid primary key references auth.users (id) on delete cascade,
  email text,
  plan text not null default 'free' check (plan in ('free', 'pro')),
  stripe_customer_id text,
  llm_calls_used integer not null default 0,
  llm_calls_limit integer not null default 3,
  daily_tokens_used integer not null default 0,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists user_profiles_stripe_customer_id_idx
  on public.user_profiles (stripe_customer_id);

alter table public.user_profiles enable row level security;

-- Users can read their own profile
create policy "Users read own profile"
  on public.user_profiles for select
  using (auth.uid() = id);

-- Service role (backend / webhooks) bypasses RLS when using service_role key

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer set search_path = public
as $$
begin
  insert into public.user_profiles (id, email, plan, llm_calls_used, llm_calls_limit)
  values (new.id, new.email, 'free', 0, 3)
  on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();
