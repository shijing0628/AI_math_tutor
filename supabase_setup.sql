-- Run this once in the Supabase SQL editor to enable cloud profile sync.
-- Then add to .env (and Streamlit Cloud secrets):
--   SUPABASE_URL=https://YOUR_PROJECT.supabase.co
--   SUPABASE_KEY=YOUR_PUBLISHABLE_OR_ANON_KEY

create table if not exists student_profiles (
    student text primary key,
    data jsonb not null,
    updated_at timestamptz not null default now()
);

-- Keep updated_at fresh on upserts
create or replace function touch_updated_at()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language plpgsql;

drop trigger if exists student_profiles_touch on student_profiles;
create trigger student_profiles_touch
before update on student_profiles
for each row execute function touch_updated_at();

-- Required for the app's publishable/anon API key to read and write.
-- Without these policies, inserts fail silently and the table stays empty.
alter table student_profiles enable row level security;

drop policy if exists "student_profiles_select" on student_profiles;
drop policy if exists "student_profiles_insert" on student_profiles;
drop policy if exists "student_profiles_update" on student_profiles;
drop policy if exists "student_profiles_delete" on student_profiles;

create policy "student_profiles_select"
on student_profiles for select
to anon, authenticated
using (true);

create policy "student_profiles_insert"
on student_profiles for insert
to anon, authenticated
with check (true);

create policy "student_profiles_update"
on student_profiles for update
to anon, authenticated
using (true)
with check (true);

create policy "student_profiles_delete"
on student_profiles for delete
to anon, authenticated
using (true);
