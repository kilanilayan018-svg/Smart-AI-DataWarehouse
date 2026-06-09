import { createClient } from '@supabase/supabase-js'

const url = process.env.NEXT_PUBLIC_SUPABASE_URL || ''
const anon = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || ''

function looksPlaceholder(value: string) {
  const v = (value || '').toLowerCase()
  return !v || v.includes('your_project_ref') || v.includes('your_') || v.includes('placeholder') || v.includes('replace_me')
}

function hasRealSupabaseConfig() {
  return url.startsWith('https://') && !looksPlaceholder(url) && !looksPlaceholder(anon) && anon.length > 20
}

const fallbackAuth = {
  async getSession() { return { data: { session: null }, error: null } },
  async signInWithPassword() { return { data: null, error: { message: 'Supabase Auth is not configured yet. This is okay for demo mode. Add real values to frontend/.env.local later.' } } },
  async signUp() { return { data: null, error: { message: 'Supabase Auth is not configured yet. This is okay for demo mode. Add real values to frontend/.env.local later.' } } },
  async signOut() { return { error: null } },
}

export const supabaseConfigured = hasRealSupabaseConfig()
export const supabase = supabaseConfigured
  ? createClient(url, anon)
  : ({ auth: fallbackAuth } as any)
