import { supabase, supabaseConfigured } from './supabase'

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'

export async function authHeaders() {
  if (!supabaseConfigured) return {}
  const { data } = await supabase.auth.getSession()
  const token = data.session?.access_token
  return token ? { Authorization: `Bearer ${token}` } : {}
}

export async function apiGet(path: string) {
  const res = await fetch(`${API_BASE}${path}`, { headers: await authHeaders(), cache: 'no-store' })
  if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`)
  return res.json()
}
