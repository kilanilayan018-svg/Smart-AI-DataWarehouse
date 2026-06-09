'use client'
import { useState } from 'react'
import { supabase } from '../../lib/supabase'

export default function LoginPage(){
 const [email,setEmail]=useState(''); const [password,setPassword]=useState(''); const [message,setMessage]=useState(''); const [loading,setLoading]=useState(false)
 async function signIn(){setLoading(true);setMessage(''); const {error}=await supabase.auth.signInWithPassword({email,password}); setLoading(false); setMessage(error?error.message:'Signed in. Go to Dashboard.')}
 async function signUp(){setLoading(true);setMessage(''); const {error}=await supabase.auth.signUp({email,password}); setLoading(false); setMessage(error?error.message:'Account created. Check email if confirmation is enabled, then sign in.')}
 async function signOut(){await supabase.auth.signOut(); setMessage('Signed out.')}
 return <main className="grid grid-2"><section className="card stack"><h1>Account access</h1><p className="muted">Use Supabase Auth so each user gets their own datasets, plans, and runs.</p><input className="input" placeholder="Email" value={email} onChange={e=>setEmail(e.target.value)} /><input className="input" type="password" placeholder="Password" value={password} onChange={e=>setPassword(e.target.value)} /><div style={{display:'flex',gap:10,flexWrap:'wrap'}}><button className="button" disabled={loading} onClick={signIn}>Sign in</button><button className="button secondary" disabled={loading} onClick={signUp}>Create account</button><button className="button secondary" onClick={signOut}>Sign out</button></div>{message&&<p className="muted">{message}</p>}</section><section className="card stack"><h2>Before using login</h2><p className="muted">Create a Supabase project, run <code>backend/supabase/schema.sql</code>, then add your anon key to <code>frontend/.env.local</code>.</p></section></main>
}
