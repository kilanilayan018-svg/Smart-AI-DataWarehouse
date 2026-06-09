'use client'
import { useEffect,useState } from 'react'
import { apiGet } from '../../lib/api'

type Run={id:string,status:string,step?:string,message?:string,created_at?:string,dataset_id?:string,plan_id?:string,duration_ms?:number}
export default function RunsPage(){const [runs,setRuns]=useState<Run[]>([]);const [error,setError]=useState('');useEffect(()=>{apiGet('/runs?limit=50').then(d=>setRuns(d.runs||[])).catch(e=>setError(e.message))},[]);return <main className="card stack"><h1>Pipeline run history</h1><p className="muted">Every upload, automatic plan, and manual plan save appears here.</p>{error&&<p className="error">{error}</p>}<table><thead><tr><th>Status</th><th>Step</th><th>Message</th><th>Duration</th><th>Created</th></tr></thead><tbody>{runs.map(r=><tr key={r.id}><td><span className="badge">{r.status}</span></td><td>{r.step}</td><td>{r.message}</td><td>{r.duration_ms?`${r.duration_ms} ms`:'-'}</td><td>{r.created_at}</td></tr>)}</tbody></table>{!runs.length&&!error&&<p className="muted">No runs yet.</p>}</main>}
