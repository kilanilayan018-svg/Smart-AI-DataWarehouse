'use client'

import { useMemo, useState } from 'react'
import JsonEditor from '../components/JsonEditor'

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null)
  const [targetColumn, setTargetColumn] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [schemaText, setSchemaText] = useState('{}')
  const [planText, setPlanText] = useState('{}')
  const [validationText, setValidationText] = useState('{}')
  const [datasetName, setDatasetName] = useState('')
  const [summary, setSummary] = useState<{run_id?: number; task_type?: string; target_column?: string} | null>(null)

  const isJsonValid = useMemo(() => {
    try {
      JSON.parse(schemaText)
      JSON.parse(planText)
      return true
    } catch {
      return false
    }
  }, [schemaText, planText])

  async function handleAutoPlan() {
    if (!file) return
    setLoading(true)
    setError('')
    try {
      const formData = new FormData()
      formData.append('file', file)
      if (targetColumn.trim()) formData.append('target_column', targetColumn.trim())

      const res = await fetch(`${API_BASE}/upload-and-plan`, { method: 'POST', body: formData })
      if (!res.ok) throw new Error(`Request failed: ${res.status}`)
      const data = await res.json()
      setDatasetName(data.dataset_name || file.name.replace(/\.[^.]+$/, ''))
      setSchemaText(JSON.stringify(data.schema, null, 2))
      setPlanText(JSON.stringify(data.plan, null, 2))
      setValidationText(JSON.stringify(data.validation, null, 2))
      setSummary({ run_id: data.run_id, task_type: data.task_type, target_column: data.target_column })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unexpected error')
    } finally {
      setLoading(false)
    }
  }

  async function handleSaveManualPlan() {
    setLoading(true)
    setError('')
    try {
      const payload = {
        dataset_name: datasetName || (file?.name?.replace(/\.[^.]+$/, '') || 'manual_dataset'),
        target_column: targetColumn || null,
        schema: JSON.parse(schemaText),
        plan: JSON.parse(planText),
        execute_mode: 'plan_only'
      }
      const res = await fetch(`${API_BASE}/manual-plan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
      if (!res.ok) throw new Error(`Save failed: ${res.status}`)
      const data = await res.json()
      setSummary((s) => ({ ...(s || {}), run_id: data.run_id }))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unexpected error')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="stack">
      <div className="kpi">
        <div className="card"><h3>Upload</h3><p>CSV / Excel</p></div>
        <div className="card"><h3>Plan Mode</h3><p>Auto + Manual</p></div>
        <div className="card"><h3>Storage</h3><p>SQLite</p></div>
        <div className="card"><h3>Pipeline</h3><p>Repo-linked</p></div>
      </div>

      <div className="grid grid-2">
        <div className="card stack">
          <h2 style={{margin:'0 0 4px'}}>Create or edit a smart cleaning run</h2>
          <div className="muted">Upload a dataset to auto-generate schema and plan, then refine the JSON manually before saving it for the rest of the project workflow.</div>
          <input className="input" type="file" accept=".csv,.xlsx,.xls" onChange={(e) => setFile(e.target.files?.[0] || null)} />
          <input className="input" placeholder="Target column (optional override)" value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)} />
          <div style={{display:'flex', gap:12, flexWrap:'wrap'}}>
            <button className="button" onClick={handleAutoPlan} disabled={!file || loading}>{loading ? 'Working...' : 'Auto-generate schema and plan'}</button>
            <button className="button secondary" onClick={handleSaveManualPlan} disabled={!isJsonValid || loading}>Save manual JSON plan</button>
          </div>
          {summary && (
            <div style={{display:'flex', gap:10, flexWrap:'wrap'}}>
              {summary.run_id ? <span className="badge">Run #{summary.run_id}</span> : null}
              {summary.target_column ? <span className="badge">Target: {summary.target_column}</span> : null}
              {summary.task_type ? <span className="badge">Task: {summary.task_type}</span> : null}
            </div>
          )}
          {error ? <div style={{color:'#fca5a5'}}>{error}</div> : null}
        </div>

        <div className="card stack">
          <h2 style={{margin:'0 0 4px'}}>How this page fits your project</h2>
          <ul className="muted" style={{margin:0, paddingLeft:18, lineHeight:1.8}}>
            <li>Uploads feed your existing ingestion module.</li>
            <li>Schema comes from your current schema extractor.</li>
            <li>Validation uses your validation module.</li>
            <li>Auto-plan uses your current PlanGenerator.</li>
            <li>Users can override or author their own JSON plan before the smart pipeline executes later.</li>
            <li>Each run is recorded in SQLite for future runs/history pages.</li>
          </ul>
        </div>
      </div>

      <div className="grid grid-2">
        <div className="card">
          <JsonEditor label="Schema JSON" value={schemaText} onChange={setSchemaText} />
        </div>
        <div className="card">
          <JsonEditor label="Plan JSON" value={planText} onChange={setPlanText} />
        </div>
      </div>

      <div className="card">
        <JsonEditor label="Validation JSON" value={validationText} onChange={setValidationText} />
      </div>
    </div>
  )
}
