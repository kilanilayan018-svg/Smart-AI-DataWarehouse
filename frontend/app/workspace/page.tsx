'use client'
import { useMemo, useState } from 'react'
import JsonEditor from '../../components/JsonEditor'
import { API_BASE, authHeaders } from '../../lib/api'

type Summary = {
  run_id?: string | number
  dataset_id?: string | number
  plan_id?: string | number
  task_type?: string
  target_column?: string
  storage_path?: string
  curated_path?: string
  exec_status?: string
  download_ready?: boolean
  model_status?: { source?: string; model_enabled?: boolean; reason?: string }
}

export default function WorkspacePage() {
  const [file, setFile] = useState<File | null>(null)
  const [targetColumn, setTargetColumn] = useState('')
  const [loading, setLoading] = useState(false)
  const [downloading, setDownloading] = useState(false)
  const [error, setError] = useState('')
  const [schemaText, setSchemaText] = useState('{}')
  const [planText, setPlanText] = useState('{}')
  const [validationText, setValidationText] = useState('{}')
  const [datasetName, setDatasetName] = useState('')
  const [summary, setSummary] = useState<Summary | null>(null)

  const valid = useMemo(() => {
    try { JSON.parse(schemaText); JSON.parse(planText); return true } catch { return false }
  }, [schemaText, planText])

  async function upload() {
    if (!file) return
    setLoading(true)
    setError('')
    setSummary(null)
    try {
      const fd = new FormData()
      fd.append('file', file)
      if (targetColumn.trim()) fd.append('target_column', targetColumn.trim())
      const res = await fetch(`${API_BASE}/upload-and-plan`, {
        method: 'POST',
        headers: await authHeaders(),
        body: fd,
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setDatasetName(data.dataset_name || file.name.replace(/\.[^.]+$/, ''))
      setSchemaText(JSON.stringify(data.schema, null, 2))
      setPlanText(JSON.stringify(data.plan, null, 2))
      setValidationText(JSON.stringify(data.validation, null, 2))
      setSummary({
        run_id: data.run_id,
        dataset_id: data.dataset_id,
        plan_id: data.plan_id,
        task_type: data.task_type,
        target_column: data.target_column,
        storage_path: data.storage_path,
        curated_path: data.curated_path,
        exec_status: data.exec_status,
        download_ready: data.download_ready,
        model_status: data.model_status,
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unexpected error')
    } finally {
      setLoading(false)
    }
  }

  async function saveManual() {
    setLoading(true)
    setError('')
    try {
      const payload = {
        dataset_name: datasetName || file?.name?.replace(/\.[^.]+$/, '') || 'manual_dataset',
        target_column: targetColumn || null,
        schema: JSON.parse(schemaText),
        plan: JSON.parse(planText),
        execute_mode: 'plan_only',
      }
      const res = await fetch(`${API_BASE}/manual-plan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...(await authHeaders()) },
        body: JSON.stringify(payload),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setSummary(s => ({ ...s, run_id: data.run_id }))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unexpected error')
    } finally {
      setLoading(false)
    }
  }

  async function downloadCurated() {
    if (!datasetName) return
    setDownloading(true)
    setError('')
    try {
      const res = await fetch(`${API_BASE}/download-curated/${datasetName}`, {
        headers: await authHeaders(),
      })
      if (!res.ok) throw new Error(await res.text())
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${datasetName}_curated.csv`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Download failed')
    } finally {
      setDownloading(false)
    }
  }

  return (
    <main className="stack">
      <section className="grid grid-2">
        <div className="card stack">
          <span className="badge">Workspace</span>
          <h1>Upload, analyze, and plan</h1>
          <p className="muted">
            This connects the website to your Python pipeline: ingestion, schema extraction,
            validation, DeepSeek model-plan generation, plan execution, and Supabase metadata storage.
          </p>
          <input
            className="input"
            type="file"
            accept=".csv,.xlsx,.xls"
            onChange={e => setFile(e.target.files?.[0] || null)}
          />
          <input
            className="input"
            placeholder="Target column override (optional)"
            value={targetColumn}
            onChange={e => setTargetColumn(e.target.value)}
          />
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            <button className="button" onClick={upload} disabled={!file || loading}>
              {loading ? 'Processing...' : 'Generate AI plan'}
            </button>
            <button className="button secondary" onClick={saveManual} disabled={!valid || loading}>
              Save edited plan
            </button>
          </div>
          {error && <p className="error">{error}</p>}
        </div>

        <div className="card stack">
          <h2>Run summary</h2>
          {summary ? (
            <>
              <span className="pill">Run: {summary.run_id}</span>
              <span className="pill">Dataset: {summary.dataset_id}</span>
              <span className="pill">Plan: {summary.plan_id}</span>
              <span className="pill">Task: {summary.task_type || 'unknown'}</span>
              <span className="pill">Target: {summary.target_column || 'auto'}</span>
              {summary.storage_path && (
                <span className="pill">Supabase file: {summary.storage_path}</span>
              )}
              <span className="pill">
                Model: {summary.model_status?.source || 'not run yet'}
              </span>
              {summary.model_status?.reason && (
                <p className="muted">Fallback reason: {summary.model_status.reason}</p>
              )}

              <div style={{ marginTop: 12, paddingTop: 12, borderTop: '1px solid #e5e5e5' }}>
                <p style={{ fontWeight: 500, marginBottom: 8 }}>Processed CSV</p>
                {summary.download_ready ? (
                  <>
                    <p className="muted" style={{ marginBottom: 8 }}>
                      Plan executed successfully. Your curated dataset is ready.
                    </p>
                    <button
                      className="button"
                      onClick={downloadCurated}
                      disabled={downloading}
                      style={{ width: '100%' }}
                    >
                      {downloading ? 'Downloading...' : '⬇ Download processed CSV'}
                    </button>
                  </>
                ) : (
                  <p className="muted">
                    {summary.exec_status
                      ? `Execution status: ${summary.exec_status}`
                      : 'Not available yet — upload a dataset first.'}
                  </p>
                )}
              </div>
            </>
          ) : (
            <p className="muted">Upload a dataset to see run metadata.</p>
          )}
        </div>
      </section>

      <section className="grid grid-2">
        <div className="card">
          <JsonEditor label="Schema JSON" value={schemaText} onChange={setSchemaText} />
        </div>
        <div className="card">
          <JsonEditor label="Preprocessing Plan JSON" value={planText} onChange={setPlanText} />
        </div>
      </section>

      <section className="card">
        <JsonEditor label="Validation JSON" value={validationText} onChange={setValidationText} />
      </section>
    </main>
  )
}
