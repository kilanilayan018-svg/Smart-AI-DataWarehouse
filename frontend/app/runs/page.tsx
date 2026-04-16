const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'

async function getRuns() {
  const res = await fetch(`${API_BASE}/runs`, { cache: 'no-store' })
  if (!res.ok) return { runs: [] }
  return res.json()
}

export default async function RunsPage() {
  const data = await getRuns()
  const runs = data.runs || []

  return (
    <div className="card stack">
      <h2 style={{margin:0}}>Saved runs</h2>
      <div className="muted">This page reads from SQLite so your users can revisit uploaded datasets and JSON plans later.</div>
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Dataset</th>
            <th>Target</th>
            <th>Task</th>
            <th>Status</th>
            <th>Output</th>
            <th>Created</th>
          </tr>
        </thead>
        <tbody>
          {runs.length === 0 ? (
            <tr><td colSpan={7} className="muted">No runs yet.</td></tr>
          ) : runs.map((run: any) => (
            <tr key={run.id}>
              <td>{run.id}</td>
              <td>{run.dataset_name}</td>
              <td>{run.target_column || '—'}</td>
              <td>{run.task_type || '—'}</td>
              <td>{run.status}</td>
              <td>{run.output_file || '—'}</td>
              <td>{run.created_at}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
