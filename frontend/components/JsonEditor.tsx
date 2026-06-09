'use client'

type Props = {
  label: string
  value: string
  onChange: (value: string) => void
}

export default function JsonEditor({ label, value, onChange }: Props) {
  return (
    <div className="stack">
      <label>{label}</label>
      <textarea className="textarea" value={value} onChange={(e) => onChange(e.target.value)} spellCheck={false} />
    </div>
  )
}
