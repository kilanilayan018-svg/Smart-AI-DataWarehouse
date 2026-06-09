import './globals.css'
import Link from 'next/link'
import type { ReactNode } from 'react'

export const metadata = {
  title: 'Smart AI Data Warehouse',
  description: 'Upload datasets, review schema, edit JSON plans, and run the smart cleaning workflow.'
}

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="container">
          <div className="header">
            <div>
              <div className="brand">Smart AI Data Warehouse</div>
              <div className="muted">Dataset upload, schema analysis, JSON planning, and SQLite-backed run tracking.</div>
            </div>
            <div className="nav">
              <Link href="/">Workspace</Link>
              <Link href="/runs">Runs</Link>
            </div>
          </div>
          {children}
        </div>
      </body>
    </html>
  )
}
