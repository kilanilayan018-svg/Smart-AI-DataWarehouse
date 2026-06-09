import './globals.css'
import Link from 'next/link'
import type { ReactNode } from 'react'

export const metadata = { title: 'Smart AI Data Warehouse', description: 'AI-powered dataset preprocessing platform' }

export default function RootLayout({ children }: { children: ReactNode }) {
  return <html lang="en"><body><div className="container"><header className="header"><Link href="/" className="brand">Smart AI Data Warehouse</Link><nav className="nav"><Link href="/dashboard">Dashboard</Link><Link href="/workspace">Workspace</Link><Link href="/runs">Runs</Link><Link href="/login">Login</Link></nav></header>{children}</div></body></html>
}
