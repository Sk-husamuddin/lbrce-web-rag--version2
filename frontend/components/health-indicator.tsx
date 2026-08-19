'use client'

import { useEffect, useState } from 'react'
import { checkHealth } from '@/lib/api'

type Status = 'checking' | 'online' | 'offline'

const LABEL: Record<Status, string> = {
  checking: 'Checking service',
  online: 'Service online',
  offline: 'Service unreachable',
}

const DOT: Record<Status, string> = {
  checking: 'bg-muted-foreground animate-pulse',
  online: 'bg-primary',
  offline: 'bg-destructive',
}

export function HealthIndicator() {
  const [status, setStatus] = useState<Status>('checking')

  useEffect(() => {
    let active = true
    const controller = new AbortController()

    const ping = async () => {
      const ok = await checkHealth(controller.signal)
      if (active) setStatus(ok ? 'online' : 'offline')
    }

    ping()
    const id = setInterval(ping, 30_000)

    return () => {
      active = false
      controller.abort()
      clearInterval(id)
    }
  }, [])

  return (
    <div
      className="inline-flex items-center gap-2 font-mono text-[0.68rem] uppercase tracking-[0.14em] text-muted-foreground"
      role="status"
      aria-live="polite"
    >
      <span className={`size-1.5 rounded-full ${DOT[status]}`} aria-hidden="true" />
      {LABEL[status]}
    </div>
  )
}
