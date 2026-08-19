import { ArrowUpRight, FileText, Globe, Code, Image as ImageIcon } from 'lucide-react'
import type { Source, SourceType } from '@/lib/api'

const TYPE_META: Record<
  SourceType,
  { label: string; Icon: typeof FileText }
> = {
  pdf: { label: 'PDF', Icon: FileText },
  web: { label: 'Web', Icon: Globe },
  html: { label: 'Page', Icon: Code },
  timetable_image: { label: 'Timetable image', Icon: ImageIcon },
}

function safeSourceUrl(value: string): string | null {
  try {
    const url = new URL(value)
    return url.protocol === 'https:' ? url.toString() : null
  } catch {
    return null
  }
}

function hostname(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, '')
  } catch {
    return url
  }
}

export function SourceList({ sources }: { sources: Source[] }) {
  const safeSources = sources
    .map((source) => ({ source, url: safeSourceUrl(source.url) }))
    .filter((item): item is { source: Source; url: string } => Boolean(item.url))

  if (safeSources.length === 0) return null

  return (
    <section aria-label="Sources" className="mt-5 border-t border-border pt-4">
      <h3 className="mb-3 font-mono text-[0.68rem] font-medium uppercase tracking-[0.18em] text-muted-foreground">
        Sources · {safeSources.length}
      </h3>
      <ol className="flex flex-col gap-3">
        {safeSources.map(({ source, url }, i) => {
          const meta = TYPE_META[source.source_type] ?? TYPE_META.web
          const { Icon } = meta
          return (
            <li key={`${url}-${i}`} className="flex gap-3 text-sm leading-relaxed">
              <span
                aria-hidden="true"
                className="mt-0.5 select-none font-mono text-xs tabular-nums text-muted-foreground"
              >
                {String(i + 1).padStart(2, '0')}
              </span>
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
                  <a
                    href={url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="group inline-flex min-w-0 max-w-full items-baseline gap-1 font-medium text-foreground decoration-border underline-offset-4 hover:text-primary hover:underline"
                  >
                    <span className="break-words">{source.title || hostname(url)}</span>
                    <ArrowUpRight
                      className="size-3.5 shrink-0 translate-y-0.5 text-muted-foreground transition-colors group-hover:text-primary"
                      aria-label="Opens in a new tab"
                    />
                  </a>
                </div>
                <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 font-mono text-[0.68rem] uppercase tracking-wide text-muted-foreground">
                  <span className="inline-flex items-center gap-1 rounded-sm border border-border bg-secondary px-1.5 py-0.5 text-secondary-foreground">
                    <Icon className="size-3" aria-hidden="true" />
                    {meta.label}
                  </span>
                  {source.source_type === 'pdf' && source.page != null && (
                    <span className="normal-case">p.&nbsp;{source.page}</span>
                  )}
                  <span className="min-w-0 max-w-full truncate lowercase tracking-normal">{hostname(url)}</span>
                </div>
              </div>
            </li>
          )
        })}
      </ol>
    </section>
  )
}
