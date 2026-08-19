import { ArrowUpRight, FileText, Image as ImageIcon } from 'lucide-react'
import type { VisualResource } from '@/lib/api'

const ALLOWED_HOSTS = new Set(['lbrce.ac.in', 'www.lbrce.ac.in'])

export function safeResourceUrl(value: string): string | null {
  try {
    const url = new URL(value)
    if (url.protocol !== 'https:') return null
    const hostname = url.hostname.toLowerCase()
    if (ALLOWED_HOSTS.has(hostname) || hostname.endsWith('.lbrce.ac.in')) {
      return url.toString()
    }
    return null
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

function ResourceImage({ resource, url }: { resource: VisualResource; url: string }) {
  return (
    // External timetable URLs are rendered directly; Next Image would require
    // a fixed remote-image allowlist or a proxy for these changing URLs.
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={url}
      alt={resource.title}
      loading="lazy"
      referrerPolicy="no-referrer"
      className="max-h-[32rem] max-w-full rounded-md border border-border bg-secondary object-contain"
    />
  )
}

function PdfPreview({ resource, url }: { resource: VisualResource; url: string }) {
  return (
    <div className="overflow-hidden rounded-md border border-border bg-secondary">
      <iframe
        src={url}
        title={resource.title}
        loading="lazy"
        className="h-[32rem] w-full bg-background"
      />
      <p className="border-t border-border px-3 py-2 text-xs text-muted-foreground">
        If the preview does not load, use the full PDF link below.
      </p>
    </div>
  )
}

export function VisualResourceList({ resources }: { resources: VisualResource[] }) {
  const safeResources = resources
    .map((resource) => ({ resource, url: safeResourceUrl(resource.url) }))
    .filter((item): item is { resource: VisualResource; url: string } => Boolean(item.url))

  if (safeResources.length === 0) return null

  return (
    <section aria-label="Related visual resources" className="mt-6 border-t border-border pt-4">
      <h3 className="mb-3 font-mono text-[0.68rem] font-medium uppercase tracking-[0.18em] text-muted-foreground">
        Related resources · {safeResources.length}
      </h3>
      <div className="flex flex-col gap-5">
        {safeResources.map(({ resource, url }) => (
          <article key={`${resource.type}-${url}`} className="min-w-0 overflow-hidden rounded-md border border-border bg-card p-2.5 sm:p-3">
            <div className="mb-3 flex min-w-0 items-start justify-between gap-2 sm:gap-3">
              <div className="min-w-0">
                <h4 className="break-words font-medium text-foreground">{resource.title || hostname(url)}</h4>
                <p className="mt-1 font-mono text-[0.68rem] uppercase tracking-wide text-muted-foreground">
                  {resource.type === 'image' ? 'Timetable image' : 'PDF document'} · {hostname(url)}
                </p>
                {(resource.academic_year || resource.semester || resource.section || resource.term) && (
                  <div className="mt-2 flex flex-wrap gap-1.5 font-mono text-[0.66rem] uppercase tracking-wide text-muted-foreground">
                    {resource.academic_year && <span className="rounded-sm bg-secondary px-1.5 py-0.5">A.Y. {resource.academic_year}</span>}
                    {resource.term && <span className="rounded-sm bg-secondary px-1.5 py-0.5">{resource.term}</span>}
                    {resource.semester && <span className="rounded-sm bg-secondary px-1.5 py-0.5">Semester {resource.semester}</span>}
                    {resource.section && <span className="rounded-sm bg-secondary px-1.5 py-0.5">Section {resource.section}</span>}
                  </div>
                )}
              </div>
              {resource.type === 'image' ? (
                <ImageIcon className="mt-0.5 size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
              ) : (
                <FileText className="mt-0.5 size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
              )}
            </div>

            {resource.type === 'image' ? (
              <ResourceImage resource={resource} url={url} />
            ) : (
              <PdfPreview resource={resource} url={url} />
            )}

            <a
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-3 inline-flex items-center gap-1 text-sm font-medium text-foreground underline decoration-border underline-offset-4 hover:text-primary"
            >
              Open {resource.type === 'image' ? 'full-size image' : 'PDF'}
              <ArrowUpRight className="size-3.5" aria-hidden="true" />
            </a>
          </article>
        ))}
      </div>
    </section>
  )
}
