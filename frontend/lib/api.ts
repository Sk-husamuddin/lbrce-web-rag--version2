// Client for the FastAPI + LangGraph RAG backend.
// The base URL is configurable via NEXT_PUBLIC_API_BASE_URL and defaults to
// the backend's local dev address. Requests go directly from the browser to
// the backend, so the backend's CORS_ORIGINS must include this frontend origin.

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, '') ??
  'http://localhost:8000'

export type SourceType = 'html' | 'pdf' | 'web' | 'timetable_image'

export interface Source {
  title: string
  url: string
  page: number | null
  source_type: SourceType
}

export type VisualResourceType = 'image' | 'pdf'

export interface VisualResource {
  title: string
  url: string
  type: VisualResourceType
  department?: string | null
  academic_year?: string | null
  semester?: string | null
  section?: string | null
  term?: string | null
}

export interface ChatResponse {
  answer: string
  sources: Source[]
  visual_resources: VisualResource[]
  // Present even on 200 if the agent had a partial internal failure but
  // still produced an answer.
  error: string | null
}

// Thrown for hard failures: network errors or a 500 from the backend
// (which returns FastAPI's { detail } shape, not a ChatResponse).
export class ChatRequestError extends Error {
  constructor(
    message: string,
    public kind: 'network' | 'server',
  ) {
    super(message)
    this.name = 'ChatRequestError'
  }
}

function normalizeSources(value: unknown): Source[] {
  if (!Array.isArray(value)) return []
  return value.filter((item): item is Source => {
    if (!item || typeof item !== 'object') return false
    const source = item as Partial<Source>
    return typeof source.title === 'string' && typeof source.url === 'string'
  }).map((source) => ({
    title: source.title,
    url: source.url,
    page: typeof source.page === 'number' ? source.page : null,
    source_type:
      source.source_type === 'pdf' ||
      source.source_type === 'html' ||
      source.source_type === 'web' ||
      source.source_type === 'timetable_image'
        ? source.source_type
        : 'web',
  }))
}

function normalizeVisualResources(value: unknown): VisualResource[] {
  if (!Array.isArray(value)) return []
  return value.filter((item): item is VisualResource => {
    if (!item || typeof item !== 'object') return false
    const resource = item as Partial<VisualResource>
    return (
      typeof resource.title === 'string' &&
      typeof resource.url === 'string' &&
      (resource.type === 'image' || resource.type === 'pdf')
    )
  }).map((resource) => ({
    title: resource.title,
    url: resource.url,
    type: resource.type,
    department: resource.department ?? null,
    academic_year: resource.academic_year ?? null,
    semester: resource.semester ?? null,
    section: resource.section ?? null,
    term: resource.term ?? null,
  }))
}

function normalizeChatResponse(value: unknown): ChatResponse {
  const body = value && typeof value === 'object'
    ? (value as Record<string, unknown>)
    : {}

  return {
    answer: typeof body.answer === 'string' ? body.answer : '',
    sources: normalizeSources(body.sources),
    visual_resources: normalizeVisualResources(body.visual_resources),
    error: typeof body.error === 'string' ? body.error : null,
  }
}

export async function askQuestion(query: string): Promise<ChatResponse> {
  let res: Response
  try {
    res = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    })
  } catch {
    throw new ChatRequestError(
      'Could not reach the knowledge service. Check that the backend is running and that this origin is allowed by its CORS settings.',
      'network',
    )
  }

  if (!res.ok) {
    let detail = `The service responded with an error (${res.status}).`
    try {
      const body = (await res.json()) as { detail?: string }
      if (body?.detail) detail = body.detail
    } catch {
      // Non-JSON error body; keep the generic message.
    }
    throw new ChatRequestError(detail, 'server')
  }

  return normalizeChatResponse(await res.json())
}

export async function checkHealth(signal?: AbortSignal): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, { signal })
    if (!res.ok) return false
    const body = (await res.json()) as { status?: string }
    return body?.status === 'ok'
  } catch {
    return false
  }
}
