'use client'

import { useEffect, useState } from 'react'
import { TriangleAlert, Info } from 'lucide-react'
import type { Source, VisualResource } from '@/lib/api'
import { SourceList } from '@/components/source-list'
import { VisualResourceList } from '@/components/visual-resource-list'

export interface ConversationTurn {
  id: string
  question: string
  status: 'loading' | 'answered' | 'error'
  answer?: string
  sources?: Source[]
  visualResources?: VisualResource[]
  /** The `error` field from a 200 response (partial internal failure). */
  partialError?: string | null
  /** Message for a hard failure (network / 500). */
  errorMessage?: string
}

function TurnLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="mb-2 font-mono text-[0.68rem] font-medium uppercase tracking-[0.18em] text-muted-foreground">
      {children}
    </div>
  )
}

// Honest progressive status text tied to the backend's real routing:
// it answers from Pinecone first and only falls back to a live web search
// when needed, which is the slow path.
const PHASES = [
  'Searching the knowledge base',
  'Consulting reference documents',
  'Checking live web sources',
]

function SearchingState() {
  const [phase, setPhase] = useState(0)

  useEffect(() => {
    if (phase >= PHASES.length - 1) return
    const id = setTimeout(() => setPhase((p) => p + 1), 4500)
    return () => clearTimeout(id)
  }, [phase])

  return (
    <div>
      <TurnLabel>Answer</TurnLabel>
      <p
        className="font-mono text-sm text-muted-foreground"
        role="status"
        aria-live="polite"
      >
        {PHASES[phase]}
        <span className="ml-1 inline-block w-4 text-left">
          <AnimatedEllipsis />
        </span>
      </p>
    </div>
  )
}

function AnimatedEllipsis() {
  const [dots, setDots] = useState('')
  useEffect(() => {
    const id = setInterval(() => {
      setDots((d) => (d.length >= 3 ? '' : d + '.'))
    }, 400)
    return () => clearInterval(id)
  }, [])
  return <span aria-hidden="true">{dots}</span>
}

function AnswerBody({ text }: { text: string }) {
  return (
    <div className="space-y-3 font-serif text-[1.02rem] leading-relaxed text-foreground">
      {text
        .split(/\n{2,}/)
        .filter((p) => p.trim().length > 0)
        .map((para, i) => (
          <p key={i} className="text-pretty whitespace-pre-line">
            {para}
          </p>
        ))}
    </div>
  )
}

export function ChatMessage({ turn }: { turn: ConversationTurn }) {
  return (
    <article className="border-b border-border/70 py-8 first:pt-2">
      {/* Question */}
      <div className="mb-6">
        <TurnLabel>Question</TurnLabel>
        <h2 className="text-balance font-serif text-xl font-medium leading-snug text-foreground">
          {turn.question}
        </h2>
      </div>

      {/* Response region */}
      {turn.status === 'loading' && <SearchingState />}

      {turn.status === 'error' && (
        <div
          role="alert"
          className="flex gap-3 rounded-md border border-destructive/40 bg-destructive/5 p-4 text-sm text-foreground"
        >
          <TriangleAlert
            className="mt-0.5 size-4 shrink-0 text-destructive"
            aria-hidden="true"
          />
          <div>
            <p className="font-medium text-destructive">Request failed</p>
            <p className="mt-1 leading-relaxed text-muted-foreground">
              {turn.errorMessage}
            </p>
          </div>
        </div>
      )}

      {turn.status === 'answered' && (
        <div>
          <TurnLabel>Answer</TurnLabel>

          {turn.partialError && (
            <div className="mb-4 flex gap-2 rounded-md border border-border bg-secondary/60 px-3 py-2 text-xs leading-relaxed text-muted-foreground">
              <Info className="mt-0.5 size-3.5 shrink-0" aria-hidden="true" />
              <span>
                Some retrieval steps did not complete, so this answer may be incomplete.
              </span>
            </div>
          )}

          <AnswerBody text={turn.answer ?? ''} />

          <VisualResourceList resources={turn.visualResources ?? []} />

          {turn.sources && turn.sources.length > 0 ? (
            <SourceList sources={turn.sources} />
          ) : (
            <div className="mt-5 border-t border-border pt-4">
              <p className="font-mono text-[0.72rem] uppercase tracking-[0.14em] text-muted-foreground">
                No sources cited
              </p>
              <p className="mt-2 max-w-prose text-sm italic leading-relaxed text-muted-foreground">
                The assistant found no reliable evidence for this question and
                answered without grounding in a specific document. Treat this
                response with extra caution.
              </p>
            </div>
          )}
        </div>
      )}
    </article>
  )
}
