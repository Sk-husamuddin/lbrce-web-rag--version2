'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { Library } from 'lucide-react'
import { askQuestion, ChatRequestError } from '@/lib/api'
import { HealthIndicator } from '@/components/health-indicator'
import { ChatMessage, type ConversationTurn } from '@/components/chat-message'
import { Composer } from '@/components/composer'
import { ThemeToggle } from '@/components/theme-toggle'

const SUGGESTIONS = [
  'Show the CSE V semester Section F timetable for 2026-27',
  'What is the distance between vijayawada and LBRCE?',
  'Show me the Students list of CSE 3rd year F section',
  'Who is the current HOD of CSE department ?',
]

export function ReferenceDesk() {
  const [turns, setTurns] = useState<ConversationTurn[]>([])
  const [pending, setPending] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [turns])

  const ask = useCallback(
    async (query: string) => {
      const id =
        typeof crypto !== 'undefined' && 'randomUUID' in crypto
          ? crypto.randomUUID()
          : `${Date.now()}-${Math.random()}`

      setTurns((prev) => [
        ...prev,
        { id, question: query, status: 'loading' },
      ])
      setPending(true)

      try {
        const res = await askQuestion(query)
        setTurns((prev) =>
          prev.map((t) =>
            t.id === id
              ? {
                  ...t,
                  status: 'answered',
                  answer: res.answer,
                  sources: res.sources,
                  visualResources: res.visual_resources,
                  partialError: res.error,
                }
              : t,
          ),
        )
      } catch (err) {
        const message =
          err instanceof ChatRequestError
            ? err.message
            : 'Something went wrong while contacting the service.'
        setTurns((prev) =>
          prev.map((t) =>
            t.id === id
              ? { ...t, status: 'error', errorMessage: message }
              : t,
          ),
        )
      } finally {
        setPending(false)
      }
    },
    [],
  )

  const isEmpty = turns.length === 0

  return (
    <div className="flex min-h-dvh min-w-0 flex-col overflow-x-hidden bg-background">
      <header className="sticky top-0 z-10 border-b border-border bg-background/85 backdrop-blur">
        <div className="mx-auto flex w-full max-w-3xl items-center justify-between gap-3 px-3 py-3 sm:gap-4 sm:px-5 sm:py-4">
          <div className="flex items-center gap-3">
            <span className="flex size-9 items-center justify-center rounded-md border border-border bg-secondary text-primary">
              <Library className="size-5" aria-hidden="true" />
            </span>
            <div className="leading-tight">
              <p className="font-serif text-base font-semibold text-foreground">
                LBRCE Reference Desk
              </p>
              <p className="font-mono text-[0.68rem] uppercase tracking-[0.14em] text-muted-foreground">
                Grounded answers with citations
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 sm:gap-3">
            <HealthIndicator />
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="mx-auto flex min-w-0 w-full max-w-3xl flex-1 flex-col px-3 pb-36 sm:px-5 sm:pb-0">
        {isEmpty ? (
          <div className="flex flex-1 flex-col justify-center py-10 sm:py-16">
            <h1 className="max-w-xl text-balance font-serif text-xl font-medium leading-snug text-foreground sm:text-2xl">
              Ask about Lakireddy Bali Reddy College of Engineering.
            </h1>
            <p className="mt-3 max-w-prose text-pretty leading-relaxed text-muted-foreground">
              Every answer is assembled from the college&apos;s published pages,
              documents, and — when needed — a live web search. Sources are
              listed with each response so you can verify them yourself.
            </p>

            <section className="mt-6 rounded-lg border border-border bg-card/70 p-3 sm:mt-8 sm:p-5" aria-labelledby="usage-guide-title">
              <div className="flex min-w-0 flex-col justify-between gap-3 sm:flex-row sm:items-start">
                <div>
                  <p className="font-mono text-[0.68rem] uppercase tracking-[0.18em] text-primary">
                    Reference desk guide
                  </p>
                  <h2 id="usage-guide-title" className="mt-2 font-serif text-lg font-medium text-foreground">
                    How to ask effective questions
                  </h2>
                  <p className="mt-2 max-w-prose text-sm leading-relaxed text-muted-foreground">
                    Ask about the indexed LBRCE knowledge base. For timetable questions,
                    include the department, semester, section, and academic year.
                  </p>
                </div>
                <span className="w-fit rounded-full border border-border bg-secondary px-3 py-1 font-mono text-[0.62rem] uppercase tracking-[0.12em] text-muted-foreground">
                  Grounded LBRCE sources
                </span>
              </div>

              <div className="mt-4 grid gap-3 sm:mt-5 md:grid-cols-3">
                <div className="rounded-md border border-border bg-background/70 p-4">
                  <p className="font-mono text-[0.65rem] font-semibold tracking-[0.12em] text-primary">01 · TIMETABLES</p>
                  <p className="mt-2 text-sm leading-relaxed text-foreground">
                    Ask for official 2026–27 department-wise timetables. Add details such as
                    CSE, V semester, Section F, and 2026–27. Matching images or PDFs appear
                    with the answer.
                  </p>
                </div>
                <div className="rounded-md border border-border bg-background/70 p-4">
                  <p className="font-mono text-[0.65rem] font-semibold tracking-[0.12em] text-primary">02 · R23 DOCUMENTS</p>
                  <p className="mt-2 text-sm leading-relaxed text-foreground">
                    Ask about the R23 regulation PDFs for B.Tech, M.Tech, MBA, and
                    Honors / Minors programmes.
                  </p>
                </div>
                <div className="rounded-md border border-border bg-background/70 p-4">
                  <p className="font-mono text-[0.65rem] font-semibold tracking-[0.12em] text-primary">03 · WEBPAGES</p>
                  <p className="mt-2 text-sm leading-relaxed text-foreground">
                    Ask about LBRCE webpages, departments, courses, admissions,
                    facilities, contacts, bus fares and other official college information.
                  </p>
                </div>
              </div>

              <p className="mt-4 text-xs leading-relaxed text-muted-foreground">
                The assistant uses indexed or retrieved LBRCE evidence. If an exact timetable
                section image is unavailable, it may show the relevant official semester PDF
                instead of inventing timetable details.
              </p>
            </section>

            <div className="mt-8">
              <p className="mb-3 font-mono text-[0.68rem] uppercase tracking-[0.18em] text-muted-foreground">
                Try asking
              </p>
              <ul className="flex flex-col gap-2">
                {SUGGESTIONS.map((s) => (
                  <li key={s}>
                    <button
                      type="button"
                      onClick={() => ask(s)}
                      disabled={pending}
                      className="w-full rounded-md border border-border bg-card px-4 py-3 text-left text-sm text-foreground transition-colors hover:border-primary/50 hover:bg-secondary disabled:opacity-50"
                    >
                      {s}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        ) : (
          <div className="flex-1 pt-2">
            {turns.map((turn) => (
              <ChatMessage key={turn.id} turn={turn} />
            ))}
          </div>
        )}
        <div ref={bottomRef} />
      </main>

      <div className="fixed inset-x-0 bottom-0 z-10 border-t border-border bg-background/95 backdrop-blur sm:sticky sm:inset-x-auto sm:bg-background/85">
        <div className="mx-auto w-full max-w-3xl px-3 py-3 sm:px-5 sm:py-4">
          <Composer onSubmit={ask} disabled={pending} />
          <p className="mt-2 text-center font-mono text-[0.66rem] uppercase tracking-[0.12em] text-muted-foreground">
            Each question is answered independently · no conversation history is
            stored <span aria-hidden="true">·</span>{' '}
            <span className="whitespace-nowrap">Created by SHAIK HUSAMUDDIN CSE-F/LBRCE</span>
          </p>
        </div>
      </div>
    </div>
  )
}
