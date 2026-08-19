'use client'

import { useRef, useState } from 'react'
import { CornerDownLeft } from 'lucide-react'

export function Composer({
  onSubmit,
  disabled,
}: {
  onSubmit: (query: string) => void
  disabled: boolean
}) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const submit = () => {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSubmit(trimmed)
    setValue('')
    // Reset the auto-grown height.
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Respect IME composition (CJK input) and Safari's 229 keyCode quirk.
    if (
      e.key === 'Enter' &&
      !e.shiftKey &&
      !e.nativeEvent.isComposing &&
      e.keyCode !== 229
    ) {
      e.preventDefault()
      submit()
    }
  }

  const autoGrow = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value)
    const el = e.target
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault()
        submit()
      }}
      className="flex min-w-0 flex-col items-stretch gap-2 rounded-lg border border-border bg-card p-2 shadow-sm focus-within:border-primary/60 focus-within:ring-1 focus-within:ring-primary/30 sm:flex-row sm:items-end sm:gap-3 sm:p-2.5"
    >
      <label htmlFor="question" className="sr-only">
        Ask a question about LBRCE
      </label>
      <textarea
        id="question"
        ref={textareaRef}
        rows={1}
        value={value}
        onChange={autoGrow}
        onKeyDown={handleKeyDown}
        placeholder="Ask about admissions, departments, facilities, placements…"
        className="max-h-[200px] min-h-14 w-full min-w-0 flex-1 resize-none bg-transparent px-2 py-2 font-sans text-[0.95rem] leading-relaxed text-foreground outline-none placeholder:text-muted-foreground sm:min-h-10 sm:py-1.5"
      />
      <button
        type="submit"
        disabled={disabled || value.trim().length === 0}
        className="inline-flex min-h-10 w-full shrink-0 items-center justify-center gap-2 rounded-md bg-primary px-4 py-2 font-sans text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-40 sm:w-auto"
      >
        {disabled ? 'Searching' : 'Ask'}
        <CornerDownLeft className="size-3.5" aria-hidden="true" />
      </button>
    </form>
  )
}
