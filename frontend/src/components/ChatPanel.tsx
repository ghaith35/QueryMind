import { useEffect, useRef, useState } from 'react'
import { useAppStore } from '../store/appStore'
import { useShallow } from 'zustand/react/shallow'
import { sendChat } from '../api/endpoints'
import { MessageBubble } from './MessageBubble'

const starterPrompts = [
  'Summarize the main idea of this document set.',
  'Explain the most important findings in simple terms.',
  'Answer in Arabic and cite the supporting pages.',
]

export function ChatPanel() {
  const { sessionId, currentSetId, messages, addUserMessage, addAssistantPlaceholder, clearActivePath } = useAppStore(useShallow(s => ({
    sessionId: s.sessionId,
    currentSetId: s.currentSetId,
    messages: s.messages,
    addUserMessage: s.addUserMessage,
    addAssistantPlaceholder: s.addAssistantPlaceholder,
    clearActivePath: s.clearActivePath,
  })))

  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const submit = async () => {
    const text = input.trim()
    if (!text || !currentSetId || sending) return

    setInput('')
    setSending(true)
    clearActivePath()
    addUserMessage(text)

    try {
      const res = await sendChat(sessionId, currentSetId, text)
      addAssistantPlaceholder(res.turn_id)
    } catch (e: any) {
      const status = e?.response?.status
      const detail: string = e?.response?.data?.detail ?? ''
      const isQuota =
        status === 429 ||
        detail.toLowerCase().includes('resource_exhausted') ||
        detail.toLowerCase().includes('quota') ||
        detail.toLowerCase().includes('rate limit')
      useAppStore.getState().showError({
        code: 'LLM_FAILED',
        message: isQuota
          ? 'This version is a demo version, not a professional version.'
          : detail || 'Chat request failed',
        recoverable: true,
      })
    } finally {
      setSending(false)
    }
  }

  if (!currentSetId) {
    return (
      <div className="flex h-full items-center justify-center px-6">
        <div className="max-w-md text-center">
          <div className="text-xs font-mono uppercase tracking-[0.18em] text-[var(--text-2)]">
            Ready when you are
          </div>
          <p className="mt-4 text-sm leading-relaxed text-[var(--text-2)]">
            Select a document set on the left, upload your PDFs, then we can ask grounded questions with citations.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-[var(--border)] px-5 py-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="text-sm font-mono font-semibold uppercase tracking-[0.18em] text-[var(--text-2)]">
              Chat grounded in your set
            </div>
            <div className="mt-3 text-sm leading-relaxed text-[var(--text-1)]">
              Ask broad questions, follow up naturally, or switch languages mid-conversation.
            </div>
          </div>
          <div className="rounded-xl border border-[var(--border)] bg-[var(--surface-2)] px-3 py-2 text-xs font-mono text-[var(--text-2)]">
            {messages.length} message{messages.length === 1 ? '' : 's'}
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-5 py-5">
        {messages.length === 0 && (
          <div className="flex h-full flex-col items-center justify-center">
            <div className="max-w-xl text-center">
              <div className="text-xl font-semibold text-[var(--text-0)]">Let&apos;s interrogate the documents.</div>
              <p className="mt-4 text-sm leading-relaxed text-[var(--text-2)]">
                QueryMind keeps answers anchored to the retrieved chunks, so you can ask for summaries, comparisons, translation, or evidence-backed explanations.
              </p>
            </div>
            <div className="mt-6 flex w-full max-w-3xl flex-wrap justify-center gap-3">
              {starterPrompts.map(prompt => (
                <button
                  key={prompt}
                  onClick={() => setInput(prompt)}
                  className="rounded-2xl border border-[var(--border)] bg-[var(--surface-2)] px-4 py-3 text-left text-sm text-[var(--text-1)] transition-colors hover:border-[var(--accent-border)] hover:bg-[var(--surface-3)]"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map(m => <MessageBubble key={m.id} message={m} />)}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-[var(--border)] px-5 py-4">
        <div className="rounded-[1.5rem] border border-[var(--border)] bg-[var(--surface-2)] p-4 shadow-[var(--shadow-sm)] transition-colors focus-within:border-[var(--accent-border)]">
          <div className="mb-3 flex items-center justify-between px-1 text-xs font-mono font-semibold uppercase tracking-[0.18em] text-[var(--text-2)]">
            <span>Ask the set</span>
            <span>{sending ? 'Thinking...' : 'Enter to send'}</span>
          </div>
          <div className="flex gap-3">
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  void submit()
                }
              }}
              placeholder="Ask a grounded question, request a summary, or switch languages..."
              rows={3}
              className="min-h-[6rem] flex-1 resize-none bg-transparent px-1 text-base leading-relaxed text-[var(--text-1)] outline-none placeholder:text-[var(--text-2)]"
            />
            <button
              onClick={() => void submit()}
              disabled={!input.trim() || sending}
              className="self-end rounded-2xl border border-[var(--accent-border)] bg-[var(--accent-subtle)] px-5 py-3 text-sm font-semibold uppercase tracking-[0.18em] text-[var(--accent)] transition-colors hover:bg-[var(--accent-hover)] disabled:cursor-not-allowed disabled:opacity-40 whitespace-nowrap"
            >
              {sending ? 'Sending...' : 'Send'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
