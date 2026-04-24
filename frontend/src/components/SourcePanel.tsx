import { useEffect, useState } from 'react'
import { getChunkDetail } from '../api/endpoints'
import { useAppStore } from '../store/appStore'
import { useShallow } from 'zustand/react/shallow'
import type { ChunkDetail } from '../types/api'

function highlightExcerpt(text: string, excerpt?: string | null) {
  if (!excerpt) return text

  const normalizedExcerpt = excerpt.trim()
  if (!normalizedExcerpt) return text

  const lowerText = text.toLowerCase()
  const lowerExcerpt = normalizedExcerpt.toLowerCase()
  const start = lowerText.indexOf(lowerExcerpt)
  if (start === -1) return text

  const end = start + normalizedExcerpt.length
  return (
    <>
      {text.slice(0, start)}
      <mark>{text.slice(start, end)}</mark>
      {text.slice(end)}
    </>
  )
}

export function SourcePanel() {
  const { sourcePanelChunkId, sourcePanelCitation, closeSourcePanel } = useAppStore(useShallow(s => ({
    sourcePanelChunkId: s.sourcePanelChunkId,
    sourcePanelCitation: s.sourcePanelCitation,
    closeSourcePanel: s.closeSourcePanel,
  })))
  const [chunk, setChunk] = useState<ChunkDetail | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!sourcePanelChunkId) {
      setChunk(null)
      return
    }
    let cancelled = false
    setLoading(true)
    getChunkDetail(sourcePanelChunkId)
      .then((data) => {
        if (!cancelled) setChunk(data)
      })
      .catch(() => {
        if (!cancelled) setChunk(null)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [sourcePanelChunkId])

  if (!sourcePanelChunkId) return null

  const citation = sourcePanelCitation

  return (
    <div className="fixed inset-x-3 bottom-3 z-40 flex max-h-[78vh] flex-col rounded-[1.75rem] border border-[var(--border)] bg-[var(--surface-1)] shadow-2xl lg:inset-x-auto lg:bottom-auto lg:right-0 lg:top-0 lg:h-full lg:max-h-none lg:w-[26rem] lg:rounded-none lg:border-l">
      <div className="border-b border-[var(--border)] px-5 py-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="text-sm font-mono font-semibold uppercase tracking-[0.18em] text-[var(--text-2)]">
              Source panel
            </div>
            <div className="mt-2 text-sm text-[var(--text-1)]">
              Inspect the exact chunk behind the citation.
            </div>
          </div>
          <button
            onClick={closeSourcePanel}
            className="rounded-full border border-[var(--border)] px-2 py-1 text-xs font-mono font-semibold uppercase tracking-[0.18em] text-[var(--text-2)] transition-colors hover:border-[var(--accent-border)] hover:text-[var(--text-1)]"
          >
            Close
          </button>
        </div>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto p-5">
        {loading && (
          <div className="text-sm font-mono text-[var(--text-2)]">Loading source...</div>
        )}

        {chunk && (
          <>
            <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface-2)] p-4 shadow-[var(--shadow-sm)]">
              <div className="text-sm font-semibold text-[var(--accent)]">{chunk.document_name}</div>
              <div className="mt-3 flex flex-wrap gap-2 text-xs font-mono text-[var(--text-2)]">
                <span className="rounded-lg border border-[var(--border)] bg-[var(--surface-1)] px-2 py-1">Page {chunk.page_number}</span>
                <span className="rounded-lg border border-[var(--border)] bg-[var(--surface-1)] px-2 py-1">Paragraph {chunk.paragraph_index + 1}</span>
                {citation && (
                  <span className="rounded-lg border border-[var(--border)] bg-[var(--surface-1)] px-2 py-1">
                    Score {(citation.relevance_score * 100).toFixed(0)}%
                  </span>
                )}
              </div>
            </div>

            {citation && (
              <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface-2)] p-4 shadow-[var(--shadow-sm)]">
                <div className="text-[11px] font-mono uppercase tracking-[0.16em] text-[var(--text-2)]">
                  Cited excerpt
                </div>
                <div className="mt-3 whitespace-pre-wrap text-sm leading-relaxed text-[var(--text-1)]">
                  {citation.excerpt}
                </div>
              </div>
            )}

            <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface-2)] p-4 shadow-[var(--shadow-sm)]">
              <div className="text-[11px] font-mono uppercase tracking-[0.16em] text-[var(--text-2)]">
                Full chunk
              </div>
              <div dir="auto" className="mt-3 whitespace-pre-wrap text-sm leading-relaxed text-[var(--text-1)]">
                {highlightExcerpt(chunk.text, citation?.excerpt)}
              </div>
            </div>

            <button
              type="button"
              disabled
              className="w-full rounded-2xl border border-[var(--border)] bg-[var(--surface-2)] px-4 py-3 text-[11px] font-mono uppercase tracking-[0.14em] text-[var(--text-2)] opacity-75"
            >
              Open PDF at page {chunk.page_number} (coming soon)
            </button>
          </>
        )}
      </div>
    </div>
  )
}
