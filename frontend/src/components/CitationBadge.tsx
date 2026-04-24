import { useState } from 'react'
import { useAppStore } from '../store/appStore'
import type { Citation } from '../types/api'

export function CitationBadge({ citation }: { citation: Citation }) {
  const [hovering, setHovering] = useState(false)
  const openSourcePanel = useAppStore(s => s.openSourcePanel)

  return (
    <span className="relative inline-block align-middle">
      <span
        className="inline-flex cursor-pointer select-none items-center gap-1 rounded-lg border border-[var(--accent-border)] bg-[var(--accent-subtle)] px-2 py-1 text-xs font-semibold text-[var(--accent)] transition-colors hover:bg-[var(--accent-hover)]"
        onMouseEnter={() => setHovering(true)}
        onMouseLeave={() => setHovering(false)}
        onClick={() => openSourcePanel(citation.chunk_id, citation)}
      >
        <span>p.{citation.page_number}</span>
      </span>
      {hovering && (
        <div className="absolute bottom-full left-1/2 z-50 mb-3 w-80 -translate-x-1/2 rounded-2xl border border-[var(--border)] bg-[var(--surface-1)] p-4 shadow-2xl">
          <div className="text-xs font-mono uppercase tracking-[0.18em] text-[var(--text-2)]">
            Citation preview
          </div>
          <div className="mt-3 text-sm font-semibold text-[var(--accent)]">
            {citation.document_name}
          </div>
          <div className="mt-2 text-xs text-[var(--text-2)]">
            Page {citation.page_number} - Paragraph {citation.paragraph_index + 1}
          </div>
          <div className="mt-3 line-clamp-5 text-sm leading-relaxed text-[var(--text-1)]">
            {citation.excerpt}
          </div>
        </div>
      )}
    </span>
  )
}
