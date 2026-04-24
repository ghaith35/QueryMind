import { CitationBadge } from './CitationBadge'
import type { Message } from '../types/api'

function renderInlineMarkdown(text: string, baseKey: number): React.ReactNode {
  const parts = text.split(/(\*\*[^*\n]+\*\*|\*[^*\n]+\*)/g)
  if (parts.length === 1) return text
  return parts.map((p, i) => {
    if (p.startsWith('**') && p.endsWith('**'))
      return <strong key={`${baseKey}-${i}`}>{p.slice(2, -2)}</strong>
    if (p.startsWith('*') && p.endsWith('*'))
      return <em key={`${baseKey}-${i}`}>{p.slice(1, -1)}</em>
    return p
  })
}

function renderContent(content: string, message: Message) {
  if (message.role === 'user') return <span>{content}</span>

  const parts = content.split(/(\[CITE:\s*[a-f0-9_]+\])/gi)
  const cMap = Object.fromEntries(message.citations.map(c => [c.chunk_id, c]))

  return (
    <>
      {parts.map((part, i) => {
        const match = part.match(/\[CITE:\s*([a-f0-9_]+)\]/i)
        if (match) {
          const citation = cMap[match[1]]
          if (!citation) return null
          return <CitationBadge key={i} citation={citation} />
        }
        return <span key={i}>{renderInlineMarkdown(part, i)}</span>
      })}
    </>
  )
}

export function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user'
  const sourceCount = message.citations.length
  const footer = !isUser && !message.streaming && sourceCount > 0
    ? `Answered from ${sourceCount} source${sourceCount > 1 ? 's' : ''}${message.elapsedSeconds ? ` - ${message.elapsedSeconds.toFixed(1)}s` : ''}`
    : null

  return (
    <div className={`mb-6 flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`${isUser ? 'max-w-[72%]' : 'max-w-[85%]'} space-y-2`}>
        <div className={`text-xs font-mono uppercase tracking-[0.18em] ${isUser ? 'text-right text-[var(--text-2)]' : 'text-[var(--text-2)]'}`}>
          {isUser ? 'You' : 'QueryMind'}
        </div>
        <div
          className={`rounded-[1.5rem] px-4 py-3 text-base leading-relaxed shadow-[var(--shadow-sm)] ${
            isUser
              ? 'rounded-tr-md border border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-0)]'
              : 'rounded-tl-md border border-[var(--border-subtle)] border-l-2 border-l-[var(--accent)] bg-[var(--surface-message)] text-[var(--text-0)]'
          }`}
        >
          {message.error ? (
            <span className="text-[var(--error)]">{message.error}</span>
          ) : (
            <div dir="auto" className="whitespace-pre-wrap">
              {renderContent(message.content, message)}
              {message.streaming && (
                <span className="ml-1 inline-block h-4 w-1.5 animate-pulse rounded-sm bg-[var(--accent)] align-middle" />
              )}
            </div>
          )}
        </div>
        {footer && (
          <div className="text-xs font-mono text-[var(--text-2)] mt-2">
            {footer}
          </div>
        )}
      </div>
    </div>
  )
}
