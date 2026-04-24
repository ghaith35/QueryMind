export function StyleGuide() {
  const swatches = [
    ['Background 0', 'var(--bg-0)'],
    ['Background 1', 'var(--bg-1)'],
    ['Surface 1', 'var(--surface-1)'],
    ['Surface 2', 'var(--surface-2)'],
    ['Surface 3', 'var(--surface-3)'],
    ['Accent', 'var(--accent)'],
    ['Success', 'var(--success)'],
    ['Warning', 'var(--warning)'],
    ['Error', 'var(--error)'],
  ]

  return (
    <div className="min-h-screen bg-[var(--bg-0)] text-[var(--text-0)] px-6 py-10">
      <div className="max-w-6xl mx-auto space-y-10">
        <div className="space-y-3">
          <div className="text-xs uppercase tracking-[0.3em] text-[var(--text-2)] font-mono">
            QueryMind
          </div>
          <h1 className="text-5xl font-semibold">Style Guide</h1>
          <p className="max-w-2xl text-[var(--text-1)] leading-relaxed">
            Visual tokens, typography, and component samples for the current QueryMind interface.
          </p>
        </div>

        <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {swatches.map(([label, color]) => (
            <div
              key={label}
              className="rounded-2xl border border-[var(--border)] p-4 bg-[var(--surface-1)]"
            >
              <div
                className="h-24 rounded-xl border border-white/10 mb-3"
                style={{ background: color }}
              />
              <div className="font-medium">{label}</div>
              <div className="text-sm text-[var(--text-2)] font-mono">{color}</div>
            </div>
          ))}
        </section>

        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="rounded-3xl border border-[var(--border)] bg-[var(--surface-1)] p-6 space-y-4 shadow-[var(--shadow-md)]">
            <div className="text-sm text-[var(--text-2)] font-mono">Typography</div>
            <div className="text-4xl font-semibold">Explain the document in Arabic</div>
            <div className="text-base leading-relaxed text-[var(--text-1)]">
              QueryMind keeps the visual language calm and technical while still feeling alive.
            </div>
            <div className="font-mono text-sm text-[var(--text-2)]">
              Citation / graph label / source metadata
            </div>
            <div dir="auto" className="text-xl text-[var(--text-0)]">
              هذا مثال على دعم العربية داخل نفس النظام البصري.
            </div>
          </div>

          <div className="rounded-3xl border border-[var(--border)] bg-[var(--surface-1)] p-6 space-y-4 shadow-[var(--shadow-md)]">
            <div className="text-sm text-[var(--text-2)] font-mono">Chat Sample</div>
            <div className="flex justify-end">
              <div className="max-w-[70%] rounded-2xl rounded-tr-sm border border-[var(--accent-border)] bg-[var(--accent-subtle)] px-4 py-3">
                What are the side effects of ACE inhibitors?
              </div>
            </div>
            <div className="space-y-2">
              <div className="max-w-[85%] rounded-2xl rounded-tl-sm border-l-2 border-[var(--accent)] bg-[var(--surface-1)] px-4 py-3 shadow-[var(--shadow-sm)]">
                ACE inhibitors commonly cause dry cough and may also increase potassium levels.
              </div>
              <div className="text-xs text-[var(--text-2)] font-mono">
                Answered from 2 sources
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}
