import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { useAppStore } from './store/appStore'
import { useShallow } from 'zustand/react/shallow'
import { useWebSocket } from './hooks/useWebSocket'
import { useSession } from './hooks/useSession'
import { getChatHistory, getGraph, getGraphNodeChunks } from './api/endpoints'
import { KnowledgeGraph } from './components/KnowledgeGraph'
import { ChatPanel } from './components/ChatPanel'
import { DocumentSetSelector } from './components/DocumentSetSelector'
import { SourcePanel } from './components/SourcePanel'
import { ALL_GRAPH_ENTITY_TYPES } from './store/appStore'
import type { GraphChunkPreview, GraphNode } from './types/api'

type Theme = 'dark' | 'light'

function initTheme(): Theme {
  const stored = localStorage.getItem('qm_theme')
  if (stored === 'light' || stored === 'dark') return stored
  return 'dark'
}

function SunIcon(): ReactNode {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
      <circle cx="12" cy="12" r="4" fill="currentColor" stroke="none" />
      <line x1="12" y1="2" x2="12" y2="5.5" />
      <line x1="12" y1="18.5" x2="12" y2="22" />
      <line x1="2" y1="12" x2="5.5" y2="12" />
      <line x1="18.5" y1="12" x2="22" y2="12" />
      <line x1="4.93" y1="4.93" x2="7.34" y2="7.34" />
      <line x1="16.66" y1="16.66" x2="19.07" y2="19.07" />
      <line x1="4.93" y1="19.07" x2="7.34" y2="16.66" />
      <line x1="16.66" y1="7.34" x2="19.07" y2="4.93" />
    </svg>
  )
}

function MoonIcon(): ReactNode {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" stroke="none">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  )
}

function ThemeToggle({ theme, onToggle }: { theme: Theme; onToggle: () => void }) {
  const isLight = theme === 'light'
  return (
    <button
      onClick={onToggle}
      title={isLight ? 'Switch to dark mode' : 'Switch to light mode'}
      aria-label={isLight ? 'Switch to dark mode' : 'Switch to light mode'}
      className={`relative flex h-8 w-[3.5rem] shrink-0 items-center rounded-full border p-0.5 transition-all duration-500 focus:outline-none ${
        isLight
          ? 'border-amber-300/70 bg-gradient-to-r from-sky-100 to-amber-100/80'
          : 'border-[var(--accent-border)] bg-[var(--surface-2)]'
      }`}
    >
      {/* Icon on left side of track */}
      <span
        className={`absolute left-1.5 transition-all duration-300 ${
          isLight ? 'opacity-30 text-slate-500' : 'text-[var(--text-2)] opacity-70'
        }`}
      >
        <MoonIcon />
      </span>
      {/* Icon on right side of track */}
      <span
        className={`absolute right-1.5 transition-all duration-300 ${
          isLight ? 'text-amber-500 opacity-70' : 'opacity-30 text-slate-400'
        }`}
      >
        <SunIcon />
      </span>
      {/* Sliding thumb */}
      <span
        className={`relative z-10 flex h-6 w-6 items-center justify-center rounded-full shadow-md transition-all duration-500 ${
          isLight
            ? 'translate-x-[calc(3.5rem-3.25rem)] bg-amber-400 text-amber-900'
            : 'translate-x-0 bg-[var(--accent)] text-white'
        }`}
      >
        {isLight ? <SunIcon /> : <MoonIcon />}
      </span>
    </button>
  )
}

function ToastError() {
  const { toastError, dismissError } = useAppStore(useShallow(s => ({
    toastError: s.toastError,
    dismissError: s.dismissError,
  })))

  if (!toastError) return null

  return (
    <div className="fixed bottom-5 left-1/2 z-50 w-[min(34rem,calc(100vw-2rem))] -translate-x-1/2">
      <div className="flex items-start gap-3 rounded-2xl border border-red-500/35 bg-[#23151a]/95 px-4 py-3 text-sm text-red-200 shadow-2xl backdrop-blur">
        <span className="mt-0.5 text-base">!</span>
        <span className="flex-1 leading-relaxed">{toastError}</span>
        <button
          onClick={dismissError}
          className="rounded-full border border-white/10 px-2 py-0.5 text-[11px] font-mono uppercase tracking-[0.14em] text-red-200/70 transition-colors hover:text-red-100"
        >
          Close
        </button>
      </div>
    </div>
  )
}

function IndexingBadge() {
  const jobs = useAppStore(s => s.indexingJobs)
  const running = Object.values(jobs).filter(j => !j.done)
  if (running.length === 0) return null

  const lead = running[0]

  return (
    <div className="rounded-3xl border border-[var(--border)] bg-[var(--surface-1)] px-4 py-4 shadow-[var(--shadow-sm)]">
      <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-[0.18em] text-[var(--text-2)]">
        <span className="h-2 w-2 rounded-full bg-[var(--accent)] animate-pulse" />
        Indexing live
      </div>
      <div className="mt-3 text-sm font-medium text-[var(--text-0)]">{lead.filename}</div>
      <div className="mt-2 flex items-center justify-between text-xs text-[var(--text-2)]">
        <span>Indexing</span>
        <span className="font-semibold text-[var(--accent)]">{lead.percent.toFixed(0)}%</span>
      </div>
      <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-[var(--border)]">
        <div
          className="h-full rounded-full bg-[var(--accent)] transition-all duration-300"
          style={{ width: `${lead.percent}%` }}
        />
      </div>
      {running.length > 1 && (
        <div className="mt-2 text-xs text-[var(--text-2)]">
          +{running.length - 1} more file{running.length > 2 ? 's' : ''} in progress
        </div>
      )}
    </div>
  )
}

function GraphFiltersPanel({
  minFrequency,
  maxFrequency,
  entityTypes,
  documentNames,
  availableDocuments,
  totalNodes,
  totalEdges,
  onMinFrequencyChange,
  onEntityTypeToggle,
  onDocumentToggle,
  onResetFilters,
  onRebuild,
}: {
  minFrequency: number
  maxFrequency: number
  entityTypes: string[]
  documentNames: string[]
  availableDocuments: string[]
  totalNodes: number
  totalEdges: number
  onMinFrequencyChange: (value: number) => void
  onEntityTypeToggle: (type: string) => void
  onDocumentToggle: (name: string) => void
  onResetFilters: () => void
  onRebuild: () => void
}) {
  return (
    <div className="absolute left-3 right-3 top-3 z-10 max-h-[calc(100%-1.5rem)] overflow-y-auto rounded-2xl border border-[var(--border)] bg-[var(--surface-1)]/96 p-4 shadow-2xl backdrop-blur-xl sm:left-auto sm:right-4 sm:top-4 sm:w-80 sm:max-h-[calc(100%-2rem)]">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-xs font-mono uppercase tracking-[0.18em] text-[var(--text-2)]">
            Graph controls
          </div>
          <div className="mt-2 text-sm text-[var(--text-1)]">
            Tune the view, then rebuild the layout if it gets messy.
          </div>
        </div>
        <div className="grid shrink-0 grid-cols-2 gap-2 text-center text-xs font-mono font-semibold">
          <div className="rounded-xl border border-[var(--border)] bg-[var(--surface-2)] px-2 py-2 text-[var(--text-1)]">
            <div className="text-sm">{totalNodes}</div>
            <div className="mt-2 text-[11px] text-[var(--text-2)]">nodes</div>
          </div>
          <div className="rounded-xl border border-[var(--border)] bg-[var(--surface-2)] px-2 py-2 text-[var(--text-1)]">
            <div className="text-sm">{totalEdges}</div>
            <div className="mt-2 text-[11px] text-[var(--text-2)]">edges</div>
          </div>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-2">
        <button
          onClick={onRebuild}
          className="rounded-xl border border-[var(--accent-border)] bg-[var(--accent-subtle)] px-3 py-2 text-xs font-mono font-semibold uppercase tracking-[0.18em] text-[var(--accent)] transition-colors hover:bg-[var(--accent-hover)]"
        >
          Rebuild
        </button>
        <button
          onClick={onResetFilters}
          className="rounded-xl border border-[var(--border)] bg-[var(--surface-2)] px-3 py-2 text-xs font-mono font-semibold uppercase tracking-[0.18em] text-[var(--text-2)] transition-colors hover:border-[var(--accent-border)] hover:text-[var(--text-0)]"
        >
          Reset
        </button>
      </div>

      <div className="mt-4 rounded-2xl border border-[var(--border)] bg-[var(--surface-2)] p-3">
        <div className="mb-3 flex items-center justify-between text-xs font-semibold text-[var(--text-2)]">
          <span>Min frequency</span>
          <span className="font-mono text-[var(--text-1)]">{minFrequency}</span>
        </div>
        <input
          type="range"
          min={1}
          max={maxFrequency}
          value={minFrequency}
          onChange={e => onMinFrequencyChange(Number(e.target.value))}
          className="w-full accent-[var(--accent)]"
        />
      </div>

      <div className="mt-4">
        <div className="mb-3 text-xs font-mono font-semibold uppercase tracking-[0.18em] text-[var(--text-2)]">
          Entity types
        </div>
        <div className="grid grid-cols-2 gap-2">
          {ALL_GRAPH_ENTITY_TYPES.map(type => {
            const checked = entityTypes.includes(type)
            return (
              <label
                key={type}
                className={`flex items-center gap-2 rounded-xl border px-3 py-2 text-xs transition-colors ${
                  checked
                    ? 'border-[var(--accent-border)] bg-[var(--accent-subtle)] text-[var(--text-1)]'
                    : 'border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-2)] hover:text-[var(--text-1)]'
                }`}
              >
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => onEntityTypeToggle(type)}
                  className="accent-[var(--accent)]"
                />
                <span className="capitalize">{type}</span>
              </label>
            )
          })}
        </div>
      </div>

      {availableDocuments.length > 0 && (
        <div className="mt-4">
          <div className="mb-3 flex items-center justify-between text-xs font-mono font-semibold uppercase tracking-[0.18em] text-[var(--text-2)]">
            <span>Document focus</span>
            {documentNames.length > 0 && (
              <span className="rounded-md border border-[var(--accent-border)] bg-[var(--accent-subtle)] px-1.5 py-0.5 text-[10px] text-[var(--accent)]">
                {documentNames.length}/{availableDocuments.length}
              </span>
            )}
          </div>
          <div className="flex flex-col gap-1.5">
            {availableDocuments.map(name => {
              const checked = documentNames.includes(name)
              return (
                <label
                  key={name}
                  className={`flex items-center gap-2 rounded-xl border px-3 py-2 text-xs transition-colors cursor-pointer ${
                    checked
                      ? 'border-[var(--accent-border)] bg-[var(--accent-subtle)] text-[var(--text-1)]'
                      : 'border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-2)] hover:text-[var(--text-1)]'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => onDocumentToggle(name)}
                    className="accent-[var(--accent)]"
                  />
                  <span className="truncate" title={name}>{name}</span>
                </label>
              )
            })}
          </div>
          {documentNames.length === 0 && (
            <div className="mt-1.5 text-[10px] text-[var(--text-2)]">
              No selection = all documents shown
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function GraphNodePanel({
  documentSetId,
  node,
  documentName,
  onClose,
}: {
  documentSetId: string
  node: GraphNode
  documentName?: string | null
  onClose: () => void
}) {
  const openSourcePanel = useAppStore(s => s.openSourcePanel)
  const [loading, setLoading] = useState(false)
  const [chunks, setChunks] = useState<GraphChunkPreview[]>([])

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    getGraphNodeChunks(documentSetId, node.id, documentName)
      .then((data) => {
        if (!cancelled) setChunks(data)
      })
      .catch(() => {
        if (!cancelled) setChunks([])
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [documentSetId, node.id, documentName])

  return (
    <div className="fixed inset-x-3 bottom-3 z-30 flex max-h-[72vh] flex-col rounded-[1.75rem] border border-[var(--border)] bg-[var(--surface-1)] shadow-2xl lg:inset-x-auto lg:bottom-auto lg:left-80 lg:top-0 lg:h-full lg:max-h-none lg:w-[28rem] lg:rounded-none lg:border-r">
      <div className="border-b border-[var(--border)] px-5 py-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="text-base font-semibold text-[var(--text-0)]">{node.label}</div>
            <div className="mt-2 text-sm capitalize text-[var(--text-2)]">
              {node.type} - {node.frequency} chunk{node.frequency > 1 ? 's' : ''}
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-full border border-[var(--border)] px-2 py-1 text-xs font-mono font-semibold uppercase tracking-[0.18em] text-[var(--text-2)] transition-colors hover:border-[var(--accent-border)] hover:text-[var(--text-1)]"
          >
            Close
          </button>
        </div>
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto p-4">
        {loading && (
          <div className="text-xs font-mono text-[var(--text-2)]">Loading chunks...</div>
        )}

        {!loading && chunks.length === 0 && (
          <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface-2)] p-4 text-sm text-center text-[var(--text-2)]">
            No chunks found for this node.
          </div>
        )}

        {chunks.map(chunk => (
          <button
            key={chunk.chunk_id}
            onClick={() => openSourcePanel(chunk.chunk_id)}
            className="w-full rounded-2xl border border-[var(--border)] bg-[var(--surface-2)] p-4 text-left transition-colors hover:border-[var(--accent-border)] hover:bg-[var(--surface-3)]"
          >
            <div className="text-sm font-semibold text-[var(--accent)]">{chunk.document_name}</div>
            <div className="mt-2 text-xs font-mono text-[var(--text-2)]">
              Page {chunk.page_number} - Paragraph {chunk.paragraph_index + 1}
            </div>
            <div className="mt-3 whitespace-pre-wrap text-sm leading-relaxed text-[var(--text-1)]">
              {chunk.excerpt}
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}

function ViewToggle({
  active,
  onChange,
  graphLoading,
  disabled,
}: {
  active: 'chat' | 'graph'
  onChange: (view: 'chat' | 'graph') => void
  graphLoading: boolean
  disabled: boolean
}) {
  const base = 'inline-flex items-center gap-2 rounded-xl px-3 py-2 text-xs font-mono font-semibold uppercase tracking-[0.18em] border transition-colors'

  return (
    <div className="inline-flex items-center gap-2 rounded-2xl border border-[var(--border)] bg-[var(--surface-1)]/80 p-1 shadow-[var(--shadow-sm)] backdrop-blur">
      <button
        onClick={() => onChange('chat')}
        disabled={disabled}
        className={`${base} ${
          active === 'chat'
            ? 'border-[var(--accent-border)] bg-[var(--accent-subtle)] text-[var(--accent)]'
            : 'border-transparent bg-transparent text-[var(--text-2)] hover:text-[var(--text-0)]'
        } disabled:opacity-50`}
      >
        Chatbot
      </button>

      <button
        onClick={() => onChange('graph')}
        disabled={disabled}
        className={`${base} ${
          active === 'graph'
            ? 'border-[var(--accent-border)] bg-[var(--accent-subtle)] text-[var(--accent)]'
            : 'border-transparent bg-transparent text-[var(--text-2)] hover:text-[var(--text-0)]'
        } disabled:opacity-50`}
      >
        {graphLoading && (
          <span className="h-3.5 w-3.5 rounded-full border border-[var(--accent-border)] border-r-[var(--accent)] border-t-[var(--accent)] animate-spin" />
        )}
        Graph
      </button>
    </div>
  )
}

export default function App() {
  const {
    currentSetId,
    graphNodes,
    graphEdges,
    graphFilters,
    graphMeta,
    activeNodeIds,
    activeEdgeIds,
    setGraphFilters,
    resetGraphFilters,
    setGraphResponse,
    setMessages,
    showError,
  } = useAppStore(useShallow(s => ({
    currentSetId: s.currentSetId,
    graphNodes: s.graphNodes,
    graphEdges: s.graphEdges,
    graphFilters: s.graphFilters,
    graphMeta: s.graphMeta,
    activeNodeIds: s.activeNodeIds,
    activeEdgeIds: s.activeEdgeIds,
    setGraphFilters: s.setGraphFilters,
    resetGraphFilters: s.resetGraphFilters,
    setGraphResponse: s.setGraphResponse,
    setMessages: s.setMessages,
    showError: s.showError,
  })))

  const [theme, setTheme] = useState<Theme>(initTheme)

  useEffect(() => {
    if (theme === 'light') document.documentElement.setAttribute('data-theme', 'light')
    else document.documentElement.removeAttribute('data-theme')
    localStorage.setItem('qm_theme', theme)
  }, [theme])

  const toggleTheme = useCallback(() => setTheme(t => t === 'dark' ? 'light' : 'dark'), [])

  const [activeView, setActiveView] = useState<'chat' | 'graph'>('chat')
  const [graphLoading, setGraphLoading] = useState(false)
  const [graphRenderVersion, setGraphRenderVersion] = useState(0)
  const [selectedGraphNode, setSelectedGraphNode] = useState<GraphNode | null>(null)
  const latestGraphRequestRef = useRef(0)
  const maxGraphFrequency = Math.max(10, graphFilters.min_frequency, ...graphNodes.map(node => node.frequency))
  const { sessionId } = useSession(currentSetId)

  useWebSocket(sessionId)

  const loadGraph = useCallback(async (documentSetId: string, loading = true) => {
    const requestId = latestGraphRequestRef.current + 1
    latestGraphRequestRef.current = requestId

    if (loading) setGraphLoading(true)
    try {
      const data = await getGraph(documentSetId, graphFilters)
      if (latestGraphRequestRef.current !== requestId) return
      setGraphResponse(data)
    } catch (e: any) {
      if (latestGraphRequestRef.current !== requestId) return
      showError({
        code: 'GRAPH_LOAD_FAILED',
        message: e?.response?.data?.detail ?? 'Failed to load graph',
        recoverable: true,
      })
    } finally {
      if (latestGraphRequestRef.current === requestId) setGraphLoading(false)
    }
  }, [graphFilters, setGraphResponse, showError])

  useEffect(() => {
    if (!currentSetId) return
    setActiveView('chat')
    setSelectedGraphNode(null)
    setGraphRenderVersion(0)
  }, [currentSetId])

  useEffect(() => {
    if (!currentSetId || activeView !== 'graph') return
    void loadGraph(currentSetId)
  }, [activeView, currentSetId, graphFilters, loadGraph])

  useEffect(() => {
    if (!selectedGraphNode) return
    if (graphNodes.some(node => node.id === selectedGraphNode.id)) return
    setSelectedGraphNode(null)
  }, [graphNodes, selectedGraphNode])

  useEffect(() => {
    if (!currentSetId) return
    getChatHistory(sessionId)
      .then(turns => setMessages(turns))
      .catch(() => setMessages([]))
  }, [currentSetId, sessionId, setMessages])

  const handleViewChange = (view: 'chat' | 'graph') => {
    setActiveView(view)
    if (view === 'chat') setSelectedGraphNode(null)
  }

  const toggleEntityType = (type: string) => {
    const current = graphFilters.entity_types
    const next = current.includes(type)
      ? current.filter(t => t !== type)
      : [...current, type]
    if (next.length === 0) return
    setSelectedGraphNode(null)
    setGraphFilters({ entity_types: next })
  }

  const rebuildGraphView = useCallback(() => {
    if (!currentSetId) return
    setSelectedGraphNode(null)
    setGraphRenderVersion(version => version + 1)
    void loadGraph(currentSetId)
  }, [currentSetId, loadGraph])

  const handleResetGraphFilters = () => {
    setSelectedGraphNode(null)
    resetGraphFilters()
  }

  return (
    <div className="flex h-full flex-col bg-[var(--bg)] text-[var(--text-0)] lg:flex-row">
      <aside className="shrink-0 border-b border-[var(--border)] bg-[var(--bg-1)]/92 lg:flex lg:h-full lg:w-80 lg:flex-col lg:overflow-hidden lg:border-b-0 lg:border-r">
        {/* Logo — never scrolls on desktop */}
        <div className="shrink-0 px-4 pt-4">
          <div className="rounded-3xl border border-[var(--border)] bg-[var(--surface-1)] px-5 py-5 shadow-[var(--shadow-md)]">
            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-2xl border border-[var(--accent-border)] bg-[var(--accent-subtle)] text-lg font-bold text-[var(--accent)]">
                ◈
              </div>
              <div>
                <div className="text-lg font-bold text-[var(--text-0)]">QueryMind</div>
                <div className="mt-1 text-xs leading-snug text-[var(--text-2)]">Multilingual RAG + Knowledge Graph</div>
              </div>
            </div>
          </div>
        </div>

        {/* Scrollable content — hidden scrollbar */}
        <div className="flex flex-1 flex-col gap-4 overflow-y-auto px-4 pb-4 pt-4 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          <DocumentSetSelector />
          <div className="mt-auto">
            <IndexingBadge />
          </div>
        </div>
      </aside>

      <main className="min-h-0 flex-1 bg-[var(--bg)]">
        <div className="flex h-full min-h-0 flex-col">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--border)] bg-[var(--bg-1)]/72 px-4 py-3 backdrop-blur lg:min-h-[4.5rem] lg:px-5">
            <div className="flex min-w-0 flex-wrap items-center gap-3">
              <ViewToggle
                active={activeView}
                onChange={handleViewChange}
                graphLoading={graphLoading}
                disabled={!currentSetId}
              />
              {currentSetId && (
                <div className="hidden rounded-2xl border border-[var(--border)] bg-[var(--surface-1)] px-3 py-2 text-sm text-[var(--text-2)] xl:block">
                  Ask in English, French, or Arabic. Citations stay anchored to the source.
                </div>
              )}
            </div>

            <div className="flex items-center gap-3">
              {!currentSetId && (
                <span className="rounded-xl border border-[var(--border)] bg-[var(--surface-1)] px-3 py-2 text-xs font-mono text-[var(--text-2)]">
                  No document set selected
                </span>
              )}
              <ThemeToggle theme={theme} onToggle={toggleTheme} />
            </div>
          </div>

          <div className="min-h-0 flex-1">
            {activeView === 'graph' ? (
              <div className="h-full p-3 lg:p-5">
                <div className="relative h-full overflow-hidden rounded-[2rem] border border-[var(--border)] bg-[var(--surface-1)] shadow-[var(--shadow-md)]">
                  {currentSetId && (
                    <GraphFiltersPanel
                      minFrequency={graphFilters.min_frequency}
                      maxFrequency={maxGraphFrequency}
                      entityTypes={graphFilters.entity_types}
                      documentNames={graphFilters.document_names}
                      availableDocuments={graphMeta.availableDocuments}
                      totalNodes={graphNodes.length}
                      totalEdges={graphEdges.length}
                      onMinFrequencyChange={(value) => {
                        setSelectedGraphNode(null)
                        setGraphFilters({ min_frequency: value })
                      }}
                      onEntityTypeToggle={toggleEntityType}
                      onDocumentToggle={(name) => {
                        setSelectedGraphNode(null)
                        const current = graphFilters.document_names
                        setGraphFilters({
                          document_names: current.includes(name)
                            ? current.filter(n => n !== name)
                            : [...current, name],
                        })
                      }}
                      onResetFilters={handleResetGraphFilters}
                      onRebuild={rebuildGraphView}
                    />
                  )}

                  {graphLoading && (
                    <div className="absolute left-3 top-3 z-10 inline-flex items-center gap-2 rounded-xl border border-[var(--border)] bg-[var(--surface-1)] px-3 py-2 text-xs font-semibold text-[var(--text-2)] shadow-[var(--shadow-sm)] sm:left-4 sm:top-4">
                      <span className="h-3 w-3 rounded-full border-2 border-[var(--accent-border)] border-r-[var(--accent)] border-t-[var(--accent)] animate-spin" />
                      Loading...
                    </div>
                  )}

                  {(graphMeta.message || graphNodes.length > 0 || graphEdges.length > 0) && (
                    <div className="absolute bottom-3 left-3 right-3 z-10 rounded-2xl border border-[var(--border)] bg-[var(--surface-1)]/96 px-4 py-3 text-xs font-mono font-semibold text-[var(--text-2)] shadow-[var(--shadow-sm)] backdrop-blur sm:left-4 sm:right-auto sm:max-w-xl">
                      <div>
                        {graphNodes.length} nodes / {graphEdges.length} edges
                        {graphMeta.totalNodeCount > 0 && (
                          <span className="text-[var(--text-2)]">, filtered from {graphMeta.totalNodeCount}</span>
                        )}
                      </div>
                      {graphMeta.message && (
                        <div className="mt-2 font-semibold text-[var(--accent)]">{graphMeta.message}</div>
                      )}
                    </div>
                  )}

                  <KnowledgeGraph
                    key={[currentSetId ?? 'none', graphRenderVersion].join(':')}
                    nodes={graphNodes}
                    edges={graphEdges}
                    activeNodeIds={activeNodeIds}
                    activeEdgeIds={activeEdgeIds}
                    onNodeClick={(node) => setSelectedGraphNode(node)}
                  />
                </div>
              </div>
            ) : (
              <div className="h-full p-3 lg:p-5">
                <div className="h-full overflow-hidden rounded-[2rem] border border-[var(--border)] bg-[var(--surface-1)] shadow-[var(--shadow-md)]">
                  <ChatPanel />
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      {currentSetId && selectedGraphNode && activeView === 'graph' && (
        <GraphNodePanel
          documentSetId={currentSetId}
          node={selectedGraphNode}
          documentName={graphFilters.document_names.length === 1 ? graphFilters.document_names[0] : undefined}
          onClose={() => setSelectedGraphNode(null)}
        />
      )}

      <SourcePanel />
      <ToastError />
    </div>
  )
}
