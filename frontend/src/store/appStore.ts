import { create } from 'zustand'
import type {
  GraphFilters,
  Message,
  GraphNode,
  GraphEdge,
  GraphResponse,
  GraphUpdatePayload,
  IndexProgressPayload,
  IndexCompletePayload,
  AnswerCompletePayload,
  ErrorPayload,
  IndexingJob,
  Citation,
  Turn,
} from '../types/api'

export const GLOBAL_SESSION_KEY = 'qm_session'
export const SESSION_SET_KEY = 'qm_session_set'
export const CURRENT_SET_KEY = 'qm_current_set'
export const ALL_GRAPH_ENTITY_TYPES = ['concept', 'person', 'organization', 'location', 'technique']

function defaultGraphFilters(): GraphFilters {
  return {
    min_frequency: 2,
    entity_types: [...ALL_GRAPH_ENTITY_TYPES],
    document_names: [],
  }
}

function readStoredSessionId(): string {
  const existing = localStorage.getItem(GLOBAL_SESSION_KEY)
  if (existing) return existing
  const id = crypto.randomUUID()
  localStorage.setItem(GLOBAL_SESSION_KEY, id)
  return id
}

function edgeKey(edge: Pick<GraphEdge, 'source' | 'target'>): string {
  return `${edge.source}->${edge.target}`
}

function mergeNodes(
  existing: GraphNode[],
  added: GraphNode[],
  updated: Array<{ id: string; frequency: number }>,
): GraphNode[] {
  const map = new Map(existing.map(n => [n.id, n]))
  for (const n of added) if (!map.has(n.id)) map.set(n.id, n)
  for (const u of updated) {
    const n = map.get(u.id)
    if (n) map.set(u.id, { ...n, frequency: u.frequency })
  }
  return Array.from(map.values())
}

function mergeEdges(existing: GraphEdge[], added: GraphEdge[]): GraphEdge[] {
  const ids = new Set(existing.map(edgeKey))
  return [...existing, ...added.filter(e => !ids.has(edgeKey(e)))]
}

interface AppStore {
  sessionId: string
  currentSetId: string | null
  messages: Message[]
  graphNodes: GraphNode[]
  graphEdges: GraphEdge[]
  graphLoaded: boolean
  graphFilters: GraphFilters
  graphMeta: {
    totalNodeCount: number
    totalEdgeCount: number
    truncated: boolean
    message: string | null
    availableDocuments: string[]
  }
  activeNodeIds: Set<string>
  activeEdgeIds: Set<string>
  indexingJobs: Record<string, IndexingJob>
  sourcePanelChunkId: string | null
  sourcePanelCitation: Citation | null
  toastError: string | null

  setSessionId: (id: string) => void
  setCurrentSet: (id: string | null) => void
  addUserMessage: (content: string) => string
  addAssistantPlaceholder: (turnId: string) => void
  appendStreamingToken: (turnId: string, token: string) => void
  finalizeAnswer: (payload: AnswerCompletePayload) => void
  setActivePath: (nodes: string[], edges: string[]) => void
  clearActivePath: () => void
  setGraphFilters: (patch: Partial<GraphFilters>) => void
  resetGraphFilters: () => void
  mergeGraphUpdate: (payload: GraphUpdatePayload) => void
  setGraph: (nodes: GraphNode[], edges: GraphEdge[]) => void
  setGraphResponse: (payload: GraphResponse) => void
  updateIndexProgress: (payload: IndexProgressPayload) => void
  markIndexComplete: (payload: IndexCompletePayload) => void
  openSourcePanel: (chunkId: string, citation?: Citation | null) => void
  closeSourcePanel: () => void
  showError: (payload: ErrorPayload) => void
  dismissError: () => void
  clearMessages: () => void
  setMessages: (turns: Turn[]) => void
}

export const useAppStore = create<AppStore>((set, _get) => ({
  sessionId: readStoredSessionId(),
  currentSetId: null,
  messages: [],
  graphNodes: [],
  graphEdges: [],
  graphLoaded: false,
  graphFilters: defaultGraphFilters(),
  graphMeta: {
    totalNodeCount: 0,
    totalEdgeCount: 0,
    truncated: false,
    message: null,
    availableDocuments: [],
  },
  activeNodeIds: new Set(),
  activeEdgeIds: new Set(),
  indexingJobs: {},
  sourcePanelChunkId: null,
  sourcePanelCitation: null,
  toastError: null,

  setSessionId: (id) => {
    localStorage.setItem(GLOBAL_SESSION_KEY, id)
    set({
      sessionId: id,
      messages: [],
      activeNodeIds: new Set(),
      activeEdgeIds: new Set(),
      sourcePanelChunkId: null,
      sourcePanelCitation: null,
    })
  },

  setCurrentSet: (id) => {
    if (_get().currentSetId === id) return
    if (id) localStorage.setItem(CURRENT_SET_KEY, id)
    else localStorage.removeItem(CURRENT_SET_KEY)
    set({
      currentSetId: id,
      messages: [],
      graphNodes: [],
      graphEdges: [],
      graphLoaded: false,
      graphFilters: defaultGraphFilters(),
      graphMeta: {
        totalNodeCount: 0,
        totalEdgeCount: 0,
        truncated: false,
        message: null,
        availableDocuments: [],
      },
      activeNodeIds: new Set(),
      activeEdgeIds: new Set(),
      sourcePanelChunkId: null,
      sourcePanelCitation: null,
    })
  },

  addUserMessage: (content) => {
    const id = crypto.randomUUID()
    set(s => ({ messages: [...s.messages, { id, role: 'user', content, citations: [], streaming: false, elapsedSeconds: null }] }))
    return id
  },

  addAssistantPlaceholder: (turnId) => {
    set(s => ({
      messages: [...s.messages, { id: turnId, role: 'assistant', content: '', citations: [], streaming: true, elapsedSeconds: null }]
    }))
  },

  appendStreamingToken: (turnId, token) => {
    set(s => ({
      messages: s.messages.some(m => m.id === turnId)
        ? s.messages.map(m =>
            m.id === turnId ? { ...m, content: m.content + token } : m
          )
        : [...s.messages, { id: turnId, role: 'assistant', content: token, citations: [], streaming: true, elapsedSeconds: null }]
    }))
  },

  finalizeAnswer: (payload) => {
    set(s => ({
      messages: s.messages.some(m => m.id === payload.turn_id)
        ? s.messages.map(m =>
            m.id === payload.turn_id
              ? {
                  ...m,
                  content: payload.full_answer,
                  citations: payload.citations,
                  streaming: false,
                  elapsedSeconds: payload.elapsed_seconds ?? null,
                }
              : m
          )
        : [...s.messages, {
            id: payload.turn_id,
            role: 'assistant',
            content: payload.full_answer,
            citations: payload.citations,
            streaming: false,
            elapsedSeconds: payload.elapsed_seconds ?? null,
          }]
    }))
  },

  setActivePath: (nodes, edges) => {
    set({ activeNodeIds: new Set(nodes), activeEdgeIds: new Set(edges) })
  },

  clearActivePath: () => {
    set({ activeNodeIds: new Set(), activeEdgeIds: new Set() })
  },

  setGraphFilters: (patch) => {
    set(s => ({
      graphFilters: {
        ...s.graphFilters,
        ...patch,
      },
    }))
  },

  resetGraphFilters: () => {
    set({ graphFilters: defaultGraphFilters() })
  },

  mergeGraphUpdate: (payload) => {
    set(s => ({
      graphNodes: mergeNodes(s.graphNodes, payload.added_nodes, payload.updated_nodes),
      graphEdges: mergeEdges(s.graphEdges, payload.added_edges),
    }))
  },

  setGraph: (nodes, edges) => {
    set({ graphNodes: nodes, graphEdges: edges, graphLoaded: true })
  },

  setGraphResponse: (payload) => {
    set({
      graphNodes: payload.nodes,
      graphEdges: payload.edges,
      graphLoaded: true,
      graphMeta: {
        totalNodeCount: payload.total_node_count,
        totalEdgeCount: payload.total_edge_count,
        truncated: payload.truncated,
        message: payload.message ?? null,
        availableDocuments: payload.available_documents,
      },
    })
  },

  updateIndexProgress: (payload) => {
    set(s => ({
      indexingJobs: {
        ...s.indexingJobs,
        [payload.job_id]: {
          filename: payload.document_name,
          stage: payload.stage,
          percent: payload.percent,
          done: false,
        },
      },
    }))
  },

  markIndexComplete: (payload) => {
    set(s => ({
      indexingJobs: {
        ...s.indexingJobs,
        [payload.job_id]: {
          filename: payload.document_name,
          stage: 'complete',
          percent: 100,
          done: true,
        },
      },
    }))
  },

  openSourcePanel: (chunkId, citation) => {
    set({ sourcePanelChunkId: chunkId, sourcePanelCitation: citation ?? null })
  },

  closeSourcePanel: () => {
    set({ sourcePanelChunkId: null, sourcePanelCitation: null })
  },

  showError: (payload) => {
    const msg = payload.message ?? ''
    const isQuota =
      payload.code === 'RATE_LIMIT' ||
      msg.toLowerCase().includes('resource_exhausted') ||
      msg.toLowerCase().includes('quota') ||
      msg.toLowerCase().includes('rate limit') ||
      msg.includes('429')
    const friendly = isQuota
      ? 'This version is a demo version, not a professional version.'
      : payload.code === 'INVALID_PDF'
        ? payload.message
        : payload.code === 'DUPLICATE_DOCUMENT'
          ? 'That PDF is already indexed in this document set.'
          : payload.code === 'CITATION_MISSING'
            ? 'The answer was generated, but citation verification was weak. Please double-check the sources.'
            : payload.message
    set({ toastError: friendly })
  },

  dismissError: () => {
    set({ toastError: null })
  },

  clearMessages: () => {
    set({ messages: [] })
  },

  setMessages: (turns) => {
    set({
      messages: turns.map(t => ({
        id: t.turn_id,
        role: t.role,
        content: t.content,
        citations: t.citations,
        streaming: false,
        elapsedSeconds: null,
      })),
    })
  },

}))
