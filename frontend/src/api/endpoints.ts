import { api } from './client'
import type {
  ChunkDetail,
  DocumentInSet,
  DocumentSet,
  GraphChunkPreview,
  GraphFilters,
  GraphResponse,
  Turn,
  UploadResponse,
} from '../types/api'

export const createDocumentSet = (name: string) =>
  api.post<DocumentSet>('/documents/set', { name }).then(r => r.data)

export const listDocumentSets = () =>
  api.get<DocumentSet[]>('/documents/sets').then(r => r.data)

export const uploadDocument = (
  file: File,
  documentSetId: string,
  sessionId: string,
  onProgress?: (pct: number) => void,
) => {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('document_set_id', documentSetId)
  fd.append('session_id', sessionId)
  return api.post<UploadResponse>('/documents/upload', fd, {
    onUploadProgress: (e) => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded * 100) / e.total))
    },
  }).then(r => r.data)
}

export const deleteDocument = (docId: string) =>
  api.delete(`/documents/${docId}`).then(r => r.data)

export const listDocumentsInSet = (setId: string) =>
  api.get<DocumentInSet[]>(`/documents/set/${setId}/documents`).then(r => r.data)

export const renameDocumentSet = (setId: string, name: string) =>
  api.patch(`/documents/set/${setId}`, { name }).then(r => r.data)

export const deleteDocumentSet = (setId: string) =>
  api.delete(`/documents/set/${setId}`).then(r => r.data)

export const sendChat = (sessionId: string, documentSetId: string, message: string) =>
  api.post('/chat', { session_id: sessionId, document_set_id: documentSetId, message }).then(r => r.data)

export const getChatHistory = (sessionId: string) =>
  api.get<Turn[]>(`/chat/history/${sessionId}`).then(r => r.data)

export const deleteChatSession = (sessionId: string) =>
  api.delete(`/chat/session/${sessionId}`).then(r => r.data)

export const getGraph = (documentSetId: string, filters?: GraphFilters) =>
  api.get<GraphResponse>(`/graph/${documentSetId}`, {
    params: {
      min_frequency: filters?.min_frequency,
      entity_types: filters?.entity_types?.length ? filters.entity_types : undefined,
      document_names: filters?.document_names?.length ? filters.document_names : undefined,
    },
    paramsSerializer: {
      indexes: null,
    },
  }).then(r => r.data)

export const getGraphNodeChunks = (
  documentSetId: string,
  nodeId: string,
  documentName?: string | null,
) =>
  api.get<GraphChunkPreview[]>(`/graph/${documentSetId}/node/${nodeId}/chunks`, {
    params: {
      document_name: documentName || undefined,
    },
  }).then(r => r.data)

export const getChunkDetail = (chunkId: string) =>
  api.get<ChunkDetail>(`/graph/chunk/${chunkId}`).then(r => r.data)

export const checkHealth = () =>
  api.get('/health').then(r => r.data)
