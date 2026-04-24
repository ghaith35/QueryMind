// Mirrors backend Pydantic schemas

export interface Citation {
  chunk_id: string
  document_name: string
  page_number: number
  paragraph_index: number
  excerpt: string
  relevance_score: number
}

export interface GraphNode {
  id: string
  label: string
  type: string
  frequency: number
  document_sources: string[]
  chunk_ids: string[]
}

export interface GraphEdge {
  source: string
  target: string
  weight: number
  co_occurrence_count: number
  relation_type?: string
}

export interface GraphFilters {
  min_frequency: number
  entity_types: string[]
  document_names: string[]
}

export interface GraphResponse {
  nodes: GraphNode[]
  edges: GraphEdge[]
  node_count: number
  edge_count: number
  total_node_count: number
  total_edge_count: number
  truncated: boolean
  message?: string | null
  available_documents: string[]
}

export interface GraphChunkPreview {
  chunk_id: string
  document_name: string
  page_number: number
  paragraph_index: number
  excerpt: string
}

export interface ChunkDetail {
  chunk_id: string
  document_set_id: string
  document_name: string
  page_number: number
  paragraph_index: number
  text: string
  language: string
  entities: string[]
}

export interface DocumentSet {
  id: string
  name: string
  created_at: string
  doc_count: number
  chunk_count: number
}

export interface DocumentInSet {
  doc_id: string
  document_name: string
  chunk_count: number
}


export interface UploadResponse {
  job_id: string
  filename: string
  status: 'queued' | 'indexing' | 'complete' | 'failed'
  message?: string | null
}

export interface Turn {
  turn_id: string
  session_id: string
  role: 'user' | 'assistant'
  content: string
  citations: Citation[]
  retrieved_chunk_ids: string[]
  timestamp: string
}

// WebSocket message payloads

export interface IndexProgressPayload {
  job_id: string
  document_name: string
  chunks_processed: number
  total_chunks_estimated: number
  stage: 'extracting' | 'chunking' | 'embedding' | 'storing' | 'complete'
  percent: number
}

export interface IndexCompletePayload {
  job_id: string
  document_name: string
  chunk_count: number
  elapsed_seconds: number
}

export interface GraphUpdatePayload {
  document_set_id: string
  added_nodes: GraphNode[]
  added_edges: GraphEdge[]
  updated_nodes: Array<{ id: string; frequency: number }>
}

export interface AnswerStreamPayload {
  session_id: string
  turn_id: string
  token: string
  is_final: boolean
}

export interface AnswerCompletePayload {
  session_id: string
  turn_id: string
  full_answer: string
  citations: Citation[]
  active_node_ids: string[]
  active_edge_ids: string[]
  elapsed_seconds?: number | null
}

export interface ErrorPayload {
  code: string
  message: string
  recoverable: boolean
}

// App-level message type (combines streaming + complete)

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations: Citation[]
  streaming: boolean
  elapsedSeconds?: number | null
  error?: string
}

export interface IndexingJob {
  filename: string
  stage: string
  percent: number
  done: boolean
}
