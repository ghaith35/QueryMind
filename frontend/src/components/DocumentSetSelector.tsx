import { useEffect, useRef, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import {
  listDocumentSets,
  createDocumentSet,
  listDocumentsInSet,
  renameDocumentSet,
  deleteDocumentSet,
  deleteDocument,
  uploadDocument,
} from '../api/endpoints'
import { CURRENT_SET_KEY, useAppStore } from '../store/appStore'
import { useShallow } from 'zustand/react/shallow'
import type { DocumentInSet, DocumentSet } from '../types/api'

// ── Confirm dialog ────────────────────────────────────────────────

interface DialogState {
  open: boolean
  message: string
  onConfirm: () => void
}

function ConfirmDialog({ message, onConfirm, onCancel }: { message: string; onConfirm: () => void; onCancel: () => void }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.5)', backdropFilter: 'blur(4px)' }}
      onClick={onCancel}
    >
      <div
        className="w-[min(22rem,calc(100vw-2rem))] rounded-2xl border border-[var(--border)] bg-[var(--surface-1)] p-6 shadow-2xl"
        onClick={e => e.stopPropagation()}
      >
        <p className="text-sm leading-relaxed text-[var(--text-0)]">{message}</p>
        <div className="mt-5 flex justify-end gap-2">
          <button
            onClick={onCancel}
            className="rounded-xl border border-[var(--border)] bg-[var(--surface-2)] px-4 py-2 text-sm font-semibold text-[var(--text-1)] transition-colors hover:border-[var(--accent-border)] hover:text-[var(--text-0)]"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="rounded-xl border border-red-500/40 bg-red-500/10 px-4 py-2 text-sm font-semibold text-red-400 transition-colors hover:bg-red-500/20 hover:text-red-300"
          >
            OK
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Upload zone per set ───────────────────────────────────────────

function SetUploadZone({ setId, sessionId }: { setId: string; sessionId: string }) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { 'application/pdf': ['.pdf'] },
    multiple: true,
    onDrop: async (files) => {
      for (const file of files) {
        try {
          await uploadDocument(file, setId, sessionId)
        } catch (e: any) {
          useAppStore.getState().showError({
            code: 'INDEX_FAILED',
            message: e?.response?.data?.detail ?? 'Upload failed',
            recoverable: true,
          })
        }
      }
    },
  })

  return (
    <div
      {...getRootProps()}
      className={`cursor-pointer select-none rounded-xl border border-dashed px-4 py-3 text-center text-xs transition-all ${
        isDragActive
          ? 'border-[var(--accent)] bg-[var(--accent-subtle)]'
          : 'border-[var(--border)] bg-[var(--surface-2)] hover:border-[var(--accent-border)] hover:bg-[var(--surface-3)]'
      }`}
    >
      <input {...getInputProps()} />
      <span className="font-semibold text-[var(--text-2)]">
        {isDragActive ? 'Drop PDFs here' : '+ Upload PDF to this set'}
      </span>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────

export function DocumentSetSelector() {
  const { currentSetId, setCurrentSet, sessionId } = useAppStore(useShallow(s => ({
    currentSetId: s.currentSetId,
    setCurrentSet: s.setCurrentSet,
    sessionId: s.sessionId,
  })))
  const indexingJobs = useAppStore(s => s.indexingJobs)

  const [sets, setSets] = useState<DocumentSet[]>([])
  const [expandedSetId, setExpandedSetId] = useState<string | null>(null)
  const [setDocuments, setSetDocuments] = useState<Record<string, DocumentInSet[]>>({})
  const [loadingDocs, setLoadingDocs] = useState<Record<string, boolean>>({})
  const [newName, setNewName] = useState('')
  const [creating, setCreating] = useState(false)
  const [renamingSetId, setRenamingSetId] = useState<string | null>(null)
  const [renameValue, setRenameValue] = useState('')
  const renameInputRef = useRef<HTMLInputElement>(null)
  const prevCompletedCountRef = useRef(0)

  const [dialog, setDialog] = useState<DialogState>({ open: false, message: '', onConfirm: () => {} })

  const openDialog = (message: string, onConfirm: () => void) =>
    setDialog({ open: true, message, onConfirm })
  const closeDialog = () => setDialog(d => ({ ...d, open: false }))
  const confirmDialog = () => { dialog.onConfirm(); closeDialog() }

  const load = async () => {
    const data = await listDocumentSets()
    setSets(data)
    if (!currentSetId && data.length > 0) {
      const storedSetId = localStorage.getItem(CURRENT_SET_KEY)
      const preferred = data.find(set => set.id === storedSetId) ?? data[0]
      setCurrentSet(preferred.id)
    }
  }

  useEffect(() => {
    void load()
  }, [])

  const completedCount = Object.values(indexingJobs).filter(j => j.done).length
  useEffect(() => {
    if (completedCount > prevCompletedCountRef.current) {
      prevCompletedCountRef.current = completedCount
      void load()
      if (expandedSetId) void loadDocsForSet(expandedSetId, true)
    } else {
      prevCompletedCountRef.current = completedCount
    }
  }, [completedCount, expandedSetId])

  const loadDocsForSet = async (setId: string, force = false) => {
    if (loadingDocs[setId] && !force) return
    setLoadingDocs(prev => ({ ...prev, [setId]: true }))
    try {
      const docs = await listDocumentsInSet(setId)
      setSetDocuments(prev => ({ ...prev, [setId]: docs }))
    } catch {
      setSetDocuments(prev => ({ ...prev, [setId]: [] }))
    } finally {
      setLoadingDocs(prev => ({ ...prev, [setId]: false }))
    }
  }

  const toggleExpand = (setId: string) => {
    const isExpanding = expandedSetId !== setId
    setExpandedSetId(isExpanding ? setId : null)
    if (isExpanding && setDocuments[setId] === undefined) {
      void loadDocsForSet(setId)
    }
    setCurrentSet(setId)
  }

  const handleCreate = async () => {
    const name = newName.trim()
    if (!name) return
    setCreating(true)
    try {
      const set = await createDocumentSet(name)
      setNewName('')
      await load()
      setCurrentSet(set.id)
      setExpandedSetId(set.id)
    } catch (e: any) {
      useAppStore.getState().showError({
        code: 'INDEX_FAILED',
        message: e?.response?.data?.detail ?? 'Failed to create set',
        recoverable: true,
      })
    } finally {
      setCreating(false)
    }
  }

  const handleDeleteSet = (e: React.MouseEvent, setId: string, setName: string) => {
    e.stopPropagation()
    openDialog(
      `Delete "${setName}" and all its documents? This cannot be undone.`,
      async () => {
        try {
          await deleteDocumentSet(setId)
          if (currentSetId === setId) setCurrentSet(null)
          if (expandedSetId === setId) setExpandedSetId(null)
          setSetDocuments(prev => {
            const next = { ...prev }
            delete next[setId]
            return next
          })
          await load()
        } catch (e: any) {
          useAppStore.getState().showError({
            code: 'INDEX_FAILED',
            message: e?.response?.data?.detail ?? 'Failed to delete set',
            recoverable: true,
          })
        }
      },
    )
  }

  const startRename = (e: React.MouseEvent, set: DocumentSet) => {
    e.stopPropagation()
    setRenamingSetId(set.id)
    setRenameValue(set.name)
    setTimeout(() => renameInputRef.current?.focus(), 0)
  }

  const confirmRename = async (setId: string) => {
    const name = renameValue.trim()
    if (!name) { setRenamingSetId(null); return }
    try {
      await renameDocumentSet(setId, name)
      setSets(prev => prev.map(s => s.id === setId ? { ...s, name } : s))
      setRenamingSetId(null)
    } catch (e: any) {
      useAppStore.getState().showError({
        code: 'INDEX_FAILED',
        message: e?.response?.data?.detail ?? 'Failed to rename set',
        recoverable: true,
      })
    }
  }

  const handleDeleteDoc = (docId: string, docName: string, setId: string) => {
    openDialog(
      `Remove "${docName}" from this set?`,
      async () => {
        try {
          await deleteDocument(docId)
          setSetDocuments(prev => ({
            ...prev,
            [setId]: (prev[setId] ?? []).filter(d => d.doc_id !== docId),
          }))
          await load()
        } catch (e: any) {
          useAppStore.getState().showError({
            code: 'INDEX_FAILED',
            message: e?.response?.data?.detail ?? 'Failed to remove document',
            recoverable: true,
          })
        }
      },
    )
  }

  return (
    <>
      {dialog.open && (
        <ConfirmDialog
          message={dialog.message}
          onConfirm={confirmDialog}
          onCancel={closeDialog}
        />
      )}

      <section className="rounded-3xl border border-[var(--border)] bg-[var(--surface-1)] p-4 shadow-[var(--shadow-sm)]">
        <div className="text-sm font-mono font-semibold uppercase tracking-[0.18em] text-[var(--text-2)]">
          Document sets
        </div>

        {/* Create new set — at top */}
        <div className="mt-3 flex gap-2">
          <input
            value={newName}
            onChange={e => setNewName(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && void handleCreate()}
            placeholder="New set name..."
            className="min-w-0 flex-1 rounded-xl border border-[var(--border)] bg-[var(--surface-2)] px-3 py-2 text-sm text-[var(--text-1)] outline-none transition-colors placeholder:text-[var(--text-2)] focus:border-[var(--accent-border)]"
          />
          <button
            onClick={() => void handleCreate()}
            disabled={creating || !newName.trim()}
            className="whitespace-nowrap rounded-xl border border-[var(--accent-border)] bg-[var(--accent-subtle)] px-3 py-2 text-xs font-semibold uppercase tracking-[0.18em] text-[var(--accent)] transition-colors hover:bg-[var(--accent-hover)] disabled:cursor-not-allowed disabled:opacity-40"
          >
            {creating ? '...' : '+ Create'}
          </button>
        </div>

        {/* Accordion sets list */}
        <div className="mt-4 space-y-2">
          {sets.map(set => {
            const active = currentSetId === set.id
            const expanded = expandedSetId === set.id
            const docs = setDocuments[set.id] ?? []
            const isLoadingDocs = !!loadingDocs[set.id]

            return (
              <div
                key={set.id}
                className={`rounded-2xl border transition-all ${
                  active
                    ? 'border-[var(--accent-border)] bg-[var(--accent-subtle)]'
                    : 'border-[var(--border)] bg-[var(--surface-2)]'
                }`}
              >
                {/* Set header row */}
                <div
                  className="flex cursor-pointer items-start gap-2 px-3 py-3"
                  onClick={() => toggleExpand(set.id)}
                >
                  <span
                    className="mt-0.5 shrink-0 text-[10px] text-[var(--text-2)] transition-transform duration-200"
                    style={{ display: 'inline-block', transform: expanded ? 'rotate(90deg)' : 'rotate(0deg)' }}
                  >
                    ▶
                  </span>

                  {/* Name / rename input */}
                  {renamingSetId === set.id ? (
                    <input
                      ref={renameInputRef}
                      value={renameValue}
                      onChange={e => setRenameValue(e.target.value)}
                      onKeyDown={e => {
                        if (e.key === 'Enter') void confirmRename(set.id)
                        if (e.key === 'Escape') setRenamingSetId(null)
                      }}
                      onClick={e => e.stopPropagation()}
                      className="min-w-0 flex-1 rounded-lg border border-[var(--accent-border)] bg-[var(--surface-1)] px-2 py-1 text-sm text-[var(--text-0)] outline-none"
                    />
                  ) : (
                    <div className="min-w-0 flex-1">
                      <div className={`truncate text-sm font-semibold ${active ? 'text-[var(--text-0)]' : 'text-[var(--text-1)]'}`}>
                        {set.name}
                      </div>
                      <div className="mt-1 flex flex-wrap gap-1.5 text-xs font-mono text-[var(--text-2)]">
                        <span className="rounded-md border border-[var(--border)] bg-[var(--surface-1)] px-1.5 py-0.5">
                          {set.doc_count} doc{set.doc_count === 1 ? '' : 's'}
                        </span>
                        <span className="rounded-md border border-[var(--border)] bg-[var(--surface-1)] px-1.5 py-0.5">
                          {set.chunk_count} chunks
                        </span>
                      </div>
                    </div>
                  )}

                  {/* Action buttons */}
                  <div className="flex shrink-0 gap-1" onClick={e => e.stopPropagation()}>
                    {renamingSetId === set.id ? (
                      <>
                        <button
                          onClick={() => void confirmRename(set.id)}
                          className="rounded-lg border border-[var(--accent-border)] bg-[var(--accent-subtle)] px-2 py-1 text-xs font-semibold text-[var(--accent)] hover:bg-[var(--accent-hover)]"
                        >
                          Save
                        </button>
                        <button
                          onClick={() => setRenamingSetId(null)}
                          className="rounded-lg border border-[var(--border)] px-2 py-1 text-xs text-[var(--text-2)] hover:text-[var(--text-0)]"
                        >
                          Cancel
                        </button>
                      </>
                    ) : (
                      <>
                        <button
                          onClick={e => startRename(e, set)}
                          title="Rename set"
                          className="rounded-lg border border-[var(--border)] px-2 py-1 text-xs text-[var(--text-2)] transition-colors hover:border-[var(--accent-border)] hover:text-[var(--text-0)]"
                        >
                          Modify
                        </button>
                        <button
                          onClick={e => handleDeleteSet(e, set.id, set.name)}
                          title="Delete set"
                          className="rounded-lg border border-[var(--border)] px-2 py-1 text-xs text-red-400 transition-colors hover:border-red-400/40 hover:text-red-300"
                        >
                          Delete
                        </button>
                      </>
                    )}
                  </div>
                </div>

                {/* Expanded content */}
                {expanded && (
                  <div className="space-y-2 border-t border-[var(--border)] px-3 pb-3 pt-3">
                    <SetUploadZone setId={set.id} sessionId={sessionId} />

                    {isLoadingDocs && (
                      <div className="py-2 text-center text-xs text-[var(--text-2)]">Loading documents...</div>
                    )}

                    {!isLoadingDocs && docs.length === 0 && (
                      <div className="py-2 text-center text-xs text-[var(--text-2)]">
                        No documents yet. Upload a PDF above.
                      </div>
                    )}

                    {docs.map(doc => (
                      <div
                        key={doc.doc_id}
                        className="flex items-center gap-2 rounded-xl border border-[var(--border)] bg-[var(--surface-1)] px-3 py-2"
                      >
                        <div className="min-w-0 flex-1">
                          <div className="truncate text-xs font-medium text-[var(--text-1)]">{doc.document_name}</div>
                          <div className="mt-0.5 text-xs text-[var(--text-2)]">{doc.chunk_count} chunks</div>
                        </div>
                        <button
                          onClick={() => handleDeleteDoc(doc.doc_id, doc.document_name, set.id)}
                          className="whitespace-nowrap rounded-lg border border-[var(--border)] px-2 py-1 text-xs text-red-400 transition-colors hover:border-red-400/40 hover:text-red-300"
                        >
                          Remove
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )
          })}

          {sets.length === 0 && (
            <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface-2)] px-4 py-4 text-center text-sm text-[var(--text-2)]">
              No document sets yet. Create one above to start indexing PDFs.
            </div>
          )}
        </div>
      </section>
    </>
  )
}
