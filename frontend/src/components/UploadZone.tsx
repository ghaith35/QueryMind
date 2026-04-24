import { useDropzone } from 'react-dropzone'
import { useAppStore } from '../store/appStore'
import { useShallow } from 'zustand/react/shallow'
import { uploadDocument } from '../api/endpoints'

interface UploadZoneProps {
  documentSetId: string
}

export function UploadZone({ documentSetId }: UploadZoneProps) {
  const { sessionId, indexingJobs } = useAppStore(useShallow(s => ({
    sessionId: s.sessionId,
    indexingJobs: s.indexingJobs,
  })))

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { 'application/pdf': ['.pdf'] },
    multiple: true,
    onDrop: async (files) => {
      for (const file of files) {
        try {
          await uploadDocument(file, documentSetId, sessionId)
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

  const jobs = Object.values(indexingJobs)

  return (
    <section className="rounded-3xl border border-[var(--border)] bg-[var(--surface-1)] p-5 shadow-[var(--shadow-sm)]">
      <div className="text-sm font-mono font-semibold uppercase tracking-[0.18em] text-[var(--text-2)]">
        Upload PDFs
      </div>
      <div className="mt-3 text-sm leading-relaxed text-[var(--text-1)]">
        Add new documents to the active set and let the graph rebuild from them.
      </div>

      <div
        {...getRootProps()}
        className={`mt-4 cursor-pointer select-none rounded-[1.75rem] border border-dashed p-5 text-center transition-all ${
          isDragActive
            ? 'border-[var(--accent)] bg-[var(--accent-subtle)] shadow-[var(--shadow-glow)]'
            : 'border-[var(--border)] bg-[var(--surface-2)] hover:border-[var(--accent-border)] hover:bg-[var(--surface-3)]'
        }`}
      >
        <input {...getInputProps()} />
        <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl border border-[var(--border)] bg-[var(--surface-1)] text-lg font-bold shadow-[var(--shadow-sm)] text-[var(--text-1)]">
          PDF
        </div>
        <p className="mt-4 text-sm font-semibold text-[var(--text-1)]">
          {isDragActive ? 'Drop the PDFs here' : 'Drop PDFs here or click to browse'}
        </p>
        <p className="mt-2 text-xs leading-relaxed text-[var(--text-2)]">
          Best with text-based PDFs. OCR-heavy scans may be rejected so citations stay reliable.
        </p>
      </div>

      {jobs.length > 0 && (
        <div className="mt-3 space-y-2">
          {jobs.map((job, index) => (
            <div key={`${job.filename}-${index}`} className="rounded-2xl border border-[var(--border)] bg-[var(--surface-2)] p-4 shadow-[var(--shadow-sm)]">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="truncate text-sm font-medium text-[var(--text-1)]">{job.filename}</div>
                  <div className="mt-2 text-xs text-[var(--text-2)] capitalize">{job.stage}</div>
                </div>
                <div className="rounded-lg border border-[var(--border)] bg-[var(--surface-1)] px-2 py-1 text-xs font-mono text-[var(--text-2)]">
                  {job.done ? 'Done' : `${job.percent.toFixed(0)}%`}
                </div>
              </div>
              <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-[var(--border)]">
                <div
                  className="h-full rounded-full bg-[var(--accent)] transition-all duration-300"
                  style={{ width: `${job.percent}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  )
}
