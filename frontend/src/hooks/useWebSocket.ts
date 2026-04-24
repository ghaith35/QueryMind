import { useEffect, useRef } from 'react'
import { useAppStore } from '../store/appStore'
import { getGraph } from '../api/endpoints'

export function useWebSocket(sessionId: string) {
  const wsRef = useRef<WebSocket | null>(null)
  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const graphRefreshRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const activePathTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  // Each mount gets a unique "generation" id. The onclose handler only
  // schedules a retry if the generation hasn't changed (i.e. not a StrictMode cleanup).
  const generationRef = useRef(0)

  useEffect(() => {
    const generation = ++generationRef.current

    function scheduleGraphRefresh(delayMs = 150) {
      if (graphRefreshRef.current) clearTimeout(graphRefreshRef.current)
      graphRefreshRef.current = setTimeout(() => {
        const state = useAppStore.getState()
        if (!state.currentSetId || !state.graphLoaded) return
        getGraph(state.currentSetId, state.graphFilters)
          .then(data => useAppStore.getState().setGraphResponse(data))
          .catch(() => {})
      }, delayMs)
    }

    function connect() {
      const base = import.meta.env.VITE_WS_BASE ?? 'ws://localhost:8002'
      const url = `${base}/ws/${sessionId}`
      console.log('[WS] connecting to', url)

      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => {
        console.log('[WS] connected — session:', sessionId.slice(0, 8))
        if (retryRef.current) { clearTimeout(retryRef.current); retryRef.current = null }
        const { currentSetId, graphLoaded } = useAppStore.getState()
        if (currentSetId && graphLoaded) {
          scheduleGraphRefresh(0)
        }
      }

      ws.onmessage = (event) => {
        const s = useAppStore.getState()
        let msg: { type: string; payload: any }
        try { msg = JSON.parse(event.data) } catch { return }
        console.log('[WS] message:', msg.type)

        switch (msg.type) {
          case 'index_progress':  s.updateIndexProgress(msg.payload); break
          case 'index_complete':
            s.markIndexComplete(msg.payload)
            if (s.currentSetId && s.graphLoaded) scheduleGraphRefresh(0)
            break
          case 'graph_update':
            if (msg.payload.document_set_id === s.currentSetId && s.graphLoaded) scheduleGraphRefresh()
            break
          case 'answer_stream':
            s.appendStreamingToken(msg.payload.turn_id, msg.payload.token)
            break
          case 'answer_complete':
            s.finalizeAnswer(msg.payload)
            s.setActivePath(msg.payload.active_node_ids, msg.payload.active_edge_ids)
            if (activePathTimeoutRef.current) clearTimeout(activePathTimeoutRef.current)
            activePathTimeoutRef.current = setTimeout(() => {
              useAppStore.getState().clearActivePath()
              activePathTimeoutRef.current = null
            }, 6000)
            break
          case 'error': s.showError(msg.payload); break
        }
      }

      ws.onerror = (e) => console.warn('[WS] error', e)

      ws.onclose = (e) => {
        console.log('[WS] closed', e.code)
        // Only retry if this is still the current generation (not a StrictMode cleanup)
        if (generationRef.current !== generation) return
        retryRef.current = setTimeout(connect, 2000)
      }

      const heartbeat = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send('ping')
      }, 30_000)
      ws.addEventListener('close', () => clearInterval(heartbeat))
    }

    connect()

    return () => {
      // Bumping generation prevents the onclose from scheduling a retry
      generationRef.current++
      if (retryRef.current) { clearTimeout(retryRef.current); retryRef.current = null }
      if (graphRefreshRef.current) { clearTimeout(graphRefreshRef.current); graphRefreshRef.current = null }
      if (activePathTimeoutRef.current) { clearTimeout(activePathTimeoutRef.current); activePathTimeoutRef.current = null }
      wsRef.current?.close()
    }
  }, [sessionId])

  return wsRef
}
