import { useLayoutEffect } from 'react'
import { deleteChatSession } from '../api/endpoints'
import {
  CURRENT_SET_KEY,
  GLOBAL_SESSION_KEY,
  SESSION_SET_KEY,
  useAppStore,
} from '../store/appStore'

function persistSession(sessionId: string, currentSetId: string | null) {
  localStorage.setItem(GLOBAL_SESSION_KEY, sessionId)
  if (currentSetId) localStorage.setItem(SESSION_SET_KEY, currentSetId)
  else localStorage.removeItem(SESSION_SET_KEY)
}

function createFreshSession(currentSetId: string | null) {
  const fresh = crypto.randomUUID()
  persistSession(fresh, currentSetId)
  return fresh
}

export function useSession(currentSetId: string | null) {
  const sessionId = useAppStore(s => s.sessionId)
  const setSessionId = useAppStore(s => s.setSessionId)
  const clearMessages = useAppStore(s => s.clearMessages)
  const clearActivePath = useAppStore(s => s.clearActivePath)
  const closeSourcePanel = useAppStore(s => s.closeSourcePanel)

  useLayoutEffect(() => {
    if (!currentSetId) return

    const storedSession = localStorage.getItem(GLOBAL_SESSION_KEY)
    const storedSetId = localStorage.getItem(SESSION_SET_KEY)

    if (storedSession && storedSetId === currentSetId) {
      if (storedSession !== sessionId) setSessionId(storedSession)
      return
    }

    const fresh = createFreshSession(currentSetId)
    if (fresh !== sessionId) setSessionId(fresh)
  }, [currentSetId, sessionId, setSessionId])

  const resetSession = async () => {
    const previousSessionId = sessionId
    try {
      await deleteChatSession(previousSessionId)
    } catch {
      // We still rotate locally so the user is never blocked by cleanup.
    }

    const fresh = createFreshSession(currentSetId)
    setSessionId(fresh)
    clearMessages()
    clearActivePath()
    closeSourcePanel()

    if (currentSetId) localStorage.setItem(CURRENT_SET_KEY, currentSetId)
  }

  return { sessionId, resetSession }
}
