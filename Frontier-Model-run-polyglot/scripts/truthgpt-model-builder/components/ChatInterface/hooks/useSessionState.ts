import { useState, useCallback } from 'react'

export interface SessionState {
  recordedSessions: any[]
  currentSession: any | null
  suggestionMode: 'auto' | 'manual' | 'off'
  commandHistory: string[]
  messageTemplates: string[]
}

export interface SessionActions {
  setRecordedSessions: (sessions: any[]) => void
  addRecordedSession: (session: any) => void
  setCurrentSession: (session: any | null) => void
  setSuggestionMode: (mode: 'auto' | 'manual' | 'off') => void
  addToCommandHistory: (command: string) => void
  clearCommandHistory: () => void
  addMessageTemplate: (template: string) => void
  removeMessageTemplate: (index: number) => void
}

export function useSessionState() {
  const [recordedSessions, setRecordedSessions] = useState<any[]>([])
  const [currentSession, setCurrentSession] = useState<any | null>(null)
  const [suggestionMode, setSuggestionMode] = useState<'auto' | 'manual' | 'off'>('auto')
  const [commandHistory, setCommandHistory] = useState<string[]>([])
  const [messageTemplates, setMessageTemplates] = useState<string[]>([])

  const state: SessionState = {
    recordedSessions,
    currentSession,
    suggestionMode,
    commandHistory,
    messageTemplates
  }

  const actions: SessionActions = {
    setRecordedSessions: useCallback((sessions: any[]) => setRecordedSessions(sessions), []),
    addRecordedSession: useCallback((session: any) => {
      setRecordedSessions(prev => [...prev, session])
    }, []),
    setCurrentSession: useCallback((session: any | null) => setCurrentSession(session), []),
    setSuggestionMode: useCallback((mode: 'auto' | 'manual' | 'off') => setSuggestionMode(mode), []),
    addToCommandHistory: useCallback((command: string) => {
      setCommandHistory(prev => [...prev, command].slice(-50)) // Mantener últimas 50
    }, []),
    clearCommandHistory: useCallback(() => setCommandHistory([]), []),
    addMessageTemplate: useCallback((template: string) => {
      setMessageTemplates(prev => [...prev, template])
    }, []),
    removeMessageTemplate: useCallback((index: number) => {
      setMessageTemplates(prev => prev.filter((_, i) => i !== index))
    }, [])
  }

  return { state, actions }
}



