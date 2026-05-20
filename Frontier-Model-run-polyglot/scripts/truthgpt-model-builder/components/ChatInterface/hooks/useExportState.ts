import { useState, useCallback } from 'react'

export interface ExportState {
  exportFormats: Set<'json' | 'txt' | 'md' | 'html' | 'csv' | 'pdf'>
  backupHistory: any[]
  conversationSummary: string
  conversationInsights: any | null
}

export interface ExportActions {
  setExportFormats: (formats: Set<'json' | 'txt' | 'md' | 'html' | 'csv' | 'pdf'>) => void
  toggleExportFormat: (format: 'json' | 'txt' | 'md' | 'html' | 'csv' | 'pdf') => void
  setBackupHistory: (history: any[]) => void
  addBackup: (backup: any) => void
  setConversationSummary: (summary: string) => void
  setConversationInsights: (insights: any | null) => void
}

export function useExportState() {
  const [exportFormats, setExportFormats] = useState<Set<'json' | 'txt' | 'md' | 'html' | 'csv' | 'pdf'>>(
    new Set(['json', 'txt', 'md'])
  )
  const [backupHistory, setBackupHistory] = useState<any[]>([])
  const [conversationSummary, setConversationSummary] = useState<string>('')
  const [conversationInsights, setConversationInsights] = useState<any | null>(null)

  const state: ExportState = {
    exportFormats,
    backupHistory,
    conversationSummary,
    conversationInsights
  }

  const actions: ExportActions = {
    setExportFormats: useCallback((formats: Set<'json' | 'txt' | 'md' | 'html' | 'csv' | 'pdf'>) => {
      setExportFormats(formats)
    }, []),
    toggleExportFormat: useCallback((format: 'json' | 'txt' | 'md' | 'html' | 'csv' | 'pdf') => {
      setExportFormats(prev => {
        const newSet = new Set(prev)
        if (newSet.has(format)) {
          newSet.delete(format)
        } else {
          newSet.add(format)
        }
        return newSet
      })
    }, []),
    setBackupHistory: useCallback((history: any[]) => setBackupHistory(history), []),
    addBackup: useCallback((backup: any) => {
      setBackupHistory(prev => [...prev, backup].slice(-20)) // Mantener últimas 20
    }, []),
    setConversationSummary: useCallback((summary: string) => setConversationSummary(summary), []),
    setConversationInsights: useCallback((insights: any | null) => setConversationInsights(insights), [])
  }

  return { state, actions }
}



