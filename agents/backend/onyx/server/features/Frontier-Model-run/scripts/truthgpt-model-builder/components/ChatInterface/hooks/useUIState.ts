import { useState, useCallback } from 'react'

export interface UIState {
  showSummary: boolean
  showClustering: boolean
  showScheduler: boolean
  showArchive: boolean
  showDiffView: boolean
  showReminders: boolean
  showAnalytics: boolean
  showBackup: boolean
  showPinnedMessages: boolean
  showMarkdownPreview: boolean
  showExportMenu: boolean
  showShareMenu: boolean
  showTemplateEditor: boolean
  showSessionPlayer: boolean
  showBookmarks: boolean
  showFlashcards: boolean
  showFolders: boolean
  showConnectionInfo: boolean
  showQuickActions: boolean
  previewMessageId: string | null
  diffMessages: [string, string] | null
  shareTarget: string | null
}

export interface UIActions {
  toggleSummary: () => void
  toggleClustering: () => void
  toggleScheduler: () => void
  toggleArchive: () => void
  toggleDiffView: () => void
  toggleReminders: () => void
  toggleAnalytics: () => void
  toggleBackup: () => void
  togglePinnedMessages: () => void
  toggleMarkdownPreview: () => void
  toggleExportMenu: () => void
  toggleShareMenu: () => void
  toggleTemplateEditor: () => void
  toggleSessionPlayer: () => void
  toggleBookmarks: () => void
  toggleFlashcards: () => void
  toggleFolders: () => void
  toggleConnectionInfo: () => void
  toggleQuickActions: () => void
  setPreviewMessageId: (id: string | null) => void
  setDiffMessages: (messages: [string, string] | null) => void
  setShareTarget: (target: string | null) => void
}

export function useUIState() {
  const [showSummary, setShowSummary] = useState(false)
  const [showClustering, setShowClustering] = useState(false)
  const [showScheduler, setShowScheduler] = useState(false)
  const [showArchive, setShowArchive] = useState(false)
  const [showDiffView, setShowDiffView] = useState(false)
  const [showReminders, setShowReminders] = useState(false)
  const [showAnalytics, setShowAnalytics] = useState(false)
  const [showBackup, setShowBackup] = useState(false)
  const [showPinnedMessages, setShowPinnedMessages] = useState(true)
  const [showMarkdownPreview, setShowMarkdownPreview] = useState(false)
  const [showExportMenu, setShowExportMenu] = useState(false)
  const [showShareMenu, setShowShareMenu] = useState(false)
  const [showTemplateEditor, setShowTemplateEditor] = useState(false)
  const [showSessionPlayer, setShowSessionPlayer] = useState(false)
  const [showBookmarks, setShowBookmarks] = useState(false)
  const [showFlashcards, setShowFlashcards] = useState(false)
  const [showFolders, setShowFolders] = useState(false)
  const [showConnectionInfo, setShowConnectionInfo] = useState(false)
  const [showQuickActions, setShowQuickActions] = useState(false)
  const [previewMessageId, setPreviewMessageId] = useState<string | null>(null)
  const [diffMessages, setDiffMessages] = useState<[string, string] | null>(null)
  const [shareTarget, setShareTarget] = useState<string | null>(null)

  const state: UIState = {
    showSummary,
    showClustering,
    showScheduler,
    showArchive,
    showDiffView,
    showReminders,
    showAnalytics,
    showBackup,
    showPinnedMessages,
    showMarkdownPreview,
    showExportMenu,
    showShareMenu,
    showTemplateEditor,
    showSessionPlayer,
    showBookmarks,
    showFlashcards,
    showFolders,
    showConnectionInfo,
    showQuickActions,
    previewMessageId,
    diffMessages,
    shareTarget
  }

  const actions: UIActions = {
    toggleSummary: useCallback(() => setShowSummary(prev => !prev), []),
    setSummary: useCallback((value: boolean) => setShowSummary(value), []),
    toggleClustering: useCallback(() => setShowClustering(prev => !prev), []),
    toggleScheduler: useCallback(() => setShowScheduler(prev => !prev), []),
    toggleArchive: useCallback(() => setShowArchive(prev => !prev), []),
    toggleDiffView: useCallback(() => setShowDiffView(prev => !prev), []),
    toggleReminders: useCallback(() => setShowReminders(prev => !prev), []),
    toggleAnalytics: useCallback(() => setShowAnalytics(prev => !prev), []),
    toggleBackup: useCallback(() => setShowBackup(prev => !prev), []),
    togglePinnedMessages: useCallback(() => setShowPinnedMessages(prev => !prev), []),
    toggleMarkdownPreview: useCallback(() => setShowMarkdownPreview(prev => !prev), []),
    toggleExportMenu: useCallback(() => setShowExportMenu(prev => !prev), []),
    toggleShareMenu: useCallback(() => setShowShareMenu(prev => !prev), []),
    toggleTemplateEditor: useCallback(() => setShowTemplateEditor(prev => !prev), []),
    toggleSessionPlayer: useCallback(() => setShowSessionPlayer(prev => !prev), []),
    toggleBookmarks: useCallback(() => setShowBookmarks(prev => !prev), []),
    toggleFlashcards: useCallback(() => setShowFlashcards(prev => !prev), []),
    toggleFolders: useCallback(() => setShowFolders(prev => !prev), []),
    toggleConnectionInfo: useCallback(() => setShowConnectionInfo(prev => !prev), []),
    toggleQuickActions: useCallback(() => setShowQuickActions(prev => !prev), []),
    setPreviewMessageId: useCallback((id: string | null) => setPreviewMessageId(id), []),
    setDiffMessages: useCallback((messages: [string, string] | null) => setDiffMessages(messages), []),
    setShareTarget: useCallback((target: string | null) => setShareTarget(target), [])
  }

  return { state, actions }
}

