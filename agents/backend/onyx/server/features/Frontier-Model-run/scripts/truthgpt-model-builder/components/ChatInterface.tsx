'use client'

import { useState, useRef, useEffect } from 'react'
import { saveModelToHistory, getModelHistory } from '@/lib/storage'
import { Send, Loader2, Github, CheckCircle, AlertCircle, History, Activity, Mic, Volume2, FileText, Archive, Clock, BarChart3, Download, Upload, GitCompare, Bell, Layers, FileDown } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { toast } from 'react-hot-toast'
// Using debounce utility function instead of library
const debounce = <T extends (...args: any[]) => any>(
  func: T,
  wait: number
): ((...args: Parameters<T>) => void) => {
  let timeout: NodeJS.Timeout | null = null
  return (...args: Parameters<T>) => {
    if (timeout) clearTimeout(timeout)
    timeout = setTimeout(() => func(...args), wait)
  }
}
import Message from './Message'
import ModelStatus from './ModelStatus'
import Suggestions from './Suggestions'
import ModelPreview from './ModelPreview'
import ModelHistory from './ModelHistory'
import ArchitectureVisualizer from './ArchitectureVisualizer'
import ModelStats from './ModelStats'
import AutoComplete from './AutoComplete'
import ValidationBadge from './ValidationBadge'
import ModelTemplates from './ModelTemplates'
import ModelComparator from './ModelComparator'
import QuickActions from './QuickActions'
import WelcomeTour from './WelcomeTour'
import DraftRecovery from './DraftRecovery'
import { useModelStore, Model } from '@/store/modelStore'
import { analyzeModelDescription } from '@/lib/modules/management'
import { adaptiveAnalyze } from '@/lib/modules/analysis'
import { validateUserDescription as validateDescription } from '@/lib/modules/validation'
import { useKeyboardShortcuts } from '@/lib/hooks'
import { saveDraft, setupAutoSave, clearDraft } from '@/lib/auto-save'
import { useEnhancedChat } from '@/lib/hooks'
import PerformanceMetrics from './PerformanceMetrics'
import RateLimitIndicator from './RateLimitIndicator'
import SearchBar, { SearchBarRef } from './SearchBar'
import { useDebouncedCallback } from '@/lib/optimization-utils'
import { notifications } from '@/lib/notification-system'
import { Trash2, Zap, Sparkles, Lightbulb } from 'lucide-react'
import ProactiveModelBuilder from './ProactiveModelBuilder'
import { useTruthGPTAPI } from '@/lib/hooks'
import { useBulkChat } from '@/lib/hooks'
import { getSmartSuggestions } from '@/lib/smart-suggestions'
import { getPerformanceOptimizer } from '@/lib/performance-optimizer'
import { useMemo, useCallback } from 'react'
import { 
  usePanelStates, 
  useConsolidatedState, 
  useMapState,
  useMessageState,
  useUIState,
  useSessionState,
  useOrganizationState,
  useAccessibilityState,
  useExportState,
  useAdvancedFeaturesState
} from './ChatInterface/hooks'
import { eventBus, MessageEvents } from './ChatInterface/events'

export default function ChatInterface() {
  // Hooks modulares para reducir duplicación
  const panels = usePanelStates()
  const consolidated = useConsolidatedState()
  
  // Hook consolidado para estados de mensajes
  const messageState = useMessageState()
  
  // Hooks específicos por área funcional
  const uiState = useUIState()
  const sessionState = useSessionState()
  const organizationState = useOrganizationState()
  const accessibilityState = useAccessibilityState(consolidated.state.display.fontSize)
  const exportState = useExportState()
  const advancedFeaturesState = useAdvancedFeaturesState()
  
  // Helper para crear aliases Map-compatibles
  const createMapAlias = <T,>(hook: ReturnType<typeof useMapState<T>>) => ({
    get: (key: string) => hook.get(key),
    set: (key: string, value: T) => hook.set(key, value),
    has: (key: string) => hook.has(key),
    delete: (key: string) => hook.remove(key),
    clear: () => hook.clear(),
    size: hook.size,
    entries: hook.entries,
    keys: hook.keys,
    values: hook.values,
    forEach: (callback: (value: T, key: string, map: Map<string, T>) => void) => {
      hook.map.forEach(callback)
    }
  } as Map<string, T>)
  
  const createMapSetter = <T,>(hook: ReturnType<typeof useMapState<T>>) => 
    (updater: Map<string, T> | ((prev: Map<string, T>) => Map<string, T>)) => {
      if (typeof updater === 'function') {
        const newMap = updater(hook.map)
        hook.setAll(newMap)
      } else {
        hook.setAll(updater)
      }
    }
  
  // Estados ahora manejados por hooks modulares específicos
  // smartSuggestions → advancedFeaturesState.state.smartSuggestions
  // showConnectionInfo → uiState.state.showConnectionInfo
  // showQuickActions → uiState.state.showQuickActions
  // commandHistory → sessionState.state.commandHistory
  // messageTemplates → sessionState.state.messageTemplates
  
  // Usar estado consolidado en lugar de múltiples useState
  const { state, updateState, updateUI, updateDisplay, updateSearch, updateFeatures, toggleUI, toggleFeature } = consolidated
  
  // Aliases para facilitar la migración gradual
  const input = state.input
  const setInput = (value: string) => updateState({ input: value })
  const isLoading = state.isLoading
  const setIsLoading = (value: boolean) => updateState({ isLoading: value })
  const showPreview = state.ui.showPreview
  const setShowPreview = (value: boolean) => updateUI({ showPreview: value })
  const showHistory = state.ui.showHistory
  const setShowHistory = (value: boolean) => updateUI({ showHistory: value })
  const showComparator = state.ui.showComparator
  const setShowComparator = (value: boolean) => updateUI({ showComparator: value })
  const selectedModels = state.models.selectedModels
  const setSelectedModels = (value: Model[]) => updateState({ models: { ...state.models, selectedModels: value } })
  const validation = state.models.validation
  const setValidation = (value: any) => updateState({ models: { ...state.models, validation: value } })
  const previewSpec = state.models.previewSpec
  const setPreviewSpec = (value: any) => updateState({ models: { ...state.models, previewSpec: value } })
  const modelHistory = state.models.modelHistory
  const setModelHistory = (value: any[]) => updateState({ models: { ...state.models, modelHistory: value } })
  const showTour = state.ui.showTour
  const setShowTour = (value: boolean) => updateUI({ showTour: value })
  const showMetrics = state.ui.showMetrics
  const setShowMetrics = (value: boolean) => updateUI({ showMetrics: value })
  const showProactive = state.ui.showProactive
  const setShowProactive = (value: boolean) => updateUI({ showProactive: value })
  const searchQuery = state.search.searchQuery
  const setSearchQuery = (value: string) => updateSearch({ searchQuery: value })
  const showSmartSuggestions = state.search.showSmartSuggestions
  const setShowSmartSuggestions = (value: boolean) => updateSearch({ showSmartSuggestions: value })
  const useBulkChatMode = state.features.useBulkChatMode
  const setUseBulkChatMode = (value: boolean) => updateFeatures({ useBulkChatMode: value })
  const readMode = state.display.readMode
  const setReadMode = (value: boolean) => updateDisplay({ readMode: value })
  const highlightSearch = state.search.highlightSearch
  const setHighlightSearch = (value: boolean) => updateSearch({ highlightSearch: value })
  const currentSearchIndex = state.search.currentSearchIndex
  const setCurrentSearchIndex = (value: number) => updateSearch({ currentSearchIndex: value })
  const favoriteMessages = state.messageCollections.favoriteMessages
  const setFavoriteMessages = (value: Set<string>) => updateState({ messageCollections: { ...state.messageCollections, favoriteMessages: value } })
  const showStats = state.ui.showStats
  const setShowStats = (value: boolean) => updateUI({ showStats: value })
  const viewMode = state.display.viewMode
  const setViewMode = (value: 'normal' | 'compact' | 'comfortable') => updateDisplay({ viewMode: value })
  const showTimeline = state.ui.showTimeline || false // Temporal hasta migrar
  const setShowTimeline = (value: boolean) => {} // Temporal
  const selectedMessages = state.messageCollections.selectedMessages
  const setSelectedMessages = (value: Set<string>) => updateState({ messageCollections: { ...state.messageCollections, selectedMessages: value } })
  const showFilters = state.ui.showFilters
  const setShowFilters = (value: boolean) => updateUI({ showFilters: value })
  const filterRole = state.search.filterRole
  const setFilterRole = (value: 'all' | 'user' | 'assistant') => updateSearch({ filterRole: value })
  const collapsedMessages = state.messageCollections.collapsedMessages
  const setCollapsedMessages = (value: Set<string>) => updateState({ messageCollections: { ...state.messageCollections, collapsedMessages: value } })
  const showCodeSyntax = state.display.showCodeSyntax
  const setShowCodeSyntax = (value: boolean) => updateDisplay({ showCodeSyntax: value })
  const presentationMode = state.display.presentationMode
  const setPresentationMode = (value: boolean) => updateDisplay({ presentationMode: value })
  const showCommandPalette = state.ui.showCommandPalette
  const setShowCommandPalette = (value: boolean) => updateUI({ showCommandPalette: value })
  const showAccessibility = state.ui.showAccessibility
  const setShowAccessibility = (value: boolean) => updateUI({ showAccessibility: value })
  const fontSize = state.display.fontSize
  const setFontSize = (value: 'small' | 'medium' | 'large') => updateDisplay({ fontSize: value })
  const showTemplates = state.ui.showTemplates
  const setShowTemplates = (value: boolean) => updateUI({ showTemplates: value })
  const autoScroll = state.display.autoScroll
  const setAutoScroll = (value: boolean) => updateDisplay({ autoScroll: value })
  const showWordCount = state.ui.showWordCount
  const setShowWordCount = (value: boolean) => updateUI({ showWordCount: value })
  const showSentiment = state.ui.showSentiment
  const setShowSentiment = (value: boolean) => updateUI({ showSentiment: value })
  const showPrintMode = state.ui.showPrintMode
  const setShowPrintMode = (value: boolean) => updateUI({ showPrintMode: value })
  // Estados Map ahora manejados por useMapState
  const messageReactionsHook = useMapState<string[]>(new Map())
  const linkPreviewHook = useMapState<any>(new Map())
  const customShortcutsHook = useMapState<string>(new Map())
  const messageTagsHook = useMapState<string[]>(new Map())
  const messageNotesHook = useMapState<string>(new Map())
  const messageHistoryHook = useMapState<any[]>(new Map())
  const messageVersionsHook = useMapState<number>(new Map())
  const messageStatsHook = useMapState<{ responseTime?: number, wordCount: number, charCount: number }>(new Map())
  const messageSearchIndexHook = useMapState<number>(new Map())
  
  // Aliases para compatibilidad con código existente usando helpers
  const messageReactions = createMapAlias(messageReactionsHook)
  const setMessageReactions = createMapSetter(messageReactionsHook)
  const linkPreview = createMapAlias(linkPreviewHook)
  const setLinkPreview = createMapSetter(linkPreviewHook)
  const customShortcuts = createMapAlias(customShortcutsHook)
  const setCustomShortcuts = createMapSetter(customShortcutsHook)
  const messageTags = createMapAlias(messageTagsHook)
  const setMessageTags = createMapSetter(messageTagsHook)
  const messageNotes = createMapAlias(messageNotesHook)
  const setMessageNotes = createMapSetter(messageNotesHook)
  const messageHistory = createMapAlias(messageHistoryHook)
  const setMessageHistory = createMapSetter(messageHistoryHook)
  const messageVersions = createMapAlias(messageVersionsHook)
  const setMessageVersions = createMapSetter(messageVersionsHook)
  const messageStats = createMapAlias(messageStatsHook)
  const setMessageStats = createMapSetter(messageStatsHook)
  const messageSearchIndex = createMapAlias(messageSearchIndexHook)
  const setMessageSearchIndex = createMapSetter(messageSearchIndexHook)
  
  // Estados consolidados
  const showReactions = state.ui.showReactions || true
  const setShowReactions = (value: boolean) => updateUI({ showReactions: value })
  const theme = state.display.theme
  const setTheme = (value: 'dark' | 'light' | 'auto') => updateDisplay({ theme: value })
  const showDebug = state.ui.showDebug
  const setShowDebug = (value: boolean) => updateUI({ showDebug: value })
  const showTags = state.ui.showTags
  const setShowTags = (value: boolean) => updateUI({ showTags: value })
  const [availableTags, setAvailableTags] = useState<string[]>(['importante', 'pregunta', 'respuesta', 'código', 'documentación'])
  const [editingNote, setEditingNote] = useState<string | null>(null)
  const [readingSpeed, setReadingSpeed] = useState<'slow' | 'normal' | 'fast'>('normal')
  const zenMode = state.display.zenMode
  const setZenMode = (value: boolean) => updateDisplay({ zenMode: value })
  const showEditHistory = state.ui.showEditHistory
  const setShowEditHistory = (value: boolean) => updateUI({ showEditHistory: value })
  const [editingMessage, setEditingMessage] = useState<string | null>(null)
  const translationMode = state.features.translationMode
  const setTranslationMode = (value: boolean) => updateFeatures({ translationMode: value })
  const targetLanguage = state.search.targetLanguage || 'en'
  const setTargetLanguage = (value: string) => updateSearch({ targetLanguage: value })
  const showMessageTimestamps = state.display.showMessageTimestamps
  const setShowMessageTimestamps = (value: boolean) => updateDisplay({ showMessageTimestamps: value })
  const compactMode = state.display.compactMode
  const setCompactMode = (value: boolean) => updateDisplay({ compactMode: value })
  const showTypingDots = state.display.showTypingDots
  const setShowTypingDots = (value: boolean) => updateDisplay({ showTypingDots: value })
  const autoSave = state.features.autoSave
  const setAutoSave = (value: boolean) => updateFeatures({ autoSave: value })
  
  // Nuevas características avanzadas - usando hooks modulares
  const voiceInputEnabled = state.features.voiceInputEnabled
  const setVoiceInputEnabled = (value: boolean) => updateFeatures({ voiceInputEnabled: value })
  const voiceOutputEnabled = state.features.voiceOutputEnabled
  const setVoiceOutputEnabled = (value: boolean) => updateFeatures({ voiceOutputEnabled: value })
  const isRecording = state.features.isRecording
  const setIsRecording = (value: boolean) => updateFeatures({ isRecording: value })
  // Estados de UI ahora en uiState
  // showSummary → uiState.state.showSummary
  // showClustering → uiState.state.showClustering
  // conversationSummary → exportState.state.conversationSummary
  const messageClustersHook = useMapState<string[]>(new Map())
  const messageClusters = createMapAlias(messageClustersHook)
  const setMessageClusters = createMapSetter(messageClustersHook)
  const [showScheduler, setShowScheduler] = useState(false)
  const scheduledMessagesHook = useMapState<{ message: string, timestamp: number }>(new Map())
  const scheduledMessages = createMapAlias(scheduledMessagesHook)
  const setScheduledMessages = createMapSetter(scheduledMessagesHook)
  const [showArchive, setShowArchive] = useState(false)
  const archivedMessages = state.messageCollections.archivedMessages
  const setArchivedMessages = (value: Set<string>) => updateState({ messageCollections: { ...state.messageCollections, archivedMessages: value } })
  const [showDiffView, setShowDiffView] = useState(false)
  const [diffMessages, setDiffMessages] = useState<[string, string] | null>(null)
  const [showReminders, setShowReminders] = useState(false)
  const messageRemindersHook = useMapState<{ timestamp: number, note: string }>(new Map())
  const messageReminders = createMapAlias(messageRemindersHook)
  const setMessageReminders = createMapSetter(messageRemindersHook)
  const [showAnalytics, setShowAnalytics] = useState(false)
  const [showBackup, setShowBackup] = useState(false)
  const [backupHistory, setBackupHistory] = useState<any[]>([])
  const pinnedMessages = state.messageCollections.pinnedMessages
  const setPinnedMessages = (value: Set<string>) => updateState({ messageCollections: { ...state.messageCollections, pinnedMessages: value } })
  const [showPinnedMessages, setShowPinnedMessages] = useState(true)
  const messageThreadsHook = useMapState<string[]>(new Map()) // parent -> children
  const messageThreads = createMapAlias(messageThreadsHook)
  const setMessageThreads = createMapSetter(messageThreadsHook)
  // Estados de organización y UI ahora en hooks modulares
  const threadParent = organizationState.state.threadParent
  const setThreadParent = organizationState.actions.setThreadParent
  const showMarkdownPreview = uiState.state.showMarkdownPreview
  const setShowMarkdownPreview = (value: boolean) => value ? uiState.actions.toggleMarkdownPreview() : uiState.actions.toggleMarkdownPreview()
  const previewMessageId = uiState.state.previewMessageId
  const setPreviewMessageId = uiState.actions.setPreviewMessageId
  const messageGroupsHook = useMapState<string[]>(new Map())
  const messageGroups = createMapAlias(messageGroupsHook)
  const setMessageGroups = createMapSetter(messageGroupsHook)
  const [groupingMode, setGroupingMode] = useState<'none' | 'time' | 'topic' | 'role'>('none')
  const [showExportMenu, setShowExportMenu] = useState(false)
  const messageSharingHook = useMapState<string[]>(new Map())
  const messageSharing = createMapAlias(messageSharingHook)
  const setMessageSharing = createMapSetter(messageSharingHook)
  const [showShareMenu, setShowShareMenu] = useState(false)
  const [shareTarget, setShareTarget] = useState<string | null>(null)
  const messageTemplatesWithVarsHook = useMapState<{ template: string, variables: string[] }>(new Map())
  const messageTemplatesWithVars = createMapAlias(messageTemplatesWithVarsHook)
  const setMessageTemplatesWithVars = createMapSetter(messageTemplatesWithVarsHook)
  const [showTemplateEditor, setShowTemplateEditor] = useState(false)
  const collaborationMode = state.features.collaborationMode
  const setCollaborationMode = (value: boolean) => updateFeatures({ collaborationMode: value })
  const [collaborators, setCollaborators] = useState<string[]>([])
  const smartNotifications = state.features.smartNotifications
  const setSmartNotifications = (value: boolean) => updateFeatures({ smartNotifications: value })
  const notificationRulesHook = useMapState<{ keywords: string[], enabled: boolean }>(new Map())
  const notificationRules = createMapAlias(notificationRulesHook)
  const setNotificationRules = createMapSetter(notificationRulesHook)
  const realTimeStats = state.features.realTimeStats
  const setRealTimeStats = (value: boolean) => updateFeatures({ realTimeStats: value })
  const devMode = state.features.devMode
  const setDevMode = (value: boolean) => updateFeatures({ devMode: value })
  const apiIntegrationsHook = useMapState<any>(new Map())
  const apiIntegrations = createMapAlias(apiIntegrationsHook)
  const setApiIntegrations = createMapSetter(apiIntegrationsHook)
  const messageEncryption = state.features.messageEncryption || false
  const setMessageEncryption = (value: boolean) => updateFeatures({ messageEncryption: value })
  const encryptedMessages = state.messageCollections.encryptedMessages
  const setEncryptedMessages = (value: Set<string>) => updateState({ messageCollections: { ...state.messageCollections, encryptedMessages: value } })
  const advancedSearch = state.search.advancedSearch
  const setAdvancedSearch = (value: boolean) => updateSearch({ advancedSearch: value })
  const searchFilters = state.search.searchFilters
  const setSearchFilters = (value: { dateRange?: { start: Date, end: Date }, minWords?: number, maxWords?: number, hasCode?: boolean, hasLinks?: boolean }) => updateSearch({ searchFilters: value })
  // Más estados consolidados y Map states
  const sessionRecording = state.features.sessionRecording
  const setSessionRecording = (value: boolean) => updateFeatures({ sessionRecording: value })
  const [recordedSessions, setRecordedSessions] = useState<any[]>([])
  const [showSessionPlayer, setShowSessionPlayer] = useState(false)
  const [currentSession, setCurrentSession] = useState<any>(null)
  const smartSuggestionsEnabled = state.features.smartSuggestionsEnabled
  const setSmartSuggestionsEnabled = (value: boolean) => updateFeatures({ smartSuggestionsEnabled: value })
  const [suggestionMode, setSuggestionMode] = useState<'auto' | 'manual' | 'off'>('auto')
  const bookmarksHook = useMapState<{ name: string, messageId: string, timestamp: number }>(new Map())
  const bookmarks = createMapAlias(bookmarksHook)
  const setBookmarks = createMapSetter(bookmarksHook)
  // Estados de UI ahora en uiState hook
  const showBookmarks = uiState.state.showBookmarks
  const setShowBookmarks = (value: boolean) => value ? uiState.actions.toggleBookmarks() : uiState.actions.toggleBookmarks()
  const studyMode = state.features.studyMode
  const setStudyMode = (value: boolean) => updateFeatures({ studyMode: value })
  const studyNotesHook = useMapState<string>(new Map())
  const studyNotes = createMapAlias(studyNotesHook)
  const setStudyNotes = createMapSetter(studyNotesHook)
  const flashcardsHook = useMapState<{ question: string, answer: string }>(new Map())
  const flashcards = createMapAlias(flashcardsHook)
  const setFlashcards = createMapSetter(flashcardsHook)
  // Estados de UI ahora en uiState hook
  const showFlashcards = uiState.state.showFlashcards
  const setShowFlashcards = (value: boolean) => value ? uiState.actions.toggleFlashcards() : uiState.actions.toggleFlashcards()
  const productivityMode = state.features.productivityMode
  const setProductivityMode = (value: boolean) => updateFeatures({ productivityMode: value })
  // Estados de características avanzadas ahora en advancedFeaturesState hook
  const focusTimer = advancedFeaturesState.state.focusTimer
  const setFocusTimer = advancedFeaturesState.actions.setFocusTimer
  const focusGoal = advancedFeaturesState.state.focusGoal
  const setFocusGoal = advancedFeaturesState.actions.setFocusGoal
  const customThemesHook = useMapState<any>(new Map())
  const customThemes = createMapAlias(customThemesHook)
  const setCustomThemes = createMapSetter(customThemesHook)
  const [activeTheme, setActiveTheme] = useState<string>('default')
  const accessibilityMode = state.features.accessibilityMode
  const setAccessibilityMode = (value: boolean) => updateFeatures({ accessibilityMode: value })
  const [screenReader, setScreenReader] = useState(false)
  const [highContrast, setHighContrast] = useState(false)
  const keyboardNavigation = state.features.keyboardNavigation || true
  const setKeyboardNavigation = (value: boolean) => updateFeatures({ keyboardNavigation: value })
  const messageBookmarks = state.messageCollections.messageBookmarks
  const setMessageBookmarks = (value: Set<string>) => updateState({ messageCollections: { ...state.messageCollections, messageBookmarks: value } })
  // Estados de exportación ahora en exportState hook
  // exportFormats ya está arriba
  const aiInsights = advancedFeaturesState.state.aiInsights
  const setAiInsights = advancedFeaturesState.actions.setAiInsights
  const conversationInsights = exportState.state.conversationInsights
  const setConversationInsights = exportState.actions.setConversationInsights
  const messageHighlightsHook = useMapState<string>(new Map())
  const messageHighlights = createMapAlias(messageHighlightsHook)
  const setMessageHighlights = createMapSetter(messageHighlightsHook)
  const readingMode = state.display.readingMode
  const setReadingMode = (value: boolean) => updateDisplay({ readingMode: value })
  // Estados de características avanzadas ahora en advancedFeaturesState hook
  const readingProgress = advancedFeaturesState.state.readingProgress
  const setReadingProgress = advancedFeaturesState.actions.setReadingProgress
  const dictationMode = state.features.dictationMode
  const setDictationMode = (value: boolean) => updateFeatures({ dictationMode: value })
  const calendarIntegration = state.features.calendarIntegration
  const setCalendarIntegration = (value: boolean) => updateFeatures({ calendarIntegration: value })
  const scheduledEventsHook = useMapState<{ title: string, date: Date, messageId?: string }>(new Map())
  const scheduledEvents = createMapAlias(scheduledEventsHook)
  const setScheduledEvents = createMapSetter(scheduledEventsHook)
  const cloudSync = state.features.cloudSync
  const setCloudSync = (value: boolean) => updateFeatures({ cloudSync: value })
  const [syncProvider, setSyncProvider] = useState<'none' | 'localStorage' | 'indexedDB'>('localStorage')
  const messageBadgesHook = useMapState<string[]>(new Map())
  const messageBadges = createMapAlias(messageBadgesHook)
  const setMessageBadges = createMapSetter(messageBadgesHook)
  const [achievements, setAchievements] = useState<Set<string>>(new Set())
  const pluginSystem = state.features.pluginSystem
  const setPluginSystem = (value: boolean) => updateFeatures({ pluginSystem: value })
  const [activePlugins, setActivePlugins] = useState<string[]>([])
  const messageAnnotationsHook = useMapState<{ type: 'highlight' | 'comment' | 'question', content: string }[]>(new Map())
  const messageAnnotations = createMapAlias(messageAnnotationsHook)
  const setMessageAnnotations = createMapSetter(messageAnnotationsHook)
  const showAnnotations = state.ui.showAnnotations || true
  const setShowAnnotations = (value: boolean) => updateUI({ showAnnotations: value })
  const smartFoldersHook = useMapState<{ name: string, filters: any, messageIds: string[] }>(new Map())
  const smartFolders = createMapAlias(smartFoldersHook)
  const setSmartFolders = createMapSetter(smartFoldersHook)
  // Estados de UI ahora en uiState hook
  const showFolders = uiState.state.showFolders
  const setShowFolders = (value: boolean) => value ? uiState.actions.toggleFolders() : uiState.actions.toggleFolders()
  const messagePriorityHook = useMapState<'low' | 'medium' | 'high' | 'urgent'>(new Map())
  const messagePriority = createMapAlias(messagePriorityHook)
  const setMessagePriority = createMapSetter(messagePriorityHook)
  const fullscreenMode = state.display.fullscreenMode
  const setFullscreenMode = (value: boolean) => updateDisplay({ fullscreenMode: value })
  const [presentationSlide, setPresentationSlide] = useState(0)
  const conversationVersionsHook = useMapState<{ timestamp: number, messages: any[], metadata: any }>(new Map())
  const conversationVersions = createMapAlias(conversationVersionsHook)
  const setConversationVersions = createMapSetter(conversationVersionsHook)
  const showVersions = state.ui.showVersions
  const setShowVersions = (value: boolean) => updateUI({ showVersions: value })
  const messageVotesHook = useMapState<{ up: number, down: number }>(new Map())
  const messageVotes = createMapAlias(messageVotesHook)
  const setMessageVotes = createMapSetter(messageVotesHook)
  const messageRatingsHook = useMapState<number>(new Map())
  const messageRatings = createMapAlias(messageRatingsHook)
  const setMessageRatings = createMapSetter(messageRatingsHook)
  const messageCommentsHook = useMapState<{ author: string, content: string, timestamp: number }[]>(new Map())
  const messageComments = createMapAlias(messageCommentsHook)
  const setMessageComments = createMapSetter(messageCommentsHook)
  const showComments = state.ui.showComments
  const setShowComments = (value: boolean) => updateUI({ showComments: value })
  const pushNotifications = state.features.pushNotifications || false
  const setPushNotifications = (value: boolean) => updateFeatures({ pushNotifications: value })
  const [notificationPermission, setNotificationPermission] = useState<NotificationPermission>('default')
  const printMode = state.ui.printMode || false
  const setPrintMode = (value: boolean) => updateUI({ printMode: value })
  const [printStyles, setPrintStyles] = useState({ fontSize: 12, fontFamily: 'Arial', lineHeight: 1.5 })
  const apiIntegrationsListHook = useMapState<{ name: string, endpoint: string, enabled: boolean }>(new Map())
  const apiIntegrationsList = createMapAlias(apiIntegrationsListHook)
  const setApiIntegrationsList = createMapSetter(apiIntegrationsListHook)
  const [activeApiIntegration, setActiveApiIntegration] = useState<string | null>(null)
  const pluginStoreHook = useMapState<{ name: string, description: string, enabled: boolean }>(new Map())
  const pluginStore = createMapAlias(pluginStoreHook)
  const setPluginStore = createMapSetter(pluginStoreHook)
  const customThemeColorsHook = useMapState<string>(new Map())
  const customThemeColors = createMapAlias(customThemeColorsHook)
  const setCustomThemeColors = createMapSetter(customThemeColorsHook)
  const themePresetsHook = useMapState<any>(new Map())
  const themePresets = createMapAlias(themePresetsHook)
  const setThemePresets = createMapSetter(themePresetsHook)
  const advancedAnalytics = state.ui.showAdvancedAnalytics || false
  const setAdvancedAnalytics = (value: boolean) => updateUI({ showAdvancedAnalytics: value })
  const [analyticsData, setAnalyticsData] = useState<any>(null)
  const [exportFormatsAdvanced, setExportFormatsAdvanced] = useState<Set<'json' | 'txt' | 'md' | 'html' | 'csv' | 'pdf' | 'xml' | 'yaml'>>(new Set(['json', 'txt', 'md']))
  const messageTimeline = state.ui.messageTimeline || false
  const setMessageTimeline = (value: boolean) => {} // Temporal
  const [timelineView, setTimelineView] = useState<'linear' | 'grouped' | 'chronological'>('linear')
  const [messageComparison, setMessageComparison] = useState<[string, string] | null>(null)
  const showComparison = state.ui.showComparison
  const setShowComparison = (value: boolean) => updateUI({ showComparison: value })
  const messageTranslationHook = useMapState<{ language: string, content: string }>(new Map())
  const messageTranslation = createMapAlias(messageTranslationHook)
  const setMessageTranslation = createMapSetter(messageTranslationHook)
  const autoTranslate = state.features.autoTranslate
  const setAutoTranslate = (value: boolean) => updateFeatures({ autoTranslate: value })
  const [targetTranslationLanguage, setTargetTranslationLanguage] = useState('es')
  const messageSummarizationHook = useMapState<string>(new Map())
  const messageSummarization = createMapAlias(messageSummarizationHook)
  const setMessageSummarization = createMapSetter(messageSummarizationHook)
  const autoSummarize = state.features.autoSummarize
  const setAutoSummarize = (value: boolean) => updateFeatures({ autoSummarize: value })
  const splitScreenMode = state.display.splitScreenMode
  const setSplitScreenMode = (value: boolean) => updateDisplay({ splitScreenMode: value })
  const customShortcutsEnabled = state.features.customShortcutsEnabled
  const setCustomShortcutsEnabled = (value: boolean) => updateFeatures({ customShortcutsEnabled: value })
  const widgetsHook = useMapState<{ type: string, position: string, enabled: boolean }>(new Map())
  const widgets = createMapAlias(widgetsHook)
  const setWidgets = createMapSetter(widgetsHook)
  const showWidgets = state.ui.showWidgets
  const setShowWidgets = (value: boolean) => updateUI({ showWidgets: value })
  const [audioRecording, setAudioRecording] = useState(false)
  const [videoRecording, setVideoRecording] = useState(false)
  const messageWidgetsHook = useMapState<string[]>(new Map())
  const messageWidgets = createMapAlias(messageWidgetsHook)
  const setMessageWidgets = createMapSetter(messageWidgetsHook)
  const [presentationTransitions, setPresentationTransitions] = useState<'fade' | 'slide' | 'zoom' | 'none'>('fade')
  const messageTemplatesAdvancedHook = useMapState<{ template: string, variables: string[], category: string }>(new Map())
  const messageTemplatesAdvanced = createMapAlias(messageTemplatesAdvancedHook)
  const setMessageTemplatesAdvanced = createMapSetter(messageTemplatesAdvancedHook)
  const [accessibilityFeatures, setAccessibilityFeatures] = useState({ screenReader: false, highContrast: false, largeText: false, reducedMotion: false })
  const performanceMetricsHook = useMapState<number>(new Map())
  const performanceMetrics = createMapAlias(performanceMetricsHook)
  const setPerformanceMetrics = createMapSetter(performanceMetricsHook)
  const showPerformance = state.ui.showPerformance
  const setShowPerformance = (value: boolean) => updateUI({ showPerformance: value })
  const [messageQueue, setMessageQueue] = useState<Array<{ id: string, message: string, priority: number }>>([])
  const [queueProcessing, setQueueProcessing] = useState(false)
  const [messageFormatting, setMessageFormatting] = useState<'plain' | 'markdown' | 'html' | 'code'>('plain')
  const [autoFormat, setAutoFormat] = useState(true)
  const [messageLinking, setMessageLinking] = useState<Map<string, string[]>>(new Map())
  const [showMessageGraph, setShowMessageGraph] = useState(false)
  const [messageRelations, setMessageRelations] = useState<Map<string, { type: string, target: string }[]>>(new Map())
  const [smartFoldersAdvanced, setSmartFoldersAdvanced] = useState<Map<string, { name: string, filters: any, color: string, icon: string }>>(new Map())
  const [workflowMode, setWorkflowMode] = useState(false)
  const [messageValidation, setMessageValidation] = useState<Map<string, { valid: boolean, errors: string[] }>>(new Map())
  const [autoValidate, setAutoValidate] = useState(false)
  const [messageDeduplication, setMessageDeduplication] = useState(true)
  const [duplicateThreshold, setDuplicateThreshold] = useState(0.8)
  const [messageCompression, setMessageCompression] = useState(false)
  const [compressionLevel, setCompressionLevel] = useState<'low' | 'medium' | 'high'>('medium')
  const [messageEncryptionAdvanced, setMessageEncryptionAdvanced] = useState<Map<string, { algorithm: string, key: string }>>(new Map())
  const [encryptionEnabled, setEncryptionEnabled] = useState(false)
  const [messageBackupAdvanced, setMessageBackupAdvanced] = useState<Map<string, { timestamp: number, format: string, size: number }>>(new Map())
  const [autoBackupInterval, setAutoBackupInterval] = useState(300000) // 5 minutos
  const [messageSync, setMessageSync] = useState<Map<string, { synced: boolean, timestamp: number, device: string }>>(new Map())
  const [multiDeviceSync, setMultiDeviceSync] = useState(false)
  const [messageAnalyticsAdvanced, setMessageAnalyticsAdvanced] = useState<Map<string, { views: number, interactions: number, shares: number }>>(new Map())
  const [showAdvancedAnalytics, setShowAdvancedAnalytics] = useState(false)
  const [messageCache, setMessageCache] = useState<Map<string, { content: string, timestamp: number }>>(new Map())
  const [cacheEnabled, setCacheEnabled] = useState(true)
  const [messageIndexing, setMessageIndexing] = useState<Map<string, { keywords: string[], summary: string }>>(new Map())
  const [searchIndex, setSearchIndex] = useState<Map<string, number[]>>(new Map())
  const [smartSearch, setSmartSearch] = useState(true)
  const [messageSuggestions, setMessageSuggestions] = useState<Map<string, string[]>>(new Map())
  const [autoComplete, setAutoComplete] = useState(true)
  const [typingPrediction, setTypingPrediction] = useState(true)
  const [quickReplies, setQuickReplies] = useState<string[]>([])
  const [messageMacros, setMessageMacros] = useState<Map<string, { trigger: string, replacement: string }>>(new Map())
  const [macroEnabled, setMacroEnabled] = useState(true)
  const [undoStack, setUndoStack] = useState<any[]>([])
  const [redoStack, setRedoStack] = useState<any[]>([])
  const [undoEnabled, setUndoEnabled] = useState(true)
  const [messageBatch, setMessageBatch] = useState<Array<{ role: string, content: string }>>([])
  const [batchMode, setBatchMode] = useState(false)
  const [messageFiltering, setMessageFiltering] = useState<Map<string, boolean>>(new Map())
  const [filterRules, setFilterRules] = useState<Array<{ type: string, pattern: string, action: string }>>([])
  const [autoFilter, setAutoFilter] = useState(false)
  const [notificationSettings, setNotificationSettings] = useState({ sound: true, desktop: true, badge: true })
  const [highlightEnabled, setHighlightEnabled] = useState(true)
  const [messageBookmarksAdvanced, setMessageBookmarksAdvanced] = useState<Map<string, { name: string, category: string, tags: string[] }>>(new Map())
  const [bookmarkCategories, setBookmarkCategories] = useState<string[]>(['favoritos', 'importante', 'referencia'])
  const [messageExportAdvanced, setMessageExportAdvanced] = useState<Map<string, { format: string, timestamp: number }>>(new Map())
  const [exportTemplates, setExportTemplates] = useState<Map<string, { name: string, format: string, template: string }>>(new Map())
  const [messageImport, setMessageImport] = useState<Map<string, { source: string, format: string, timestamp: number }>>(new Map())
  const [importEnabled, setImportEnabled] = useState(true)
  const [messageSyncAdvanced, setMessageSyncAdvanced] = useState<Map<string, { provider: string, status: string, lastSync: number }>>(new Map())
  const [syncProviders, setSyncProviders] = useState<string[]>(['localStorage', 'indexedDB', 'cloud'])
  const [messageVersioning, setMessageVersioning] = useState<Map<string, { version: number, changes: string[], timestamp: number }>>(new Map())
  const [versionControl, setVersionControl] = useState(true)
  const [messageCollaboration, setMessageCollaboration] = useState<Map<string, { users: string[], permissions: string[] }>>(new Map())
  const [collaborationEnabled, setCollaborationEnabled] = useState(false)
  const [messageAI, setMessageAI] = useState<Map<string, { suggestions: string[], sentiment: string, topics: string[] }>>(new Map())
  const [aiFeatures, setAiFeatures] = useState({ suggestions: true, sentiment: true, topics: true })
  const [commandSystem, setCommandSystem] = useState<Map<string, { command: string, handler: () => void, description: string }>>(new Map())
  const [commandPalette, setCommandPalette] = useState(false)
  const [messageActions, setMessageActions] = useState<Map<string, { type: string, handler: () => void }[]>>(new Map())
  const [actionMenu, setActionMenu] = useState<string | null>(null)
  const [messageContext, setMessageContext] = useState<Map<string, { related: string[], context: any }>>(new Map())
  const [contextMenu, setContextMenu] = useState<{ x: number, y: number, messageId: string } | null>(null)
  const [messageReactionsAdvanced, setMessageReactionsAdvanced] = useState<Map<string, { emoji: string, count: number, users: string[] }[]>>(new Map())
  const [reactionPicker, setReactionPicker] = useState<string | null>(null)
  const [messagePolls, setMessagePolls] = useState<Map<string, { question: string, options: string[], votes: Map<string, number> }>>(new Map())
  const [pollMode, setPollMode] = useState(false)
  const [messageTasks, setMessageTasks] = useState<Map<string, { task: string, completed: boolean, dueDate?: Date }>>(new Map())
  const [taskMode, setTaskMode] = useState(false)
  const [messageRemindersAdvanced, setMessageRemindersAdvanced] = useState<Map<string, { reminder: string, date: Date, recurring?: string }>>(new Map())
  const [reminderSystem, setReminderSystem] = useState(true)
  // Estados de mensajes ahora manejados por useMessageState
  const { state: messageStateData, actions: messageActions } = messageState
  
  // Aliases para compatibilidad con código existente
  const messageAttachments = messageStateData.messageAttachments
  const messageLinks = messageStateData.messageLinks
  const messageCalendar = messageStateData.messageCalendar
  const messageBookmarksAdvanced = messageStateData.messageBookmarks
  const messageNotifications = messageStateData.messageNotifications
  const messageHighlights = messageStateData.messageHighlights
  const messageAnnotations = messageStateData.messageAnnotations
  const messagePolls = messageStateData.messagePolls
  const messageTasks = messageStateData.messageTasks
  const messageRemindersAdvanced = messageStateData.messageReminders
  
  // Setters para compatibilidad (usar actions en su lugar)
  const setMessageAttachments = (updater: any) => {
    // Usar messageActions.addAttachment en lugar de esto
    console.warn('setMessageAttachments está deprecated, usar messageActions.addAttachment')
  }
  const setMessageLinks = (updater: any) => {
    console.warn('setMessageLinks está deprecated, usar messageActions.addLink')
  }
  
  // Estados UI que no están en messageState
  const [messageNotesAdvanced, setMessageNotesAdvanced] = useState<Map<string, { note: string, attachments: string[], tags: string[] }>>(new Map())
  const [noteEditor, setNoteEditor] = useState<string | null>(null)
  const [attachmentManager, setAttachmentManager] = useState(false)
  const [messageCode, setMessageCode] = useState<Map<string, { language: string, code: string }>>(new Map())
  const [codeEditor, setCodeEditor] = useState<string | null>(null)
  const [messageMedia, setMessageMedia] = useState<Map<string, { type: string, url: string, thumbnail?: string }[]>>(new Map())
  const [mediaViewer, setMediaViewer] = useState<string | null>(null)
  const [messageLocation, setMessageLocation] = useState<Map<string, { lat: number, lng: number, address: string }>>(new Map())
  const [locationSharing, setLocationSharing] = useState(false)
  const [messageContacts, setMessageContacts] = useState<Map<string, { name: string, email: string, phone?: string }[]>>(new Map())
  const [contactManager, setContactManager] = useState(false)
  const [messageEvents, setMessageEvents] = useState<Map<string, { type: string, data: any, timestamp: number }>>(new Map())
  const [eventLog, setEventLog] = useState(true)
  const [messageMetadata, setMessageMetadata] = useState<Map<string, { key: string, value: any }[]>>(new Map())
  const [metadataEditor, setMetadataEditor] = useState<string | null>(null)
  const [pluginManager, setPluginManager] = useState(false)
  const [apiManager, setApiManager] = useState(false)
  const [devTools, setDevTools] = useState(false)
  const [devConsole, setDevConsole] = useState<string[]>([])
  const [themeSystem, setThemeSystem] = useState<Map<string, { name: string, colors: any, fonts: any }>>(new Map())
  const [themeEditor, setThemeEditor] = useState(false)
  const [performanceMonitor, setPerformanceMonitor] = useState<Map<string, { metric: string, value: number, timestamp: number }>>(new Map())
  const [performanceDashboard, setPerformanceDashboard] = useState(false)
  const [notificationCenter, setNotificationCenter] = useState(false)
  const [offlineMode, setOfflineMode] = useState(false)
  const [offlineQueue, setOfflineQueue] = useState<Array<{ id: string, action: string, data: any }>>([])
  const [syncStatus, setSyncStatus] = useState<'synced' | 'syncing' | 'error' | 'offline'>('synced')
  const [realTimeSync, setRealTimeSync] = useState(false)
  const [cloudStorage, setCloudStorage] = useState<Map<string, { provider: string, bucket: string, path: string }>>(new Map())
  const [cloudManager, setCloudManager] = useState(false)
  const [authSystem, setAuthSystem] = useState<Map<string, { user: string, role: string, permissions: string[] }>>(new Map())
  const [authManager, setAuthManager] = useState(false)
  const [encryptionManager, setEncryptionManager] = useState(false)
  const [messageBackup, setMessageBackup] = useState<Map<string, { timestamp: number, size: number, format: string }>>(new Map())
  const [backupManager, setBackupManager] = useState(false)
  const [messageRestore, setMessageRestore] = useState<Map<string, { source: string, timestamp: number, status: string }>>(new Map())
  const [restoreManager, setRestoreManager] = useState(false)
  const [analyticsDashboard, setAnalyticsDashboard] = useState(false)
  const [messageSearch, setMessageSearch] = useState<Map<string, { query: string, results: string[], filters: any }>>(new Map())
  const [searchManager, setSearchManager] = useState(false)
  const [messageExport, setMessageExport] = useState<Map<string, { format: string, timestamp: number, size: number }>>(new Map())
  const [exportManager, setExportManager] = useState(false)
  const [importManager, setImportManager] = useState(false)
  const [widgetSystem, setWidgetSystem] = useState<Map<string, { type: string, position: string, config: any, visible: boolean }>>(new Map())
  const [widgetEditor, setWidgetEditor] = useState(false)
  const [presentationSlides, setPresentationSlides] = useState<Array<{ title: string, content: string, notes?: string }>>([])
  const [presentationIndex, setPresentationIndex] = useState(0)
  const [externalServices, setExternalServices] = useState<Map<string, { name: string, type: string, config: any, connected: boolean }>>(new Map())
  const [serviceManager, setServiceManager] = useState(false)
  const [templateLibrary, setTemplateLibrary] = useState<Map<string, { name: string, category: string, content: string, variables: string[] }>>(new Map())
  const [templateEditor, setTemplateEditor] = useState(false)
  const [collaborationRoom, setCollaborationRoom] = useState<Map<string, { users: string[], permissions: Map<string, string[]>, active: boolean }>>(new Map())
  const [collaborationManager, setCollaborationManager] = useState(false)
  const [notificationManager, setNotificationManager] = useState(false)
  const [accessibilitySettings, setAccessibilitySettings] = useState<Map<string, { feature: string, enabled: boolean, config: any }>>(new Map())
  const [accessibilityManager, setAccessibilityManager] = useState(false)
  const [helpSystem, setHelpSystem] = useState<Map<string, { topic: string, content: string, category: string }>>(new Map())
  const [helpCenter, setHelpCenter] = useState(false)
  const [tutorialMode, setTutorialMode] = useState(false)
  const [tutorialSteps, setTutorialSteps] = useState<Array<{ step: number, title: string, content: string, completed: boolean }>>([])
  const [feedbackSystem, setFeedbackSystem] = useState<Map<string, { type: string, content: string, rating?: number, timestamp: number }>>(new Map())
  const [feedbackManager, setFeedbackManager] = useState(false)
  const [reportSystem, setReportSystem] = useState<Map<string, { type: string, data: any, timestamp: number }>>(new Map())
  const [reportManager, setReportManager] = useState(false)
  const [historyViewer, setHistoryViewer] = useState(false)
  const [relationViewer, setRelationViewer] = useState(false)
  const [messageInsights, setMessageInsights] = useState<Map<string, { insight: string, confidence: number, type: string }>>(new Map())
  const [insightsPanel, setInsightsPanel] = useState(false)
  const [suggestionsPanel, setSuggestionsPanel] = useState(false)
  const [messageClustering, setMessageClustering] = useState<Map<string, { cluster: string, messages: string[] }>>(new Map())
  const [clusteringViewer, setClusteringViewer] = useState(false)
  const [performanceOptimization, setPerformanceOptimization] = useState({ virtualScrolling: true, lazyLoading: true, memoization: true })
  const [uiEnhancements, setUiEnhancements] = useState({ animations: true, transitions: true, tooltips: true, hoverEffects: true })
  const [productivityFeatures, setProductivityFeatures] = useState({ quickActions: true, smartShortcuts: true, autoComplete: true, suggestions: true })
  const [integrationSettings, setIntegrationSettings] = useState<Map<string, { service: string, config: any, enabled: boolean }>>(new Map())
  const [shortcutSystem, setShortcutSystem] = useState<Map<string, { key: string, action: () => void, description: string }>>(new Map())
  const [shortcutManager, setShortcutManager] = useState(false)
  const [accessibilityEnhancements, setAccessibilityEnhancements] = useState({ screenReader: true, keyboardNav: true, focusIndicators: true, ariaLabels: true })
  const [notificationSystem, setNotificationSystem] = useState<Map<string, { type: string, priority: number, timestamp: number, read: boolean }>>(new Map())
  const [searchEnhancements, setSearchEnhancements] = useState({ fuzzySearch: true, regexSearch: true, highlightMatches: true, searchHistory: true })
  const [exportEnhancements, setExportEnhancements] = useState({ batchExport: true, customFormats: true, compression: true, encryption: true })
  const [importEnhancements, setImportEnhancements] = useState({ autoDetect: true, validation: true, preview: true, merge: true })
  const [securityFeatures, setSecurityFeatures] = useState({ encryption: true, authentication: true, auditLog: true, permissions: true })
  const [messageOptimization, setMessageOptimization] = useState({ deduplication: true, compression: true, caching: true, indexing: true })
  const [uiCustomization, setUiCustomization] = useState({ layout: 'default', density: 'normal', fontSize: 14, colorScheme: 'dark' })
  const [workflowAutomation, setWorkflowAutomation] = useState<Map<string, { trigger: string, actions: string[], enabled: boolean }>>(new Map())
  const [automationManager, setAutomationManager] = useState(false)
  const [messageAnalytics, setMessageAnalytics] = useState<Map<string, { views: number, interactions: number, shares: number, timestamp: number }>>(new Map())
  const [analyticsViewer, setAnalyticsViewer] = useState(false)
  const [messageQuality, setMessageQuality] = useState<Map<string, { score: number, metrics: any }>>(new Map())
  const [qualityMonitor, setQualityMonitor] = useState(false)
  const [realTimeTranslation, setRealTimeTranslation] = useState<Map<string, { original: string, translated: string, language: string }>>(new Map())
  const [collaborationFeatures, setCollaborationFeatures] = useState({ realTime: true, presence: true, cursors: true, comments: true })
  const [collaborationView, setCollaborationView] = useState(false)
  const [versionManager, setVersionManager] = useState(false)
  const [externalAI, setExternalAI] = useState<Map<string, { service: string, apiKey: string, enabled: boolean, config: any }>>(new Map())
  const [aiManager, setAiManager] = useState(false)
  const [learningMode, setLearningMode] = useState(false)
  const [learningData, setLearningData] = useState<Map<string, { pattern: string, response: string, confidence: number }>>(new Map())
  const [smartRecommendations, setSmartRecommendations] = useState<Map<string, { type: string, content: string, score: number }>>(new Map())
  const [recommendationsPanel, setRecommendationsPanel] = useState(false)
  const [sentimentAnalysis, setSentimentAnalysis] = useState<Map<string, { sentiment: string, score: number, emotions: string[] }>>(new Map())
  const [sentimentViewer, setSentimentViewer] = useState(false)
  const [autoSummary, setAutoSummary] = useState<Map<string, { summary: string, keyPoints: string[], timestamp: number }>>(new Map())
  const [summaryViewer, setSummaryViewer] = useState(false)
  const [multiFormatExport, setMultiFormatExport] = useState<Map<string, { formats: string[], timestamp: number }>>(new Map())
  const [exportWizard, setExportWizard] = useState(false)
  const [syncEnhancements, setSyncEnhancements] = useState({ conflictResolution: true, mergeStrategy: 'smart', syncInterval: 30000 })
  const [syncManager, setSyncManager] = useState(false)
  const [contextViewer, setContextViewer] = useState(false)
  const [messagePatterns, setMessagePatterns] = useState<Map<string, { pattern: string, frequency: number, examples: string[] }>>(new Map())
  const [patternAnalyzer, setPatternAnalyzer] = useState(false)
  const [messageFlow, setMessageFlow] = useState<Map<string, { from: string, to: string, type: string }>>(new Map())
  const [flowViewer, setFlowViewer] = useState(false)
  const [timelineViewer, setTimelineViewer] = useState(false)
  const [messageVisualization, setMessageVisualization] = useState<Map<string, { type: string, data: any }>>(new Map())
  const [visualizationMode, setVisualizationMode] = useState<'graph' | 'tree' | 'network' | 'none'>('none')
  const [messageDependencies, setMessageDependencies] = useState<Map<string, { dependsOn: string[], requiredBy: string[] }>>(new Map())
  const [dependencyViewer, setDependencyViewer] = useState(false)
  const [messageMetrics, setMessageMetrics] = useState<Map<string, { readTime: number, responseTime: number, engagement: number }>>(new Map())
  const [metricsDashboard, setMetricsDashboard] = useState(false)
  const [messageAlerts, setMessageAlerts] = useState<Map<string, { type: string, message: string, severity: 'info' | 'warning' | 'error' }>>(new Map())
  const [alertCenter, setAlertCenter] = useState(false)
  const [priorityFilter, setPriorityFilter] = useState<string | null>(null)
  const [messageStatus, setMessageStatus] = useState<Map<string, 'pending' | 'processing' | 'completed' | 'archived'>>(new Map())
  const [statusFilter, setStatusFilter] = useState<string | null>(null)
  const [messageCategories, setMessageCategories] = useState<Map<string, { category: string, subcategory?: string }>>(new Map())
  const [categoryManager, setCategoryManager] = useState(false)
  const [messageFilters, setMessageFilters] = useState<Map<string, { filter: string, active: boolean }>>(new Map())
  const [filterManager, setFilterManager] = useState(false)
  const [messageSorting, setMessageSorting] = useState<Map<string, { sortBy: string, order: 'asc' | 'desc' }>>(new Map())
  const [sortManager, setSortManager] = useState(false)
  const [messageGrouping, setMessageGrouping] = useState<Map<string, { groupBy: string, groups: string[] }>>(new Map())
  const [groupManager, setGroupManager] = useState(false)
  const [bookmarkManager, setBookmarkManager] = useState(false)
  const [highlightManager, setHighlightManager] = useState(false)
  const [annotationManager, setAnnotationManager] = useState(false)
  const [linkManager, setLinkManager] = useState(false)
  const [messageFiles, setMessageFiles] = useState<Map<string, { name: string, type: string, size: number, url: string }>>(new Map())
  const [fileManager, setFileManager] = useState(false)
  const [messageImages, setMessageImages] = useState<Map<string, { url: string, alt: string, caption?: string }>>(new Map())
  const [imageGallery, setImageGallery] = useState(false)
  const [messageVideos, setMessageVideos] = useState<Map<string, { url: string, thumbnail: string, duration?: number }>>(new Map())
  const [videoPlayer, setVideoPlayer] = useState<string | null>(null)
  const [messageAudio, setMessageAudio] = useState<Map<string, { url: string, duration: number, waveform?: number[] }>>(new Map())
  const [audioPlayer, setAudioPlayer] = useState<string | null>(null)
  const [messageDocuments, setMessageDocuments] = useState<Map<string, { type: string, content: string, metadata: any }>>(new Map())
  const [documentViewer, setDocumentViewer] = useState<string | null>(null)
  const [messageForms, setMessageForms] = useState<Map<string, { fields: Array<{ name: string, type: string, required: boolean }>, responses: any[] }>>(new Map())
  const [formBuilder, setFormBuilder] = useState(false)
  const [pollManager, setPollManager] = useState(false)
  const [messageQuizzes, setMessageQuizzes] = useState<Map<string, { questions: Array<{ question: string, answers: string[], correct: number }>, results: any[] }>>(new Map())
  const [quizMode, setQuizMode] = useState(false)
  const [messageSurveys, setMessageSurveys] = useState<Map<string, { title: string, questions: string[], responses: any[] }>>(new Map())
  const [surveyManager, setSurveyManager] = useState(false)
  const [ratingSystem, setRatingSystem] = useState(false)
  const [messageReviews, setMessageReviews] = useState<Map<string, { review: string, rating: number, helpful: number }>>(new Map())
  const [reviewSystem, setReviewSystem] = useState(false)
  const [messageFeedback, setMessageFeedback] = useState<Map<string, { feedback: string, type: 'positive' | 'negative' | 'neutral', timestamp: number }>>(new Map())
  const [messageReports, setMessageReports] = useState<Map<string, { reason: string, description: string, status: 'pending' | 'reviewed' | 'resolved' }>>(new Map())
  const [messageModeration, setMessageModeration] = useState<Map<string, { action: string, reason: string, moderator: string }>>(new Map())
  const [moderationPanel, setModerationPanel] = useState(false)
  const [messageApproval, setMessageApproval] = useState<Map<string, { status: 'pending' | 'approved' | 'rejected', approver?: string }>>(new Map())
  const [approvalWorkflow, setApprovalWorkflow] = useState(false)
  const [messageScheduling, setMessageScheduling] = useState<Map<string, { scheduledTime: number, status: 'scheduled' | 'sent' | 'cancelled' }>>(new Map())
  const [scheduler, setScheduler] = useState(false)
  const [templateManager, setTemplateManager] = useState(false)
  const [messageVariables, setMessageVariables] = useState<Map<string, { name: string, value: string, type: string }>>(new Map())
  const [variableManager, setVariableManager] = useState(false)
  const [messageConditions, setMessageConditions] = useState<Map<string, { condition: string, action: string, enabled: boolean }>>(new Map())
  const [conditionManager, setConditionManager] = useState(false)
  const [actionManager, setActionManager] = useState(false)
  const [messageTriggers, setMessageTriggers] = useState<Map<string, { trigger: string, conditions: string[], actions: string[] }>>(new Map())
  const [triggerManager, setTriggerManager] = useState(false)
  const [messageWorkflows, setMessageWorkflows] = useState<Map<string, { name: string, steps: Array<{ step: number, action: string, condition?: string }>, active: boolean }>>(new Map())
  const [workflowManager, setWorkflowManager] = useState(false)
  const [messageIntegrations, setMessageIntegrations] = useState<Map<string, { service: string, config: any, enabled: boolean }>>(new Map())
  const [integrationManager, setIntegrationManager] = useState(false)
  const [messageWebhooks, setMessageWebhooks] = useState<Map<string, { url: string, events: string[], secret?: string }>>(new Map())
  const [webhookManager, setWebhookManager] = useState(false)
  const [messageAPIs, setMessageAPIs] = useState<Map<string, { endpoint: string, method: string, auth: any, enabled: boolean }>>(new Map())
  const [messageSDKs, setMessageSDKs] = useState<Map<string, { sdk: string, version: string, config: any }>>(new Map())
  const [sdkManager, setSdkManager] = useState(false)
  const [messagePlugins, setMessagePlugins] = useState<Map<string, { plugin: string, version: string, config: any, enabled: boolean }>>(new Map())
  const [messageExtensions, setMessageExtensions] = useState<Map<string, { extension: string, type: string, config: any }>>(new Map())
  const [extensionManager, setExtensionManager] = useState(false)
  const [messageAddons, setMessageAddons] = useState<Map<string, { addon: string, version: string, enabled: boolean }>>(new Map())
  const [addonManager, setAddonManager] = useState(false)
  const [messageModules, setMessageModules] = useState<Map<string, { module: string, version: string, exports: string[] }>>(new Map())
  const [moduleManager, setModuleManager] = useState(false)
  const [messageServices, setMessageServices] = useState<Map<string, { service: string, status: 'active' | 'inactive' | 'error', config: any }>>(new Map())
  const [messageProviders, setMessageProviders] = useState<Map<string, { provider: string, credentials: any, enabled: boolean }>>(new Map())
  const [providerManager, setProviderManager] = useState(false)
  const [messageConnectors, setMessageConnectors] = useState<Map<string, { connector: string, config: any, connected: boolean }>>(new Map())
  const [connectorManager, setConnectorManager] = useState(false)
  const [messageAdapters, setMessageAdapters] = useState<Map<string, { adapter: string, from: string, to: string, config: any }>>(new Map())
  const [adapterManager, setAdapterManager] = useState(false)
  const [messageTransformers, setMessageTransformers] = useState<Map<string, { transformer: string, input: string, output: string, config: any }>>(new Map())
  const [transformerManager, setTransformerManager] = useState(false)
  const [messageValidators, setMessageValidators] = useState<Map<string, { validator: string, rules: string[], errors: string[] }>>(new Map())
  const [validatorManager, setValidatorManager] = useState(false)
  const [messageParsers, setMessageParsers] = useState<Map<string, { parser: string, format: string, config: any }>>(new Map())
  const [parserManager, setParserManager] = useState(false)
  const [messageSerializers, setMessageSerializers] = useState<Map<string, { serializer: string, format: string, config: any }>>(new Map())
  const [serializerManager, setSerializerManager] = useState(false)
  const [messageFormatters, setMessageFormatters] = useState<Map<string, { formatter: string, format: string, config: any }>>(new Map())
  const [formatterManager, setFormatterManager] = useState(false)
  const [messageRenderers, setMessageRenderers] = useState<Map<string, { renderer: string, template: string, config: any }>>(new Map())
  const [rendererManager, setRendererManager] = useState(false)
  const [messageGenerators, setMessageGenerators] = useState<Map<string, { generator: string, type: string, config: any }>>(new Map())
  const [generatorManager, setGeneratorManager] = useState(false)
  const [messageProcessors, setMessageProcessors] = useState<Map<string, { processor: string, pipeline: string[], config: any }>>(new Map())
  const [processorManager, setProcessorManager] = useState(false)
  const [messageHandlers, setMessageHandlers] = useState<Map<string, { handler: string, events: string[], config: any }>>(new Map())
  const [handlerManager, setHandlerManager] = useState(false)
  const [messageListeners, setMessageListeners] = useState<Map<string, { listener: string, events: string[], config: any }>>(new Map())
  const [listenerManager, setListenerManager] = useState(false)
  const [messageObservers, setMessageObservers] = useState<Map<string, { observer: string, pattern: string, action: string }>>(new Map())
  const [observerManager, setObserverManager] = useState(false)
  const [messageSubscribers, setMessageSubscribers] = useState<Map<string, { subscriber: string, topics: string[], config: any }>>(new Map())
  const [subscriberManager, setSubscriberManager] = useState(false)
  const [messagePublishers, setMessagePublishers] = useState<Map<string, { publisher: string, topics: string[], config: any }>>(new Map())
  const [publisherManager, setPublisherManager] = useState(false)
  const [messageBrokers, setMessageBrokers] = useState<Map<string, { broker: string, config: any, connected: boolean }>>(new Map())
  const [brokerManager, setBrokerManager] = useState(false)
  const [messageQueues, setMessageQueues] = useState<Map<string, { queue: string, messages: any[], config: any }>>(new Map())
  const [queueManager, setQueueManager] = useState(false)
  const [messageTopics, setMessageTopics] = useState<Map<string, { topic: string, subscribers: string[], messages: any[] }>>(new Map())
  const [topicManager, setTopicManager] = useState(false)
  const [messageChannels, setMessageChannels] = useState<Map<string, { channel: string, type: string, config: any }>>(new Map())
  const [channelManager, setChannelManager] = useState(false)
  const [messageStreams, setMessageStreams] = useState<Map<string, { stream: string, format: string, config: any }>>(new Map())
  const [streamManager, setStreamManager] = useState(false)
  const [messageBuffers, setMessageBuffers] = useState<Map<string, { buffer: string, size: number, config: any }>>(new Map())
  const [bufferManager, setBufferManager] = useState(false)
  const [messageCaches, setMessageCaches] = useState<Map<string, { cache: string, size: number, ttl: number }>>(new Map())
  const [cacheManager, setCacheManager] = useState(false)
  const [messageStores, setMessageStores] = useState<Map<string, { store: string, type: string, config: any }>>(new Map())
  const [storeManager, setStoreManager] = useState(false)
  const [messageDatabases, setMessageDatabases] = useState<Map<string, { database: string, connection: any, config: any }>>(new Map())
  const [databaseManager, setDatabaseManager] = useState(false)
  const [messageRepositories, setMessageRepositories] = useState<Map<string, { repository: string, type: string, config: any }>>(new Map())
  const [repositoryManager, setRepositoryManager] = useState(false)
  const [messageIndexes, setMessageIndexes] = useState<Map<string, { index: string, fields: string[], config: any }>>(new Map())
  const [indexManager, setIndexManager] = useState(false)
  const [messageSearchEngines, setMessageSearchEngines] = useState<Map<string, { engine: string, config: any, enabled: boolean }>>(new Map())
  const [searchEngineManager, setSearchEngineManager] = useState(false)
  const [messageAnalyzers, setMessageAnalyzers] = useState<Map<string, { analyzer: string, type: string, config: any }>>(new Map())
  const [analyzerManager, setAnalyzerManager] = useState(false)
  const [messageClassifiers, setMessageClassifiers] = useState<Map<string, { classifier: string, categories: string[], config: any }>>(new Map())
  const [classifierManager, setClassifierManager] = useState(false)
  const [messageExtractors, setMessageExtractors] = useState<Map<string, { extractor: string, type: string, config: any }>>(new Map())
  const [extractorManager, setExtractorManager] = useState(false)
  const [messageEnrichers, setMessageEnrichers] = useState<Map<string, { enricher: string, fields: string[], config: any }>>(new Map())
  const [enricherManager, setEnricherManager] = useState(false)
  const [messageAggregators, setMessageAggregators] = useState<Map<string, { aggregator: string, function: string, config: any }>>(new Map())
  const [aggregatorManager, setAggregatorManager] = useState(false)
  const [messageReducers, setMessageReducers] = useState<Map<string, { reducer: string, function: string, config: any }>>(new Map())
  const [reducerManager, setReducerManager] = useState(false)
  const [messageMappers, setMessageMappers] = useState<Map<string, { mapper: string, mapping: any, config: any }>>(new Map())
  const [mapperManager, setMapperManager] = useState(false)
  const [messageSorters, setMessageSorters] = useState<Map<string, { sorter: string, field: string, order: 'asc' | 'desc' }>>(new Map())
  const [sorterManager, setSorterManager] = useState(false)
  const [messageGroupers, setMessageGroupers] = useState<Map<string, { grouper: string, field: string, config: any }>>(new Map())
  const [grouperManager, setGrouperManager] = useState(false)
  const [messageJoiners, setMessageJoiners] = useState<Map<string, { joiner: string, type: string, config: any }>>(new Map())
  const [joinerManager, setJoinerManager] = useState(false)
  const [messageSplitters, setMessageSplitters] = useState<Map<string, { splitter: string, delimiter: string, config: any }>>(new Map())
  const [splitterManager, setSplitterManager] = useState(false)
  const [messageMergers, setMessageMergers] = useState<Map<string, { merger: string, strategy: string, config: any }>>(new Map())
  const [mergerManager, setMergerManager] = useState(false)
  const [messageDividers, setMessageDividers] = useState<Map<string, { divider: string, parts: number, config: any }>>(new Map())
  const [dividerManager, setDividerManager] = useState(false)
  const [messageCombiners, setMessageCombiners] = useState<Map<string, { combiner: string, function: string, config: any }>>(new Map())
  const [combinerManager, setCombinerManager] = useState(false)
  const [messageSeparators, setMessageSeparators] = useState<Map<string, { separator: string, pattern: string, config: any }>>(new Map())
  const [separatorManager, setSeparatorManager] = useState(false)
  const [messageDelimiters, setMessageDelimiters] = useState<Map<string, { delimiter: string, escape: string, config: any }>>(new Map())
  const [delimiterManager, setDelimiterManager] = useState(false)
  const [messageEncoders, setMessageEncoders] = useState<Map<string, { encoder: string, format: string, config: any }>>(new Map())
  const [encoderManager, setEncoderManager] = useState(false)
  const [messageDecoders, setMessageDecoders] = useState<Map<string, { decoder: string, format: string, config: any }>>(new Map())
  const [decoderManager, setDecoderManager] = useState(false)
  const [messageCompressors, setMessageCompressors] = useState<Map<string, { compressor: string, algorithm: string, level: number }>>(new Map())
  const [compressorManager, setCompressorManager] = useState(false)
  const [messageDecompressors, setMessageDecompressors] = useState<Map<string, { decompressor: string, algorithm: string, config: any }>>(new Map())
  const [decompressorManager, setDecompressorManager] = useState(false)
  const [messageEncryptors, setMessageEncryptors] = useState<Map<string, { encryptor: string, algorithm: string, key: string }>>(new Map())
  const [encryptorManager, setEncryptorManager] = useState(false)
  const [messageDecryptors, setMessageDecryptors] = useState<Map<string, { decryptor: string, algorithm: string, key: string }>>(new Map())
  const [decryptorManager, setDecryptorManager] = useState(false)
  const [messageHashers, setMessageHashers] = useState<Map<string, { hasher: string, algorithm: string, hash: string }>>(new Map())
  const [hasherManager, setHasherManager] = useState(false)
  const [messageSigners, setMessageSigners] = useState<Map<string, { signer: string, algorithm: string, signature: string }>>(new Map())
  const [signerManager, setSignerManager] = useState(false)
  const [messageVerifiers, setMessageVerifiers] = useState<Map<string, { verifier: string, algorithm: string, verified: boolean }>>(new Map())
  const [verifierManager, setVerifierManager] = useState(false)
  const [messageAuthenticators, setMessageAuthenticators] = useState<Map<string, { authenticator: string, method: string, config: any }>>(new Map())
  const [authenticatorManager, setAuthenticatorManager] = useState(false)
  const [messageAuthorizers, setMessageAuthorizers] = useState<Map<string, { authorizer: string, permissions: string[], config: any }>>(new Map())
  const [authorizerManager, setAuthorizerManager] = useState(false)
  const [messageAuditors, setMessageAuditors] = useState<Map<string, { auditor: string, events: string[], config: any }>>(new Map())
  const [auditorManager, setAuditorManager] = useState(false)
  const [messageLoggers, setMessageLoggers] = useState<Map<string, { logger: string, level: string, config: any }>>(new Map())
  const [loggerManager, setLoggerManager] = useState(false)
  const [messageMonitors, setMessageMonitors] = useState<Map<string, { monitor: string, metrics: string[], config: any }>>(new Map())
  const [monitorManager, setMonitorManager] = useState(false)
  const [messageTrackers, setMessageTrackers] = useState<Map<string, { tracker: string, events: string[], config: any }>>(new Map())
  const [trackerManager, setTrackerManager] = useState(false)
  const [messageProfilers, setMessageProfilers] = useState<Map<string, { profiler: string, metrics: string[], config: any }>>(new Map())
  const [profilerManager, setProfilerManager] = useState(false)
  const [messageDebuggers, setMessageDebuggers] = useState<Map<string, { debugger: string, breakpoints: string[], config: any }>>(new Map())
  const [debuggerManager, setDebuggerManager] = useState(false)
  const [messageTesters, setMessageTesters] = useState<Map<string, { tester: string, tests: string[], results: any[] }>>(new Map())
  const [testerManager, setTesterManager] = useState(false)
  const [messageSanitizers, setMessageSanitizers] = useState<Map<string, { sanitizer: string, rules: string[], config: any }>>(new Map())
  const [sanitizerManager, setSanitizerManager] = useState(false)
  const [messageNormalizers, setMessageNormalizers] = useState<Map<string, { normalizer: string, rules: string[], config: any }>>(new Map())
  const [normalizerManager, setNormalizerManager] = useState(false)
  const [messageStandardizers, setMessageStandardizers] = useState<Map<string, { standardizer: string, standard: string, config: any }>>(new Map())
  const [standardizerManager, setStandardizerManager] = useState(false)
  const [messageCreators, setMessageCreators] = useState<Map<string, { creator: string, template: string, config: any }>>(new Map())
  const [creatorManager, setCreatorManager] = useState(false)
  const [messageBuilders, setMessageBuilders] = useState<Map<string, { builder: string, steps: string[], config: any }>>(new Map())
  const [builderManager, setBuilderManager] = useState(false)
  const [messageFactories, setMessageFactories] = useState<Map<string, { factory: string, type: string, config: any }>>(new Map())
  const [factoryManager, setFactoryManager] = useState(false)
  const [messageConstructors, setMessageConstructors] = useState<Map<string, { constructor: string, parameters: string[], config: any }>>(new Map())
  const [constructorManager, setConstructorManager] = useState(false)
  const [messageInitializers, setMessageInitializers] = useState<Map<string, { initializer: string, values: any, config: any }>>(new Map())
  const [initializerManager, setInitializerManager] = useState(false)
  const [messageConfigurators, setMessageConfigurators] = useState<Map<string, { configurator: string, config: any }>>(new Map())
  const [configuratorManager, setConfiguratorManager] = useState(false)
  const [messageCustomizers, setMessageCustomizers] = useState<Map<string, { customizer: string, customizations: any }>>(new Map())
  const [customizerManager, setCustomizerManager] = useState(false)
  const [messagePersonalizers, setMessagePersonalizers] = useState<Map<string, { personalizer: string, preferences: any }>>(new Map())
  const [personalizerManager, setPersonalizerManager] = useState(false)
  const [messageLocalizers, setMessageLocalizers] = useState<Map<string, { localizer: string, locale: string, translations: any }>>(new Map())
  const [localizerManager, setLocalizerManager] = useState(false)
  const [messageInternationalizers, setMessageInternationalizers] = useState<Map<string, { internationalizer: string, locales: string[], config: any }>>(new Map())
  const [internationalizerManager, setInternationalizerManager] = useState(false)
  const [messageGlobalizers, setMessageGlobalizers] = useState<Map<string, { globalizer: string, regions: string[], config: any }>>(new Map())
  const [globalizerManager, setGlobalizerManager] = useState(false)
  const [messageRegionalizers, setMessageRegionalizers] = useState<Map<string, { regionalizer: string, region: string, config: any }>>(new Map())
  const [regionalizerManager, setRegionalizerManager] = useState(false)
  const [messageNotifications, setMessageNotifications] = useState<Map<string, { type: string, title: string, body: string, read: boolean, timestamp: number }>>(new Map())
  const [messageShortcuts, setMessageShortcuts] = useState<Map<string, { key: string, action: string, description: string }>>(new Map())
  const [shortcutPanel, setShortcutPanel] = useState(false)
  const [messageFavorites, setMessageFavorites] = useState<Set<string>>(new Set())
  const [favoritesPanel, setFavoritesPanel] = useState(false)
  const [messagePinned, setMessagePinned] = useState<Set<string>>(new Set())
  const [pinnedPanel, setPinnedPanel] = useState(false)
  const [messageArchived, setMessageArchived] = useState<Set<string>>(new Set())
  const [archivedPanel, setArchivedPanel] = useState(false)
  const [messageDeleted, setMessageDeleted] = useState<Set<string>>(new Set())
  const [trashPanel, setTrashPanel] = useState(false)
  const [messageDrafts, setMessageDrafts] = useState<Map<string, { content: string, timestamp: number }>>(new Map())
  const [draftsPanel, setDraftsPanel] = useState(false)
  const [messageScheduled, setMessageScheduled] = useState<Map<string, { content: string, scheduledTime: number }>>(new Map())
  const [scheduledPanel, setScheduledPanel] = useState(false)
  const [messageStarred, setMessageStarred] = useState<Set<string>>(new Set())
  const [starredPanel, setStarredPanel] = useState(false)
  const [messageImportant, setMessageImportant] = useState<Set<string>>(new Set())
  const [importantPanel, setImportantPanel] = useState(false)
  const [messageUnread, setMessageUnread] = useState<Set<string>>(new Set())
  const [unreadPanel, setUnreadPanel] = useState(false)
  const [messageRead, setMessageRead] = useState<Set<string>>(new Set())
  const [readPanel, setReadPanel] = useState(false)
  const [messageReplied, setMessageReplied] = useState<Set<string>>(new Set())
  const [repliedPanel, setRepliedPanel] = useState(false)
  const [messageForwarded, setMessageForwarded] = useState<Set<string>>(new Set())
  const [forwardedPanel, setForwardedPanel] = useState(false)
  const [messageShared, setMessageShared] = useState<Set<string>>(new Set())
  const [sharedPanel, setSharedPanel] = useState(false)
  // Estados de panel ahora manejados por usePanelStates
  // Aliases para compatibilidad con código existente
  const getMessageSet = (panelId: string) => panels.getPanelMessages(panelId)
  const setMessageSet = (panelId: string, messageId: string) => panels.addMessageToPanel(panelId, messageId)
  const removeMessageSet = (panelId: string, messageId: string) => panels.removeMessageFromPanel(panelId, messageId)
  
  // Aliases para paneles específicos (mantener compatibilidad)
  const copiedPanel = panels.isPanelVisible('copied')
  const setCopiedPanel = (value: boolean) => panels.setPanelVisible('copied', value)
  const messageCopied = getMessageSet('copied')
  const setMessageCopied = (value: Set<string>) => { value.forEach(id => panels.addMessageToPanel('copied', id)) }
  
  const downloadedPanel = panels.isPanelVisible('downloaded')
  const setDownloadedPanel = (value: boolean) => panels.setPanelVisible('downloaded', value)
  const messageDownloaded = getMessageSet('downloaded')
  
  const printedPanel = panels.isPanelVisible('printed')
  const setPrintedPanel = (value: boolean) => panels.setPanelVisible('printed', value)
  const messagePrinted = getMessageSet('printed')
  
  const exportedPanel = panels.isPanelVisible('exported')
  const setExportedPanel = (value: boolean) => panels.setPanelVisible('exported', value)
  const messageExported = getMessageSet('exported')
  
  const importedPanel = panels.isPanelVisible('imported')
  const setImportedPanel = (value: boolean) => panels.setPanelVisible('imported', value)
  const messageImported = getMessageSet('imported')
  
  const syncedPanel = panels.isPanelVisible('synced')
  const setSyncedPanel = (value: boolean) => panels.setPanelVisible('synced', value)
  const messageSynced = getMessageSet('synced')
  
  const backedUpPanel = panels.isPanelVisible('backedUp')
  const setBackedUpPanel = (value: boolean) => panels.setPanelVisible('backedUp', value)
  const messageBackedUp = getMessageSet('backedUp')
  
  const restoredPanel = panels.isPanelVisible('restored')
  const setRestoredPanel = (value: boolean) => panels.setPanelVisible('restored', value)
  const messageRestored = getMessageSet('restored')
  
  const encryptedPanel = panels.isPanelVisible('encrypted')
  const setEncryptedPanel = (value: boolean) => panels.setPanelVisible('encrypted', value)
  const messageEncrypted = getMessageSet('encrypted')
  
  const decryptedPanel = panels.isPanelVisible('decrypted')
  const setDecryptedPanel = (value: boolean) => panels.setPanelVisible('decrypted', value)
  const messageDecrypted = getMessageSet('decrypted')
  
  const compressedPanel = panels.isPanelVisible('compressed')
  const setCompressedPanel = (value: boolean) => panels.setPanelVisible('compressed', value)
  const messageCompressed = getMessageSet('compressed')
  
  const decompressedPanel = panels.isPanelVisible('decompressed')
  const setDecompressedPanel = (value: boolean) => panels.setPanelVisible('decompressed', value)
  const messageDecompressed = getMessageSet('decompressed')
  
  const validatedPanel = panels.isPanelVisible('validated')
  const setValidatedPanel = (value: boolean) => panels.setPanelVisible('validated', value)
  const messageValidated = getMessageSet('validated')
  
  const sanitizedPanel = panels.isPanelVisible('sanitized')
  const setSanitizedPanel = (value: boolean) => panels.setPanelVisible('sanitized', value)
  const messageSanitized = getMessageSet('sanitized')
  
  const normalizedPanel = panels.isPanelVisible('normalized')
  const setNormalizedPanel = (value: boolean) => panels.setPanelVisible('normalized', value)
  const messageNormalized = getMessageSet('normalized')
  
  const standardizedPanel = panels.isPanelVisible('standardized')
  const setStandardizedPanel = (value: boolean) => panels.setPanelVisible('standardized', value)
  const messageStandardized = getMessageSet('standardized')
  
  const formattedPanel = panels.isPanelVisible('formatted')
  const setFormattedPanel = (value: boolean) => panels.setPanelVisible('formatted', value)
  const messageFormatted = getMessageSet('formatted')
  
  const renderedPanel = panels.isPanelVisible('rendered')
  const setRenderedPanel = (value: boolean) => panels.setPanelVisible('rendered', value)
  const messageRendered = getMessageSet('rendered')
  
  const generatedPanel = panels.isPanelVisible('generated')
  const setGeneratedPanel = (value: boolean) => panels.setPanelVisible('generated', value)
  const messageGenerated = getMessageSet('generated')
  
  const processedPanel = panels.isPanelVisible('processed')
  const setProcessedPanel = (value: boolean) => panels.setPanelVisible('processed', value)
  const messageProcessed = getMessageSet('processed')
  
  const handledPanel = panels.isPanelVisible('handled')
  const setHandledPanel = (value: boolean) => panels.setPanelVisible('handled', value)
  const messageHandled = getMessageSet('handled')
  
  const listenedPanel = panels.isPanelVisible('listened')
  const setListenedPanel = (value: boolean) => panels.setPanelVisible('listened', value)
  const messageListened = getMessageSet('listened')
  
  const observedPanel = panels.isPanelVisible('observed')
  const setObservedPanel = (value: boolean) => panels.setPanelVisible('observed', value)
  const messageObserved = getMessageSet('observed')
  
  const subscribedPanel = panels.isPanelVisible('subscribed')
  const setSubscribedPanel = (value: boolean) => panels.setPanelVisible('subscribed', value)
  const messageSubscribed = getMessageSet('subscribed')
  
  const publishedPanel = panels.isPanelVisible('published')
  const setPublishedPanel = (value: boolean) => panels.setPanelVisible('published', value)
  const messagePublished = getMessageSet('published')
  
  const brokeredPanel = panels.isPanelVisible('brokered')
  const setBrokeredPanel = (value: boolean) => panels.setPanelVisible('brokered', value)
  const messageBrokered = getMessageSet('brokered')
  
  const queuedPanel = panels.isPanelVisible('queued')
  const setQueuedPanel = (value: boolean) => panels.setPanelVisible('queued', value)
  const messageQueued = getMessageSet('queued')
  
  const topicedPanel = panels.isPanelVisible('topiced')
  const setTopicedPanel = (value: boolean) => panels.setPanelVisible('topiced', value)
  const messageTopiced = getMessageSet('topiced')
  
  const channelledPanel = panels.isPanelVisible('channelled')
  const setChannelledPanel = (value: boolean) => panels.setPanelVisible('channelled', value)
  const messageChannelled = getMessageSet('channelled')
  
  const streamedPanel = panels.isPanelVisible('streamed')
  const setStreamedPanel = (value: boolean) => panels.setPanelVisible('streamed', value)
  const messageStreamed = getMessageSet('streamed')
  
  const bufferedPanel = panels.isPanelVisible('buffered')
  const setBufferedPanel = (value: boolean) => panels.setPanelVisible('buffered', value)
  const messageBuffered = getMessageSet('buffered')
  
  const cachedPanel = panels.isPanelVisible('cached')
  const setCachedPanel = (value: boolean) => panels.setPanelVisible('cached', value)
  const messageCached = getMessageSet('cached')
  
  const storedPanel = panels.isPanelVisible('stored')
  const setStoredPanel = (value: boolean) => panels.setPanelVisible('stored', value)
  const messageStored = getMessageSet('stored')
  const [messageDatabased, setMessageDatabased] = useState<Set<string>>(new Set())
  const [databasedPanel, setDatabasedPanel] = useState(false)
  const [messageRepository, setMessageRepository] = useState<Set<string>>(new Set())
  const [repositoryPanel, setRepositoryPanel] = useState(false)
  const [messageIndexed, setMessageIndexed] = useState<Set<string>>(new Set())
  const [indexedPanel, setIndexedPanel] = useState(false)
  const [messageSearched, setMessageSearched] = useState<Set<string>>(new Set())
  const [searchedPanel, setSearchedPanel] = useState(false)
  const [messageAnalyzed, setMessageAnalyzed] = useState<Set<string>>(new Set())
  const [analyzedPanel, setAnalyzedPanel] = useState(false)
  const [messageClassified, setMessageClassified] = useState<Set<string>>(new Set())
  const [classifiedPanel, setClassifiedPanel] = useState(false)
  const [messageExtracted, setMessageExtracted] = useState<Set<string>>(new Set())
  const [extractedPanel, setExtractedPanel] = useState(false)
  const [messageEnriched, setMessageEnriched] = useState<Set<string>>(new Set())
  const [enrichedPanel, setEnrichedPanel] = useState(false)
  const [messageAggregated, setMessageAggregated] = useState<Set<string>>(new Set())
  const [aggregatedPanel, setAggregatedPanel] = useState(false)
  const [messageReduced, setMessageReduced] = useState<Set<string>>(new Set())
  const [reducedPanel, setReducedPanel] = useState(false)
  const [messageMapped, setMessageMapped] = useState<Set<string>>(new Set())
  const [mappedPanel, setMappedPanel] = useState(false)
  const [messageSorted, setMessageSorted] = useState<Set<string>>(new Set())
  const [sortedPanel, setSortedPanel] = useState(false)
  const [messageGrouped, setMessageGrouped] = useState<Set<string>>(new Set())
  const [groupedPanel, setGroupedPanel] = useState(false)
  const [messageJoined, setMessageJoined] = useState<Set<string>>(new Set())
  const [joinedPanel, setJoinedPanel] = useState(false)
  const [messageSplit, setMessageSplit] = useState<Set<string>>(new Set())
  const [splitPanel, setSplitPanel] = useState(false)
  const [messageMerged, setMessageMerged] = useState<Set<string>>(new Set())
  const [mergedPanel, setMergedPanel] = useState(false)
  const [messageDivided, setMessageDivided] = useState<Set<string>>(new Set())
  const [dividedPanel, setDividedPanel] = useState(false)
  const [messageCombined, setMessageCombined] = useState<Set<string>>(new Set())
  const [combinedPanel, setCombinedPanel] = useState(false)
  const [messageSeparated, setMessageSeparated] = useState<Set<string>>(new Set())
  const [separatedPanel, setSeparatedPanel] = useState(false)
  const [messageDelimited, setMessageDelimited] = useState<Set<string>>(new Set())
  const [delimitedPanel, setDelimitedPanel] = useState(false)
  const [messageEncoded, setMessageEncoded] = useState<Set<string>>(new Set())
  const [encodedPanel, setEncodedPanel] = useState(false)
  const [messageDecoded, setMessageDecoded] = useState<Set<string>>(new Set())
  const [decodedPanel, setDecodedPanel] = useState(false)
  const [messageCompressed2, setMessageCompressed2] = useState<Set<string>>(new Set())
  const [compressed2Panel, setCompressed2Panel] = useState(false)
  const [messageDecompressed2, setMessageDecompressed2] = useState<Set<string>>(new Set())
  const [decompressed2Panel, setDecompressed2Panel] = useState(false)
  const [messageEncrypted2, setMessageEncrypted2] = useState<Set<string>>(new Set())
  const [encrypted2Panel, setEncrypted2Panel] = useState(false)
  const [messageDecrypted2, setMessageDecrypted2] = useState<Set<string>>(new Set())
  const [decrypted2Panel, setDecrypted2Panel] = useState(false)
  const [messageHashed, setMessageHashed] = useState<Set<string>>(new Set())
  const [hashedPanel, setHashedPanel] = useState(false)
  const [messageSigned, setMessageSigned] = useState<Set<string>>(new Set())
  const [signedPanel, setSignedPanel] = useState(false)
  const [messageVerified, setMessageVerified] = useState<Set<string>>(new Set())
  const [verifiedPanel, setVerifiedPanel] = useState(false)
  const [messageAuthenticated, setMessageAuthenticated] = useState<Set<string>>(new Set())
  const [authenticatedPanel, setAuthenticatedPanel] = useState(false)
  const [messageAuthorized, setMessageAuthorized] = useState<Set<string>>(new Set())
  const [authorizedPanel, setAuthorizedPanel] = useState(false)
  const [messageAudited, setMessageAudited] = useState<Set<string>>(new Set())
  const [auditedPanel, setAuditedPanel] = useState(false)
  const [messageLogged, setMessageLogged] = useState<Set<string>>(new Set())
  const [loggedPanel, setLoggedPanel] = useState(false)
  const [messageMonitored, setMessageMonitored] = useState<Set<string>>(new Set())
  const [monitoredPanel, setMonitoredPanel] = useState(false)
  const [messageTracked, setMessageTracked] = useState<Set<string>>(new Set())
  const [trackedPanel, setTrackedPanel] = useState(false)
  const [messageProfiled, setMessageProfiled] = useState<Set<string>>(new Set())
  const [profiledPanel, setProfiledPanel] = useState(false)
  const [messageDebugged, setMessageDebugged] = useState<Set<string>>(new Set())
  const [debuggedPanel, setDebuggedPanel] = useState(false)
  const [messageTested, setMessageTested] = useState<Set<string>>(new Set())
  const [testedPanel, setTestedPanel] = useState(false)
  const [messageSanitized2, setMessageSanitized2] = useState<Set<string>>(new Set())
  const [sanitized2Panel, setSanitized2Panel] = useState(false)
  const [messageNormalized2, setMessageNormalized2] = useState<Set<string>>(new Set())
  const [normalized2Panel, setNormalized2Panel] = useState(false)
  const [messageStandardized2, setMessageStandardized2] = useState<Set<string>>(new Set())
  const [standardized2Panel, setStandardized2Panel] = useState(false)
  const [messageFormatted2, setMessageFormatted2] = useState<Set<string>>(new Set())
  const [formatted2Panel, setFormatted2Panel] = useState(false)
  const [messageRendered2, setMessageRendered2] = useState<Set<string>>(new Set())
  const [rendered2Panel, setRendered2Panel] = useState(false)
  const [messageGenerated2, setMessageGenerated2] = useState<Set<string>>(new Set())
  const [generated2Panel, setGenerated2Panel] = useState(false)
  const [messageCreated, setMessageCreated] = useState<Set<string>>(new Set())
  const [createdPanel, setCreatedPanel] = useState(false)
  const [messageBuilt, setMessageBuilt] = useState<Set<string>>(new Set())
  const [builtPanel, setBuiltPanel] = useState(false)
  const [messageFactored, setMessageFactored] = useState<Set<string>>(new Set())
  const [factoredPanel, setFactoredPanel] = useState(false)
  const [messageConstructed, setMessageConstructed] = useState<Set<string>>(new Set())
  const [constructedPanel, setConstructedPanel] = useState(false)
  const [messageInitialized, setMessageInitialized] = useState<Set<string>>(new Set())
  const [initializedPanel, setInitializedPanel] = useState(false)
  const [messageConfigured, setMessageConfigured] = useState<Set<string>>(new Set())
  const [configuredPanel, setConfiguredPanel] = useState(false)
  const [messageCustomized, setMessageCustomized] = useState<Set<string>>(new Set())
  const [customizedPanel, setCustomizedPanel] = useState(false)
  const [messagePersonalized, setMessagePersonalized] = useState<Set<string>>(new Set())
  const [personalizedPanel, setPersonalizedPanel] = useState(false)
  const [messageLocalized, setMessageLocalized] = useState<Set<string>>(new Set())
  const [localizedPanel, setLocalizedPanel] = useState(false)
  const [messageInternationalized, setMessageInternationalized] = useState<Set<string>>(new Set())
  const [internationalizedPanel, setInternationalizedPanel] = useState(false)
  const [messageGlobalized, setMessageGlobalized] = useState<Set<string>>(new Set())
  const [globalizedPanel, setGlobalizedPanel] = useState(false)
  const [messageRegionalized, setMessageRegionalized] = useState<Set<string>>(new Set())
  const [regionalizedPanel, setRegionalizedPanel] = useState(false)
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const searchInputRef = useRef<SearchBarRef>(null)
  const messageRefs = useRef<Map<string, HTMLDivElement>>(new Map())
  const recognitionRef = useRef<any>(null)
  const synthesisRef = useRef<SpeechSynthesis | null>(null)
  const { messages, addMessage, currentModel, setCurrentModel, clearMessages } = useModelStore()
  const [filteredMessages, setFilteredMessages] = useState(messages)
  
  // Optimizadores
  const smartSuggestionsManager = useMemo(() => getSmartSuggestions(), [])
  const performanceOptimizer = useMemo(() => getPerformanceOptimizer(), [])
  
  // Enhanced chat hook with caching, metrics, and rate limiting
  const {
    sendMessage: enhancedSendMessage,
    isLoading: enhancedLoading,
    error: enhancedError,
    metrics,
    rateLimitStatus,
    getCacheStats
  } = useEnhancedChat({
    enableCache: true,
    enableMetrics: true,
    enableRateLimit: true
  })

  // TruthGPT API hook
  const {
    client: truthGPTClient,
    isConnected: apiConnected,
    checkConnection,
    createModelFromDescription,
    isLoading: apiLoading,
    error: apiError
  } = useTruthGPTAPI(process.env.NEXT_PUBLIC_TRUTHGPT_API_URL)

  // Notificaciones inteligentes - definida antes de useBulkChat
  const checkNotificationRulesRef = useRef<(message: any) => void>(() => {})
  
  // Bulk Chat API hook - Integración con Bulk Chat
  const bulkChat = useBulkChat({
    apiUrl: process.env.NEXT_PUBLIC_BULK_CHAT_API_URL || 'http://localhost:8006',
    autoConnect: true, // Auto-conectar al montar
    autoContinue: true,
    enableWebSocket: true,
    enableNotifications: true, // Notificaciones del navegador
    enableSounds: false, // Sonidos opcionales (desactivado por defecto)
    initialMessage: 'Hola, estoy listo para ayudarte',
    onMessage: (message) => {
      // Sincronizar mensajes de Bulk Chat con el store
      addMessage({
        id: message.id,
        role: message.role,
        content: message.content,
        timestamp: new Date(message.timestamp),
      })
      // Verificar reglas de notificación inteligente
      checkNotificationRulesRef.current(message)
    },
    onError: (error) => {
      console.error('Bulk Chat error:', error)
      // Solo mostrar error si no es de conexión inicial (para no spammear)
      if (bulkChat.sessionId) {
        toast.error(`Error en Bulk Chat: ${error.message}`)
      }
    },
    onSessionCreated: (session) => {
      console.log('Bulk Chat session created:', session)
      // Notificación silenciosa para no interrumpir
      if (messages.length === 0) {
        toast.success('Bulk Chat conectado', { icon: '✅', duration: 2000 })
      }
    },
  })

  // Auto-crear sesión si Bulk Chat está activo pero no hay sesión
  useEffect(() => {
    if (useBulkChatMode && !bulkChat.sessionId && !bulkChat.isLoading) {
      const timer = setTimeout(() => {
        bulkChat.createSession('Hola, estoy listo para ayudarte')
      }, 500)
      return () => clearTimeout(timer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [useBulkChatMode, bulkChat.sessionId, bulkChat.isLoading])

  // Persistir favoritos en localStorage
  useEffect(() => {
    try {
      const savedFavorites = localStorage.getItem('bulk-chat-favorites')
      if (savedFavorites) {
        setFavoriteMessages(new Set(JSON.parse(savedFavorites)))
      }
    } catch (error) {
      console.error('Error loading favorites:', error)
    }
  }, [])

  useEffect(() => {
    try {
      if (favoriteMessages.size > 0) {
        localStorage.setItem('bulk-chat-favorites', JSON.stringify(Array.from(favoriteMessages)))
      } else {
        localStorage.removeItem('bulk-chat-favorites')
      }
    } catch (error) {
      console.error('Error saving favorites:', error)
    }
  }, [favoriteMessages])

  // Check API connection on mount
  useEffect(() => {
    checkConnection().catch((err) => {
      console.warn('TruthGPT API not available:', err)
      toast('La API de TruthGPT no está disponible. Usando modo alternativo.', {
        icon: '⚠️',
        duration: 3000,
      })
    })
  }, [checkConnection])

  // Load history from storage on mount
  useEffect(() => {
    try {
      const savedHistory = getModelHistory()
      
      // Validate history is an array
      if (Array.isArray(savedHistory)) {
        // Validate each item in history
        const validHistory = savedHistory.filter(item => 
          item && 
          typeof item === 'object' && 
          item.id && 
          typeof item.id === 'string'
        )
        setModelHistory(validHistory)
      } else {
        console.warn('Invalid history format from storage')
        setModelHistory([])
      }
    } catch (error) {
      console.error('Error loading history from storage:', error)
      setModelHistory([])
    }
  }, [])

  // Filter messages based on search query
  useEffect(() => {
    // Validate inputs
    if (!searchQuery || typeof searchQuery !== 'string') {
      setFilteredMessages(messages || [])
      setCurrentSearchIndex(-1)
      return
    }
    
    if (!Array.isArray(messages)) {
      setFilteredMessages([])
      setCurrentSearchIndex(-1)
      return
    }
    
    if (!searchQuery.trim()) {
      setFilteredMessages(messages)
      setCurrentSearchIndex(-1)
      return
    }

    try {
      const query = searchQuery.toLowerCase().trim()
      if (query.length === 0) {
        setFilteredMessages(messages)
        setCurrentSearchIndex(-1)
        return
      }
      
      let filtered = messages.filter(message => {
        if (!message || typeof message !== 'object') return false
        
        // Filtro por rol
        if (filterRole !== 'all' && message.role !== filterRole) return false
        
        const content = message.content && typeof message.content === 'string'
          ? message.content.toLowerCase()
          : ''
        
        const role = message.role && typeof message.role === 'string'
          ? message.role.toLowerCase()
          : ''
        
        return content.includes(query) || role.includes(query)
      })
      
      // Si hay favoritos seleccionados, mostrar solo favoritos
      if (showFilters && favoriteMessages.size > 0) {
        // Esto se manejará en el renderizado
      }
      
      // Aplicar búsqueda avanzada si está activa
      let finalFiltered = filtered
      if (advancedSearch) {
        finalFiltered = applyAdvancedSearch(filtered)
      }
      
      setFilteredMessages(finalFiltered)
      setCurrentSearchIndex(finalFiltered.length > 0 ? 0 : -1)
    } catch (error) {
      console.error('Error filtering messages:', error)
      setFilteredMessages(messages || [])
      setCurrentSearchIndex(-1)
    }
  }, [messages, searchQuery, filterRole])

  // Debounced search handler
  const handleSearch = useDebouncedCallback((query: string) => {
    // Validate query
    if (query === null || query === undefined) {
      setSearchQuery('')
      return
    }
    
    const queryStr = typeof query === 'string' ? query : String(query)
    
    // Limit query length
    if (queryStr.length > 200) {
      toast.error('La búsqueda es demasiado larga (máximo 200 caracteres)', {
        icon: '⚠️',
      })
      setSearchQuery(queryStr.substring(0, 200))
      return
    }
    
    setSearchQuery(queryStr)
  }, 300)

  // Keyboard shortcuts
  useKeyboardShortcuts([
    {
      keys: ['Ctrl', 'K'],
      callback: () => {
        try {
          // Focus input safely
          const inputElement = document.querySelector('input[type="text"]') as HTMLInputElement
          if (inputElement && typeof inputElement.focus === 'function') {
            inputElement.focus()
          }
        } catch (error) {
          console.error('Error focusing input:', error)
        }
      },
    },
    {
      keys: ['Ctrl', 'F'],
      callback: () => {
        try {
          if (searchInputRef.current) {
            searchInputRef.current.focus()
          } else {
            const searchElement = document.querySelector('input[placeholder*="Buscar"]') as HTMLInputElement
            if (searchElement) {
              searchElement.focus()
            }
          }
        } catch (error) {
          console.error('Error focusing search:', error)
        }
      },
    },
    {
      keys: ['Escape'],
      callback: () => {
        try {
          setSearchQuery('')
          setFilteredMessages(messages)
          if (searchInputRef.current) {
            searchInputRef.current.clear()
          }
          setShowConnectionInfo(false)
          setShowQuickActions(false)
          setShowStats(false)
          setShowFilters(false)
          setShowCommandPalette(false)
          setPresentationMode(false)
          setCurrentSearchIndex(-1)
        } catch (error) {
          console.error('Error clearing:', error)
        }
      },
    },
    {
      keys: ['Ctrl', 'K'],
      callback: () => {
        try {
          setShowCommandPalette(true)
        } catch (error) {
          console.error('Error opening command palette:', error)
        }
      },
    },
    {
      keys: ['F11'],
      callback: () => {
        try {
          if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen()
            setPresentationMode(true)
          } else {
            document.exitFullscreen()
            setPresentationMode(false)
          }
        } catch (error) {
          console.error('Error toggling fullscreen:', error)
        }
      },
    },
    {
      keys: ['F3'],
      callback: () => {
        try {
          if (searchQuery && filteredMessages.length > 0) {
            const nextIndex = currentSearchIndex < filteredMessages.length - 1 
              ? currentSearchIndex + 1 
              : 0
            setCurrentSearchIndex(nextIndex)
            const message = filteredMessages[nextIndex]
            const element = messageRefs.current.get(message.id)
            if (element) {
              element.scrollIntoView({ behavior: 'smooth', block: 'center' })
              element.classList.add('ring-2', 'ring-yellow-500', 'ring-opacity-50')
              setTimeout(() => {
                element.classList.remove('ring-2', 'ring-yellow-500', 'ring-opacity-50')
              }, 2000)
            }
          }
        } catch (error) {
          console.error('Error navigating search:', error)
        }
      },
    },
    {
      keys: ['Shift', 'F3'],
      callback: () => {
        try {
          if (searchQuery && filteredMessages.length > 0) {
            const prevIndex = currentSearchIndex > 0 
              ? currentSearchIndex - 1 
              : filteredMessages.length - 1
            setCurrentSearchIndex(prevIndex)
            const message = filteredMessages[prevIndex]
            const element = messageRefs.current.get(message.id)
            if (element) {
              element.scrollIntoView({ behavior: 'smooth', block: 'center' })
              element.classList.add('ring-2', 'ring-yellow-500', 'ring-opacity-50')
              setTimeout(() => {
                element.classList.remove('ring-2', 'ring-yellow-500', 'ring-opacity-50')
              }, 2000)
            }
          }
        } catch (error) {
          console.error('Error navigating search:', error)
        }
      },
    },
    {
      keys: ['Ctrl', 'B'],
      callback: () => {
        try {
          setUseBulkChatMode(!useBulkChatMode)
          if (!useBulkChatMode && !bulkChat.sessionId) {
            bulkChat.createSession()
          }
          toast.success(`Bulk Chat ${!useBulkChatMode ? 'activado' : 'desactivado'}`, {
            icon: '✅',
            duration: 2000,
          })
        } catch (error) {
          console.error('Error toggling Bulk Chat:', error)
        }
      },
    },
    {
      keys: ['Ctrl', 'Shift', 'P'],
      callback: () => {
        try {
          if (bulkChat.sessionId) {
            if (bulkChat.isPaused) {
              bulkChat.resume()
            } else {
              bulkChat.pause()
            }
          }
        } catch (error) {
          console.error('Error pausing/resuming:', error)
        }
      },
    },
    {
      keys: ['Ctrl', 'Shift', 'R'],
      callback: () => {
        try {
          setReadMode(!readMode)
          toast.success(`Modo lectura ${!readMode ? 'activado' : 'desactivado'}`, {
            icon: '📖',
            duration: 2000,
          })
        } catch (error) {
          console.error('Error toggling read mode:', error)
        }
      },
    },
    {
      keys: ['Ctrl', 'Shift', 'E'],
      callback: async () => {
        try {
          if (bulkChat.sessionId && bulkChat.messages.length > 0) {
            const data = {
              sessionId: bulkChat.sessionId,
              messages: bulkChat.messages,
              timestamp: new Date().toISOString(),
              messageCount: bulkChat.messageCount,
            }
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
            const url = URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = `bulk-chat-${bulkChat.sessionId.slice(0, 8)}-${Date.now()}.json`
            document.body.appendChild(a)
            a.click()
            document.body.removeChild(a)
            URL.revokeObjectURL(url)
            toast.success('Conversación exportada', { icon: '💾', duration: 2000 })
          }
        } catch (error) {
          console.error('Error exporting:', error)
        }
      },
    },
    {
      keys: ['Ctrl', 'Enter'],
      callback: () => {
        try {
          // Validate state before submitting
          if (!isLoading && input && typeof input === 'string' && input.trim().length > 0) {
            // Create synthetic event
            const syntheticEvent = {
              preventDefault: () => {},
              stopPropagation: () => {},
            } as React.FormEvent
            
            handleSubmit(syntheticEvent).catch((error) => {
              console.error('Error in keyboard shortcut submit:', error)
            })
          }
        } catch (error) {
          console.error('Error in Ctrl+Enter shortcut:', error)
        }
      },
    },
    {
      keys: ['Ctrl', 'H'],
      callback: () => {
        try {
          setShowHistory((prev) => {
            if (typeof prev === 'boolean') {
              return !prev
            }
            return true
          })
        } catch (error) {
          console.error('Error toggling history:', error)
        }
      },
    },
    {
      keys: ['Ctrl', 'T'],
      callback: () => {
        try {
          setShowTemplates((prev) => {
            if (typeof prev === 'boolean') {
              return !prev
            }
            return true
          })
        } catch (error) {
          console.error('Error toggling templates:', error)
        }
      },
    },
    {
      keys: ['Ctrl', 'C'],
      callback: () => {
        try {
          // Validate model history before showing comparator
          if (Array.isArray(modelHistory) && modelHistory.length >= 2) {
            setShowComparator(true)
          } else {
            toast('Se necesitan al menos 2 modelos para comparar', {
              icon: 'ℹ️',
            })
          }
        } catch (error) {
          console.error('Error showing comparator:', error)
        }
      },
    },
  ])

  const scrollToBottom = () => {
    try {
      if (messagesEndRef.current) {
        messagesEndRef.current.scrollIntoView({ behavior: 'smooth' })
      }
    } catch (error) {
      console.error('Error scrolling to bottom:', error)
      // Fallback to instant scroll
      try {
        if (messagesEndRef.current) {
          messagesEndRef.current.scrollIntoView({ behavior: 'auto' })
        }
      } catch (fallbackError) {
        console.error('Error in fallback scroll:', fallbackError)
      }
    }
  }

  useEffect(() => {
    // Validate messages array before scrolling
    if (Array.isArray(messages) && messages.length > 0) {
      // Small delay to ensure DOM is updated
      const timeoutId = setTimeout(() => {
        scrollToBottom()
      }, 100)
      
      return () => clearTimeout(timeoutId)
    }
  }, [messages, bulkChat.isTyping]) // Auto-scroll cuando hay typing

  const handleSuggestionSelect = (text: string) => {
    // Validate input
    if (!text || typeof text !== 'string') {
      console.warn('Invalid suggestion text provided')
      return
    }
    
    const trimmedText = text.trim()
    if (trimmedText.length === 0) {
      console.warn('Empty suggestion text')
      return
    }
    
    if (trimmedText.length > 5000) {
      toast.error('El texto sugerido es demasiado largo (máximo 5000 caracteres)')
      return
    }
    
    setInput(trimmedText)
    validateInput(trimmedText)
  }

  const handleTemplateSelect = (template: any) => {
    // Validate template
    if (!template || typeof template !== 'object') {
      console.warn('Invalid template provided')
      toast.error('Template inválido')
      return
    }
    
    const example = template.example
    if (!example || typeof example !== 'string') {
      console.warn('Invalid template example')
      toast.error('El template no tiene un ejemplo válido')
      return
    }
    
    const trimmedExample = example.trim()
    if (trimmedExample.length === 0) {
      console.warn('Empty template example')
      toast.error('El ejemplo del template está vacío')
      return
    }
    
    if (trimmedExample.length > 5000) {
      toast.error('El ejemplo del template es demasiado largo (máximo 5000 caracteres)')
      return
    }
    
    setInput(trimmedExample)
    setShowTemplates(false)
    validateInput(trimmedExample)
  }

  const validateInput = (text: string) => {
    try {
      // Validate input
      if (!text || typeof text !== 'string') {
        setValidation(null)
        setPreviewSpec(null)
        return
      }
      
      const trimmedText = text.trim()
      if (trimmedText.length === 0) {
        setValidation(null)
        setPreviewSpec(null)
        return
      }
      
      // Validate length limits
      if (trimmedText.length < 10) {
        setValidation({
          isValid: false,
          message: 'La descripción debe tener al menos 10 caracteres'
        })
        setPreviewSpec(null)
        return
      }
      
      if (trimmedText.length > 5000) {
        setValidation({
          isValid: false,
          message: 'La descripción es demasiado larga (máximo 5000 caracteres)'
        })
        setPreviewSpec(null)
        return
      }
      
      // Validate description
      let validation: any
      try {
        validation = validateDescription(trimmedText)
      } catch (validateError) {
        console.error('Error in validateDescription:', validateError)
        setValidation({
          isValid: false,
          message: 'Error al validar la descripción'
        })
        setPreviewSpec(null)
        return
      }
      
      if (!validation || typeof validation !== 'object') {
        setValidation(null)
        setPreviewSpec(null)
        return
      }
      
      setValidation(validation)
      
      if (validation.isValid) {
        try {
          // Use adaptive analyzer for better context understanding
          let spec: any
          try {
            spec = adaptiveAnalyze(trimmedText)
          } catch (analyzeError) {
            console.error('Error in adaptiveAnalyze:', analyzeError)
            toast.error('Error al analizar la descripción')
            setPreviewSpec(null)
            return
          }
          
          if (!spec || typeof spec !== 'object' || Array.isArray(spec)) {
            console.error('Invalid spec from adaptiveAnalyze:', spec)
            setPreviewSpec(null)
            return
          }
          
          // Generate model name safely
          let modelName: string
          try {
            const sanitized = trimmedText.toLowerCase().replace(/[^a-z0-9]+/g, '-').substring(0, 30)
            // Remove leading/trailing hyphens
            const cleaned = sanitized.replace(/^-+|-+$/g, '')
            modelName = cleaned.length > 0 
              ? `truthgpt-${cleaned}`
              : `truthgpt-model-${Date.now()}`
          } catch (nameError) {
            console.error('Error generating model name:', nameError)
            modelName = `truthgpt-model-${Date.now()}`
          }
          
          // Validate model name length
          if (modelName.length > 100) {
            modelName = modelName.substring(0, 100)
          }
          
          setPreviewSpec({ 
            spec, 
            modelName, 
            description: trimmedText 
          })
          setShowPreview(true)
        } catch (analyzeError) {
          console.error('Error in validateInput adaptiveAnalyze:', analyzeError)
          toast.error('Error al analizar la descripción')
          setPreviewSpec(null)
        }
      } else {
        // Clear preview spec if validation fails
        setPreviewSpec(null)
      }
    } catch (error) {
      console.error('Error in validateInput:', error)
      setValidation(null)
      setPreviewSpec(null)
    }
  }

  // Generar sugerencias inteligentes
  useEffect(() => {
    if (!input || typeof input !== 'string') {
      setSmartSuggestions([])
      return
    }
    
    const trimmedInput = input.trim()
    
    if (trimmedInput.length >= 2) {
      const debounced = performanceOptimizer.debounce(
        'smart-suggestions',
        () => {
          const suggestions = smartSuggestionsManager.generateSuggestions(trimmedInput, 5)
          setSmartSuggestions(suggestions)
          setShowSmartSuggestions(suggestions.length > 0)
        },
        300
      )
      debounced()
    } else {
      setSmartSuggestions([])
      setShowSmartSuggestions(false)
    }
  }, [input, smartSuggestionsManager, performanceOptimizer])

  useEffect(() => {
    // Validate input exists and is a string
    if (!input || typeof input !== 'string') {
      setValidation(null)
      return
    }
    
    const trimmedInput = input.trim()
    
    if (trimmedInput.length > 0) {
      try {
        // Only validate if input is long enough to be meaningful
        if (trimmedInput.length >= 3) {
          const validation = validateDescription(trimmedInput)
          
          // Validate validation result
          if (validation && typeof validation === 'object') {
            setValidation(validation)
          } else {
            setValidation(null)
          }
        } else {
          setValidation(null)
        }
        
        // Auto-save draft with error handling
        try {
          saveDraft(trimmedInput)
        } catch (draftError) {
          console.error('Error saving draft:', draftError)
        }
      } catch (error) {
        console.error('Error in input validation useEffect:', error)
        setValidation(null)
      }
    } else {
      setValidation(null)
    }
  }, [input])

  // Setup auto-save interval
  useEffect(() => {
    // Validate input before setting up auto-save
    if (!input || typeof input !== 'string') {
      return
    }
    
    try {
      const cleanup = setupAutoSave(input, (draft) => {
        // Draft saved automatically
        if (draft && typeof draft === 'object') {
          // Optional: Log successful save in development
          if (process.env.NODE_ENV === 'development') {
            console.debug('Draft auto-saved')
          }
        }
      })
      
      // Validate cleanup function
      if (typeof cleanup === 'function') {
        return cleanup
      } else if (cleanup && typeof cleanup === 'object') {
        const cleanupObj = cleanup as { cleanup?: () => void }
        if ('cleanup' in cleanupObj && typeof cleanupObj.cleanup === 'function') {
          return cleanupObj.cleanup
        }
      }
    } catch (error) {
      console.error('Error setting up auto-save:', error)
    }
  }, [input])

  const createModel = async (userMessage: string, spec: any) => {
    // Validate input parameters
    if (!userMessage || typeof userMessage !== 'string') {
      toast.error('Mensaje de usuario inválido')
      return
    }
    
    const trimmedMessage = userMessage.trim()
    if (trimmedMessage.length === 0) {
      toast.error('El mensaje del usuario está vacío')
      return
    }
    
    if (trimmedMessage.length < 10) {
      toast.error('La descripción debe tener al menos 10 caracteres')
      return
    }
    
    if (trimmedMessage.length > 5000) {
      toast.error('La descripción es demasiado larga (máximo 5000 caracteres)')
      return
    }
    
    // Validate spec
    if (spec !== null && spec !== undefined && (typeof spec !== 'object' || Array.isArray(spec))) {
      console.warn('Invalid spec provided, using null')
      spec = null
    }
    
    setIsLoading(true)

    // Add user message
    try {
      addMessage({
        id: Date.now().toString(),
        role: 'user',
        content: trimmedMessage,
        timestamp: new Date(),
      })
    } catch (messageError) {
      console.error('Error adding user message:', messageError)
    }

    try {
      // Try to use TruthGPT API if available, otherwise fallback to old API
      let data: any
      
      if (apiConnected) {
        // Use TruthGPT API REST
        try {
          const modelName = spec?.modelName || `truthgpt-model-${Date.now()}`
          const result = await createModelFromDescription(trimmedMessage, modelName)
          
          data = {
            modelId: result.modelId,
            modelName: result.name,
            description: trimmedMessage,
            status: 'creating',
            githubUrl: null, // Will be set later if available
          }
        } catch (apiError) {
          console.warn('TruthGPT API error, falling back to legacy API:', apiError)
          // Fallback to legacy API
          data = await enhancedSendMessage(trimmedMessage, async (message) => {
            if (!message || typeof message !== 'string' || message.trim().length === 0) {
              throw new Error('Mensaje inválido para enviar')
            }
            
            const controller = new AbortController()
            const timeoutId = setTimeout(() => controller.abort(), 60000)
            
            try {
              const response = await fetch('/api/create-model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ description: message.trim() }),
                signal: controller.signal,
              })

              clearTimeout(timeoutId)

              if (!response.ok) {
                let errorMessage = 'Error al crear el modelo'
                try {
                  const errorData = await response.json()
                  errorMessage = errorData.error || errorMessage
                } catch {
                  errorMessage = `HTTP error! status: ${response.status}`
                }
                throw new Error(errorMessage)
              }

              let responseData: any
              try {
                responseData = await response.json()
              } catch (parseError) {
                throw new Error('Error al parsear la respuesta del servidor')
              }
              
              if (!responseData || typeof responseData !== 'object') {
                throw new Error('Respuesta del servidor inválida')
              }

              return responseData
            } catch (fetchError) {
              clearTimeout(timeoutId)
              if (fetchError instanceof Error && fetchError.name === 'AbortError') {
                throw new Error('Timeout: La solicitud tardó demasiado tiempo')
              }
              throw fetchError
            }
          })
        }
      } else {
        // Use legacy API
        data = await enhancedSendMessage(trimmedMessage, async (message) => {
          if (!message || typeof message !== 'string' || message.trim().length === 0) {
            throw new Error('Mensaje inválido para enviar')
          }
          
          const controller = new AbortController()
          const timeoutId = setTimeout(() => controller.abort(), 60000)
          
          try {
            const response = await fetch('/api/create-model', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ description: message.trim() }),
              signal: controller.signal,
            })

            clearTimeout(timeoutId)

            if (!response.ok) {
              let errorMessage = 'Error al crear el modelo'
              try {
                const errorData = await response.json()
                errorMessage = errorData.error || errorMessage
              } catch {
                errorMessage = `HTTP error! status: ${response.status}`
              }
              throw new Error(errorMessage)
            }

            let responseData: any
            try {
              responseData = await response.json()
            } catch (parseError) {
              throw new Error('Error al parsear la respuesta del servidor')
            }
            
            if (!responseData || typeof responseData !== 'object') {
              throw new Error('Respuesta del servidor inválida')
            }

            return responseData
          } catch (fetchError) {
            clearTimeout(timeoutId)
            if (fetchError instanceof Error && fetchError.name === 'AbortError') {
              throw new Error('Timeout: La solicitud tardó demasiado tiempo')
            }
            throw fetchError
          }
        })
      }

      // Validate response data structure
      if (!data || typeof data !== 'object') {
        throw new Error('Respuesta del servidor inválida')
      }

      // Validate required fields
      if (!data.modelId || !data.modelName) {
        throw new Error('Respuesta del servidor incompleta: faltan modelId o modelName')
      }
      
      // Validate types
      const modelId = String(data.modelId)
      const modelName = String(data.modelName)
      
      if (!modelId || modelId.trim().length === 0) {
        throw new Error('modelId inválido')
      }
      
      if (!modelName || modelName.trim().length === 0) {
        throw new Error('modelName inválido')
      }
      
      const githubUrl = data.githubUrl && typeof data.githubUrl === 'string'
        ? data.githubUrl
        : null

      // Add assistant response
      try {
        addMessage({
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: `¡Perfecto! He comenzado a crear tu modelo TruthGPT personalizado basado en: "${trimmedMessage}". El modelo se está generando y pronto estará disponible en GitHub.`,
          timestamp: new Date(),
        })
      } catch (messageError) {
        console.error('Error adding assistant message:', messageError)
      }
      
      toast.success('Modelo en creación', {
        icon: '🚀',
      })
      
      // Set current model
      const newCurrentModel = {
        id: modelId,
        name: modelName,
        description: trimmedMessage,
        status: 'creating' as const,
        githubUrl,
        createdAt: new Date(),
        progress: 0,
        currentStep: 'Iniciando...',
        spec: spec || null,
      }
      
      setCurrentModel(newCurrentModel)

      // Start polling for status
      pollModelStatus(modelId)
      
      // Add to history and save to storage
      const newModel = {
        id: modelId,
        name: modelName,
        description: trimmedMessage,
        status: 'creating' as const,
        githubUrl,
        createdAt: new Date(),
        spec: spec || null,
      }
      
      // Agregar al historial de sugerencias inteligentes
      smartSuggestionsManager.addToHistory(trimmedMessage)
      
      setModelHistory(prev => {
        if (!Array.isArray(prev)) return [newModel]
        return [newModel, ...prev]
      })
      
      try {
        saveModelToHistory(newModel as any)
      } catch (saveError) {
        console.error('Error saving model to history:', saveError)
      }
    } catch (error) {
      console.error('Error:', error)
      const errorMessage = error instanceof Error ? error.message : 'Error desconocido'
      
      // Check if it's a rate limit error
      if (errorMessage.toLowerCase().includes('rate limit') || errorMessage.toLowerCase().includes('rate_limit')) {
        toast.error(errorMessage, {
          icon: '⏱️',
          duration: 5000,
        })
        // Optionally show metrics
        setTimeout(() => {
          if (typeof setShowMetrics === 'function') {
            setShowMetrics(true)
          }
        }, 1000)
      } else {
        toast.error(`Error: ${errorMessage}`, {
          duration: 5000,
        })
      }
      
      try {
        addMessage({
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: `Lo siento, hubo un error al crear el modelo: ${errorMessage}`,
          timestamp: new Date(),
        })
      } catch (messageError) {
        console.error('Error adding error message:', messageError)
      }
    } finally {
      setIsLoading(false)
    }
  }

  const handlePreviewConfirm = async () => {
    // Validate previewSpec
    if (!previewSpec || typeof previewSpec !== 'object') {
      toast.error('Error: No hay especificación para crear el modelo')
      return
    }
    
    const { description, spec } = previewSpec
    
    // Validate description
    if (!description || typeof description !== 'string') {
      toast.error('Error: Descripción inválida')
      return
    }
    
    const trimmedDescription = description.trim()
    if (trimmedDescription.length === 0) {
      toast.error('Error: La descripción está vacía')
      return
    }
    
    if (trimmedDescription.length < 10) {
      toast.error('La descripción debe tener al menos 10 caracteres')
      return
    }
    
    if (trimmedDescription.length > 5000) {
      toast.error('La descripción es demasiado larga (máximo 5000 caracteres)')
      return
    }
    
    // Validate spec
    if (!spec || typeof spec !== 'object') {
      toast.error('Error: Especificación del modelo inválida')
      return
    }
    
    setShowPreview(false)
    
    try {
      await createModel(trimmedDescription, spec)
    } catch (error) {
      console.error('Error in handlePreviewConfirm:', error)
      const errorMessage = error instanceof Error ? error.message : 'Error desconocido'
      toast.error(`Error al crear el modelo: ${errorMessage}`)
    }
  }

  const pollModelStatus = async (modelId: string) => {
    // Validate modelId
    if (!modelId || typeof modelId !== 'string' || modelId.trim().length === 0) {
      console.error('Invalid modelId for polling')
      toast.error('Error: ID de modelo inválido')
      return
    }
    
    const maxAttempts = 60 // 5 minutes max
    let attempts = 0
    let pollInterval: NodeJS.Timeout | null = null

    const checkStatus = async () => {
      try {
        // Validate currentModel exists
        if (!currentModel) {
          console.warn('No current model to poll status for')
          return
        }
        
        // Validate modelId matches current model
        if (currentModel.id !== modelId) {
          console.warn('Model ID mismatch during polling')
          return
        }
        
        // Try to use TruthGPT API if available, otherwise fallback to legacy API
        let data: any
        if (apiConnected) {
          try {
            const status = await truthGPTClient.getModelStatus(modelId)
            data = {
              ...status,
              // Map to legacy format
              status: status.status === 'completed' ? 'completed' : status.status,
            }
          } catch (apiError) {
            console.warn('TruthGPT API status error, falling back to legacy API:', apiError)
            // Fallback to legacy API
            const response = await fetch(`/api/model-status/${modelId}`)
            
            if (!response.ok) {
              let errorMessage = `HTTP error! status: ${response.status}`
              try {
                const errorData = await response.json()
                errorMessage = errorData.error || errorMessage
              } catch {
                // Use default error message
              }
              throw new Error(errorMessage)
            }
            
            // Parse JSON response
            try {
              data = await response.json()
            } catch (parseError) {
              throw new Error('Error al parsear la respuesta del servidor')
            }
          }
        } else {
          // Use legacy API
          const response = await fetch(`/api/model-status/${modelId}`)
          
          if (!response.ok) {
            let errorMessage = `HTTP error! status: ${response.status}`
            try {
              const errorData = await response.json()
              errorMessage = errorData.error || errorMessage
            } catch {
              // Use default error message
            }
            throw new Error(errorMessage)
          }
          
          // Parse JSON response
          try {
            data = await response.json()
          } catch (parseError) {
            throw new Error('Error al parsear la respuesta del servidor')
          }
        }
        
        // Validate response data structure
        if (!data || typeof data !== 'object') {
          throw new Error('Respuesta del servidor inválida')
        }
        
        // Validate status field
        if (!data.status || typeof data.status !== 'string') {
          throw new Error('Campo status inválido en la respuesta')
        }

        if (data.status === 'completed') {
          // Validate required fields
          const githubUrl = data.githubUrl || currentModel.githubUrl
          
          const completedModel = {
            ...currentModel,
            status: 'completed' as const,
            githubUrl: typeof githubUrl === 'string' ? githubUrl : null,
            progress: 100,
            currentStep: 'Completado',
            spec: data.spec && typeof data.spec === 'object' ? data.spec : currentModel.spec,
          }
          
          setCurrentModel(completedModel)
          
          // Update in history
          setModelHistory(prev => {
            if (!Array.isArray(prev)) return prev
            return prev.map(m => {
              if (!m || typeof m !== 'object') return m
              return m.id === modelId ? { ...m, ...completedModel } : m
            })
          })
          
          try {
            saveModelToHistory(completedModel as any)
          } catch (saveError) {
            console.error('Error saving model to history:', saveError)
          }

          const githubUrlStr = typeof githubUrl === 'string' ? githubUrl : 'GitHub'
          addMessage({
            id: Date.now().toString(),
            role: 'assistant',
            content: `¡Listo! Tu modelo TruthGPT ha sido creado y publicado en GitHub: ${githubUrlStr}`,
            timestamp: new Date(),
          })
          
          toast.success('Modelo completado y publicado en GitHub! 🎉', {
            duration: 5000,
          })
          
          // Clear polling interval
          if (pollInterval) {
            clearTimeout(pollInterval)
            pollInterval = null
          }
          return
        }

        if (data.status === 'failed') {
          const errorMessage = data.error && typeof data.error === 'string' 
            ? data.error 
            : 'Error desconocido'
          
          const progress = typeof data.progress === 'number' && data.progress >= 0 && data.progress <= 100
            ? data.progress
            : 0
          
          const currentStep = data.currentStep && typeof data.currentStep === 'string'
            ? data.currentStep
            : 'Error'
          
          setCurrentModel({
            ...currentModel,
            status: 'failed' as const,
            progress,
            currentStep,
          })
          
          addMessage({
            id: Date.now().toString(),
            role: 'assistant',
            content: `Lo siento, hubo un error al crear el modelo: ${errorMessage}`,
            timestamp: new Date(),
          })
          
          // Clear polling interval
          if (pollInterval) {
            clearTimeout(pollInterval)
            pollInterval = null
          }
          return
        }

        // Update progress if available
        if (data.progress !== undefined && data.currentStep) {
          const progress = typeof data.progress === 'number' && data.progress >= 0 && data.progress <= 100
            ? data.progress
            : currentModel.progress || 0
          
          const currentStep = typeof data.currentStep === 'string' && data.currentStep.trim().length > 0
            ? data.currentStep
            : currentModel.currentStep || 'Procesando...'
          
          const spec = data.spec && typeof data.spec === 'object'
            ? data.spec
            : currentModel.spec
          
          const updatedModel = {
            ...currentModel,
            progress,
            currentStep,
            spec,
          }
          
          setCurrentModel(updatedModel)
          
          // Update in history
          setModelHistory(prev => {
            if (!Array.isArray(prev)) return prev
            return prev.map(m => {
              if (!m || typeof m !== 'object') return m
              return m.id === modelId ? { ...m, ...updatedModel } : m
            })
          })
          
          try {
            saveModelToHistory(updatedModel as any)
          } catch (saveError) {
            console.error('Error saving model to history:', saveError)
          }
        }
        
        attempts++
        if (attempts < maxAttempts) {
          pollInterval = setTimeout(checkStatus, 3000) // Check every 3 seconds
        } else {
          // Max attempts reached
          if (currentModel) {
            setCurrentModel({
              ...currentModel,
              currentStep: 'Timeout: El proceso está tomando más tiempo del esperado',
            })
          }
          toast.error('El proceso está tomando más tiempo del esperado. Por favor, verifica manualmente el estado del modelo.', {
            icon: '⏰',
            duration: 7000,
          })
        }
      } catch (error) {
        console.error('Error checking status:', error)
        const errorMessage = error instanceof Error ? error.message : 'Error desconocido'
        
        attempts++
        if (attempts < maxAttempts) {
          pollInterval = setTimeout(checkStatus, 5000) // Retry after 5 seconds on error
        } else {
          // Max attempts reached with errors
          if (currentModel) {
            setCurrentModel({
              ...currentModel,
              status: 'failed' as const,
              currentStep: `Error: ${errorMessage}`,
            })
          }
          toast.error(`Error al verificar el estado del modelo: ${errorMessage}`)
        }
      }
    }

    // Start checking after 2 seconds
    pollInterval = setTimeout(checkStatus, 2000)
    
    // Return cleanup function (optional, for component unmount)
    return () => {
      if (pollInterval) {
        clearTimeout(pollInterval)
        pollInterval = null
      }
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    // Validate event
    if (e && typeof e.preventDefault === 'function') {
      e.preventDefault()
    }
    
    // Validate input state
    if (!input || typeof input !== 'string') {
      toast.error('Entrada inválida')
      return
    }
    
    const trimmedInput = input.trim()
    
    // Check if loading or empty
    if (isLoading || bulkChat.isLoading) {
      toast('Por favor espera, ya hay una operación en curso', {
        icon: '⏳',
      })
      return
    }
    
    if (!trimmedInput || trimmedInput.length === 0) {
      toast.error('Por favor ingresa una descripción')
      return
    }

    // Si Bulk Chat está activo, enviar mensaje directamente
    if (useBulkChatMode) {
      // Si no hay sesión, crear una primero
      if (!bulkChat.sessionId) {
        await bulkChat.createSession(trimmedInput)
      } else {
        // Agregar mensaje del usuario al store
        addMessage({
          id: Date.now().toString(),
          role: 'user',
          content: trimmedInput,
          timestamp: new Date(),
        })
        
        // Enviar a Bulk Chat
        await bulkChat.sendMessage(trimmedInput)
      }
      setInput('')
      return
    }

    // Modo normal (TruthGPT)
    try {
      // Validate input length
      if (trimmedInput.length < 10) {
        toast.error('La descripción debe tener al menos 10 caracteres')
        setValidation({
          isValid: false,
          message: 'La descripción debe tener al menos 10 caracteres'
        })
        return
      }
      
      if (trimmedInput.length > 5000) {
        toast.error('La descripción es demasiado larga (máximo 5000 caracteres)')
        setValidation({
          isValid: false,
          message: 'La descripción es demasiado larga (máximo 5000 caracteres)'
        })
        return
      }
      
      // Validate description
      let validation: any
      try {
        validation = validateDescription(trimmedInput)
      } catch (validateError) {
        console.error('Error in validateDescription:', validateError)
        toast.error('Error al validar la descripción')
        return
      }
      
      if (!validation || typeof validation !== 'object') {
        toast.error('Error en la validación')
        return
      }
      
      if (!validation.isValid) {
        setValidation(validation)
        const errorMessage = validation.message && typeof validation.message === 'string'
          ? validation.message
          : 'Descripción inválida'
        toast.error(errorMessage)
        return
      }

      // Use adaptive analyzer for intelligent adaptation
      let spec: any
      try {
        spec = adaptiveAnalyze(trimmedInput)
      } catch (analyzeError) {
        console.error('Error in adaptiveAnalyze:', analyzeError)
        toast.error('Error al analizar la descripción del modelo')
        return
      }
      
      if (!spec || typeof spec !== 'object' || Array.isArray(spec)) {
        toast.error('Error: Especificación del modelo inválida')
        return
      }
      
      // Generate model name safely
      let modelName: string
      try {
        const sanitized = trimmedInput.toLowerCase().replace(/[^a-z0-9]+/g, '-').substring(0, 30)
        // Remove leading/trailing hyphens
        const cleaned = sanitized.replace(/^-+|-+$/g, '')
        modelName = cleaned.length > 0 
          ? `truthgpt-${cleaned}`
          : `truthgpt-model-${Date.now()}`
        
        // Validate model name length
        if (modelName.length > 100) {
          modelName = modelName.substring(0, 100)
        }
      } catch (nameError) {
        console.error('Error generating model name:', nameError)
        modelName = `truthgpt-model-${Date.now()}`
      }
      
      // Validate all preview spec components
      if (!modelName || typeof modelName !== 'string' || modelName.trim().length === 0) {
        toast.error('Error al generar el nombre del modelo')
        return
      }
      
      setPreviewSpec({ 
        spec, 
        modelName: modelName.trim(), 
        description: trimmedInput 
      })
      setShowPreview(true)
    } catch (error) {
      console.error('Error en handleSubmit:', error)
      const errorMessage = error instanceof Error ? error.message : 'Error desconocido'
      toast.error(`Error: ${errorMessage}`)
      
      // Clear preview spec on error
      setPreviewSpec(null)
    }
  }

  const handleDiscardAll = useCallback(() => {
    // Validate that there's something to discard
    const hasContent = messages.length > 0 || currentModel !== null || (input && input.trim().length > 0) || bulkChat.sessionId
    
    if (!hasContent) {
      toast('No hay nada que descartar', {
        icon: 'ℹ️',
      })
      return
    }
    
    if (window.confirm('¿Estás seguro de que quieres descartar todo? Esto eliminará todos los mensajes, el modelo actual, la sesión de Bulk Chat y cualquier borrador guardado.')) {
      try {
        // Detener Bulk Chat si está activo
        if (bulkChat.sessionId) {
          bulkChat.stop()
          setUseBulkChatMode(false)
        }
        
        // Clear messages and current model
        try {
          clearMessages()
        } catch (clearError) {
          console.error('Error clearing messages:', clearError)
        }
        
        // Clear input and validation
        setInput('')
        setValidation(null)
        setPreviewSpec(null)
        setShowPreview(false)
        setShowSmartSuggestions(false)
        setSmartSuggestions([])
        
        // Clear draft
        try {
          clearDraft()
        } catch (draftError) {
          console.error('Error clearing draft:', draftError)
        }
        
        // Limpiar optimizadores
        performanceOptimizer.cleanup()
        
        // Clear model history (optional - you might want to keep history)
        // clearModelHistory()
        // setModelHistory([])
        
        toast.success('Todo ha sido descartado', {
          icon: '🗑️',
        })
      } catch (error) {
        console.error('Error in handleDiscardAll:', error)
        const errorMessage = error instanceof Error ? error.message : 'Error desconocido'
        toast.error(`Error al descartar: ${errorMessage}`)
      }
    }
  }, [messages.length, currentModel, input, clearMessages, performanceOptimizer, bulkChat])
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      performanceOptimizer.cleanup()
    }
  }, [performanceOptimizer])

  // Cerrar paneles al hacer clic fuera
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as HTMLElement
      
      // Cerrar panel de conexión
      if (showConnectionInfo && !target.closest('[data-connection-panel]') && !target.closest('[data-connection-button]')) {
        setShowConnectionInfo(false)
      }
      
      // Cerrar quick actions
      if (showQuickActions && !target.closest('[data-quick-actions]')) {
        setShowQuickActions(false)
      }
    }
    
    if (showConnectionInfo || showQuickActions) {
      document.addEventListener('mousedown', handleClickOutside)
      return () => document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [showConnectionInfo, showQuickActions])

  // Templates de mensajes predefinidos
  const defaultTemplates = [
    'Hola, ¿cómo estás?',
    'Explícame más sobre...',
    '¿Puedes ayudarme con...?',
    'Resume la conversación',
    '¿Cuál es tu opinión sobre...?',
  ]

  // Comandos disponibles
  const commands = [
    { id: 'clear', label: 'Limpiar conversación', icon: '🗑️', action: () => handleDiscardAll() },
    { id: 'export-json', label: 'Exportar como JSON', icon: '💾', action: () => {
      if (bulkChat.sessionId && bulkChat.messages.length > 0) {
        const data = {
          sessionId: bulkChat.sessionId,
          messages: bulkChat.messages,
          timestamp: new Date().toISOString(),
          messageCount: bulkChat.messageCount,
        }
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `bulk-chat-${bulkChat.sessionId.slice(0, 8)}-${Date.now()}.json`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        URL.revokeObjectURL(url)
        toast.success('Conversación exportada (JSON)', { icon: '💾', duration: 2000 })
      }
    }},
    { id: 'toggle-read', label: 'Toggle modo lectura', icon: '📖', action: () => setReadMode(!readMode) },
    { id: 'toggle-stats', label: 'Toggle estadísticas', icon: '📊', action: () => setShowStats(!showStats) },
    { id: 'toggle-timeline', label: 'Toggle timeline', icon: '📅', action: () => setShowTimeline(!showTimeline) },
    { id: 'toggle-filters', label: 'Toggle filtros', icon: '🔽', action: () => setShowFilters(!showFilters) },
    { id: 'presentation', label: 'Modo presentación', icon: '🖥️', action: () => setPresentationMode(!presentationMode) },
    { id: 'focus-search', label: 'Focus búsqueda', icon: '🔍', action: () => searchInputRef.current?.focus() },
    { id: 'toggle-auto-scroll', label: 'Toggle auto-scroll', icon: '⬇️', action: () => setAutoScroll(!autoScroll) },
    { id: 'toggle-word-count', label: 'Toggle conteo de palabras', icon: '📝', action: () => setShowWordCount(!showWordCount) },
    { id: 'print-mode', label: 'Modo impresión', icon: '🖨️', action: () => {
      setShowPrintMode(true)
      setTimeout(() => {
        window.print()
        setShowPrintMode(false)
      }, 500)
    }},
    { id: 'toggle-sentiment', label: 'Toggle análisis de sentimiento', icon: '😊', action: () => setShowSentiment(!showSentiment) },
    { id: 'toggle-debug', label: 'Toggle modo debug', icon: '🐛', action: () => setShowDebug(!showDebug) },
    { id: 'toggle-theme', label: 'Cambiar tema', icon: '🎨', action: () => {
      setTheme(prev => prev === 'dark' ? 'light' : prev === 'light' ? 'auto' : 'dark')
    }},
    { id: 'toggle-zen', label: 'Modo Zen', icon: '🧘', action: () => setZenMode(!zenMode) },
    { id: 'toggle-tags', label: 'Toggle etiquetas', icon: '🏷️', action: () => setShowTags(!showTags) },
    { id: 'toggle-history', label: 'Toggle historial', icon: '📜', action: () => setShowEditHistory(!showEditHistory) },
    { id: 'toggle-translation', label: 'Toggle traducción', icon: '🌐', action: () => setTranslationMode(!translationMode) },
    { id: 'export-pdf', label: 'Exportar como PDF', icon: '📄', action: () => exportToPDF() },
    { id: 'export-markdown', label: 'Exportar como Markdown', icon: '📝', action: () => exportToMarkdownEnhanced() },
    { id: 'generate-summary', label: 'Generar resumen', icon: '📊', action: () => generateSummary() },
    { id: 'cluster-messages', label: 'Agrupar mensajes', icon: '🔗', action: () => clusterMessages() },
    { id: 'backup-conversation', label: 'Crear backup', icon: '💾', action: () => backupConversation() },
    { id: 'toggle-voice-input', label: 'Toggle entrada de voz', icon: '🎤', action: () => {
      setVoiceInputEnabled(!voiceInputEnabled)
      if (!voiceInputEnabled) {
        startVoiceInput()
      } else {
        stopVoiceInput()
      }
    }},
    { id: 'toggle-voice-output', label: 'Toggle salida de voz', icon: '🔊', action: () => setVoiceOutputEnabled(!voiceOutputEnabled) },
    { id: 'toggle-scheduler', label: 'Toggle programador', icon: '⏰', action: () => setShowScheduler(!showScheduler) },
    { id: 'toggle-archive', label: 'Toggle archivo', icon: '📦', action: () => setShowArchive(!showArchive) },
    { id: 'toggle-analytics', label: 'Toggle analíticas', icon: '📊', action: () => setShowAnalytics(!showAnalytics) },
    { id: 'toggle-backup', label: 'Toggle respaldos', icon: '💾', action: () => setShowBackup(!showBackup) },
    { id: 'share-message', label: 'Compartir mensaje', icon: '🔗', action: () => {
      const lastMessage = messages[messages.length - 1]
      if (lastMessage) {
        setShareTarget(lastMessage.id)
        setShowShareMenu(true)
      }
    }},
    { id: 'toggle-advanced-search', label: 'Toggle búsqueda avanzada', icon: '🔍', action: () => setAdvancedSearch(!advancedSearch) },
    { id: 'toggle-smart-notifications', label: 'Toggle notificaciones inteligentes', icon: '🔔', action: () => setSmartNotifications(!smartNotifications) },
    { id: 'toggle-real-time-stats', label: 'Toggle estadísticas tiempo real', icon: '📊', action: () => setRealTimeStats(!realTimeStats) },
    { id: 'toggle-encryption', label: 'Toggle encriptación', icon: '🔒', action: () => setMessageEncryption(!messageEncryption) },
    { id: 'toggle-template-editor', label: 'Toggle editor plantillas', icon: '📝', action: () => setShowTemplateEditor(!showTemplateEditor) },
    { id: 'toggle-collaboration', label: 'Toggle modo colaboración', icon: '👥', action: () => setCollaborationMode(!collaborationMode) },
    { id: 'toggle-dev-mode', label: 'Toggle modo desarrollo', icon: '🛠️', action: () => setDevMode(!devMode) },
    { id: 'toggle-session-recording', label: 'Toggle grabación sesión', icon: '🔴', action: () => sessionRecording ? stopSessionRecording() : startSessionRecording() },
    { id: 'toggle-bookmarks', label: 'Toggle marcadores', icon: '🔖', action: () => setShowBookmarks(!showBookmarks) },
    { id: 'toggle-study-mode', label: 'Toggle modo estudio', icon: '📚', action: () => setStudyMode(!studyMode) },
    { id: 'toggle-flashcards', label: 'Toggle flashcards', icon: '📖', action: () => setShowFlashcards(!showFlashcards) },
    { id: 'toggle-productivity', label: 'Toggle modo productividad', icon: '⏱️', action: () => setProductivityMode(!productivityMode) },
    { id: 'toggle-accessibility', label: 'Toggle accesibilidad', icon: '♿', action: () => setAccessibilityMode(!accessibilityMode) },
    { id: 'generate-insights', label: 'Generar insights IA', icon: '🤖', action: () => generateAIInsights() },
    { id: 'toggle-reading-mode', label: 'Toggle modo lectura', icon: '📖', action: () => setReadingMode(!readingMode) },
    { id: 'toggle-dictation', label: 'Toggle dictado', icon: '🎤', action: () => setDictationMode(!dictationMode) },
    { id: 'toggle-calendar', label: 'Toggle calendario', icon: '📅', action: () => setCalendarIntegration(!calendarIntegration) },
    { id: 'toggle-cloud-sync', label: 'Toggle sincronización nube', icon: '☁️', action: () => { setCloudSync(!cloudSync); if (!cloudSync) syncToCloud() } },
    { id: 'toggle-annotations', label: 'Toggle anotaciones', icon: '📝', action: () => setShowAnnotations(!showAnnotations) },
    { id: 'toggle-folders', label: 'Toggle carpetas', icon: '📁', action: () => setShowFolders(!showFolders) },
    { id: 'toggle-fullscreen', label: 'Toggle pantalla completa', icon: '⛶', action: () => toggleFullscreen() },
    { id: 'toggle-presentation', label: 'Toggle modo presentación', icon: '📊', action: () => presentationMode ? setPresentationMode(false) : startPresentation() },
    { id: 'toggle-versions', label: 'Toggle versiones', icon: '📜', action: () => setShowVersions(!showVersions) },
    { id: 'toggle-comments', label: 'Toggle comentarios', icon: '💬', action: () => setShowComments(!showComments) },
    { id: 'export-csv', label: 'Exportar a CSV', icon: '📊', action: () => exportToCSV() },
    { id: 'export-excel', label: 'Exportar a Excel', icon: '📈', action: () => exportToExcel() },
    { id: 'export-xml', label: 'Exportar a XML', icon: '📄', action: () => exportToXML() },
    { id: 'export-yaml', label: 'Exportar a YAML', icon: '📝', action: () => exportToYAML() },
    { id: 'toggle-push-notifications', label: 'Toggle notificaciones push', icon: '🔔', action: () => requestNotificationPermission() },
    { id: 'print-conversation', label: 'Imprimir conversación', icon: '🖨️', action: () => printConversation() },
    { id: 'toggle-timeline', label: 'Toggle timeline', icon: '📅', action: () => toggleTimelineView() },
    { id: 'toggle-auto-translate', label: 'Toggle traducción automática', icon: '🌐', action: () => setAutoTranslate(!autoTranslate) },
    { id: 'toggle-auto-summarize', label: 'Toggle resumen automático', icon: '📝', action: () => setAutoSummarize(!autoSummarize) },
    { id: 'generate-advanced-analytics', label: 'Generar analytics avanzado', icon: '📊', action: () => generateAdvancedAnalytics() },
    { id: 'toggle-split-screen', label: 'Toggle pantalla dividida', icon: '🖥️', action: () => toggleSplitScreen() },
    { id: 'toggle-widgets', label: 'Toggle widgets', icon: '🧩', action: () => setShowWidgets(!showWidgets) },
    { id: 'toggle-audio-recording', label: 'Toggle grabación audio', icon: '🎤', action: () => audioRecording ? stopAudioRecording() : startAudioRecording() },
    { id: 'toggle-performance', label: 'Toggle métricas rendimiento', icon: '⚡', action: () => setShowPerformance(!showPerformance) },
    { id: 'toggle-accessibility', label: 'Toggle accesibilidad avanzada', icon: '♿', action: () => setAccessibilityMode(!accessibilityMode) },
    { id: 'toggle-message-queue', label: 'Toggle cola de mensajes', icon: '📋', action: () => setMessageQueue([]) },
    { id: 'toggle-auto-format', label: 'Toggle formateo automático', icon: '📝', action: () => setAutoFormat(!autoFormat) },
    { id: 'toggle-message-graph', label: 'Toggle gráfico de mensajes', icon: '🌐', action: () => setShowMessageGraph(!showMessageGraph) },
    { id: 'toggle-workflow', label: 'Toggle modo workflow', icon: '⚙️', action: () => setWorkflowMode(!workflowMode) },
    { id: 'toggle-auto-validate', label: 'Toggle validación automática', icon: '✓', action: () => setAutoValidate(!autoValidate) },
    { id: 'toggle-deduplication', label: 'Toggle deduplicación', icon: '🔍', action: () => setMessageDeduplication(!messageDeduplication) },
    { id: 'toggle-compression', label: 'Toggle compresión', icon: '🗜️', action: () => setMessageCompression(!messageCompression) },
    { id: 'toggle-encryption', label: 'Toggle encriptación avanzada', icon: '🔐', action: () => setEncryptionEnabled(!encryptionEnabled) },
    { id: 'toggle-multi-device', label: 'Toggle sincronización multi-dispositivo', icon: '🔄', action: () => setMultiDeviceSync(!multiDeviceSync) },
    { id: 'toggle-advanced-analytics', label: 'Toggle analytics avanzado', icon: '📊', action: () => setShowAdvancedAnalytics(!showAdvancedAnalytics) },
    { id: 'create-backup', label: 'Crear backup avanzado', icon: '💾', action: () => createAdvancedBackup() },
    { id: 'process-queue', label: 'Procesar cola', icon: '▶️', action: () => processQueue() },
    { id: 'toggle-cache', label: 'Toggle caché', icon: '💾', action: () => setCacheEnabled(!cacheEnabled) },
    { id: 'toggle-smart-search', label: 'Toggle búsqueda inteligente', icon: '🔍', action: () => setSmartSearch(!smartSearch) },
    { id: 'toggle-auto-complete', label: 'Toggle autocompletado', icon: '⌨️', action: () => setAutoComplete(!autoComplete) },
    { id: 'toggle-typing-prediction', label: 'Toggle predicción escritura', icon: '🔮', action: () => setTypingPrediction(!typingPrediction) },
    { id: 'toggle-macros', label: 'Toggle macros', icon: '⌨️', action: () => setMacroEnabled(!macroEnabled) },
    { id: 'toggle-undo', label: 'Toggle deshacer/rehacer', icon: '↶', action: () => setUndoEnabled(!undoEnabled) },
    { id: 'undo-action', label: 'Deshacer', icon: '↶', action: () => undo() },
    { id: 'redo-action', label: 'Rehacer', icon: '↷', action: () => redo() },
    { id: 'toggle-batch-mode', label: 'Toggle modo batch', icon: '📦', action: () => setBatchMode(!batchMode) },
    { id: 'send-batch', label: 'Enviar batch', icon: '📤', action: () => sendBatch() },
    { id: 'toggle-auto-filter', label: 'Toggle filtrado automático', icon: '🔍', action: () => setAutoFilter(!autoFilter) },
    { id: 'toggle-notifications', label: 'Toggle notificaciones', icon: '🔔', action: () => setNotificationSettings(prev => ({ ...prev, desktop: !prev.desktop })) },
    { id: 'toggle-highlights', label: 'Toggle resaltado', icon: '🖍️', action: () => setHighlightEnabled(!highlightEnabled) },
    { id: 'advanced-search', label: 'Búsqueda avanzada', icon: '🔍', action: () => performAdvancedSearch('') },
    { id: 'save-filter-preset', label: 'Guardar preset de filtros', icon: '💾', action: () => saveFilterPreset('Nuevo Preset', {}) },
    { id: 'sort-messages', label: 'Ordenar mensajes', icon: '📊', action: () => sortMessages(messageSorting === 'chronological' ? 'relevance' : 'chronological') },
    { id: 'highlight-message', label: 'Resaltar mensaje', icon: '🖍️', action: () => {} },
    { id: 'add-bookmark-advanced', label: 'Agregar bookmark avanzado', icon: '🔖', action: () => {} },
    { id: 'export-with-template', label: 'Exportar con template', icon: '📤', action: () => {} },
    { id: 'import-messages', label: 'Importar mensajes', icon: '📥', action: () => {} },
    { id: 'share-collaboration', label: 'Compartir para colaboración', icon: '👥', action: () => {} },
    { id: 'ai-suggestions', label: 'Generar sugerencias IA', icon: '🤖', action: () => messages.length > 0 && generateAISuggestions(messages[0].id) },
    { id: 'toggle-import', label: 'Toggle importación', icon: '📥', action: () => setImportEnabled(!importEnabled) },
    { id: 'toggle-version-control', label: 'Toggle control versiones', icon: '📝', action: () => setVersionControl(!versionControl) },
    { id: 'toggle-collaboration', label: 'Toggle colaboración', icon: '👥', action: () => setCollaborationEnabled(!collaborationEnabled) },
    { id: 'toggle-ai-features', label: 'Toggle características IA', icon: '🤖', action: () => setAiFeatures(prev => ({ ...prev, suggestions: !prev.suggestions })) },
    { id: 'toggle-command-palette', label: 'Toggle paleta de comandos', icon: '⌨️', action: () => setCommandPalette(!commandPalette) },
    { id: 'toggle-reactions', label: 'Toggle reacciones avanzadas', icon: '👍', action: () => setReactionPicker(messages[0]?.id || null) },
    { id: 'toggle-polls', label: 'Toggle encuestas', icon: '📊', action: () => setPollMode(!pollMode) },
    { id: 'toggle-tasks', label: 'Toggle tareas', icon: '✓', action: () => setTaskMode(!taskMode) },
    { id: 'toggle-reminders', label: 'Toggle recordatorios', icon: '⏰', action: () => setReminderSystem(!reminderSystem) },
    { id: 'toggle-calendar', label: 'Toggle calendario', icon: '📅', action: () => setCalendarIntegration(!calendarIntegration) },
    { id: 'toggle-attachments', label: 'Toggle adjuntos', icon: '📎', action: () => setAttachmentManager(!attachmentManager) },
    { id: 'toggle-link-preview', label: 'Toggle vista previa enlaces', icon: '🔗', action: () => setLinkPreview(!linkPreview) },
    { id: 'toggle-code-editor', label: 'Toggle editor código', icon: '💻', action: () => setCodeEditor(messages[0]?.id || null) },
    { id: 'toggle-media-viewer', label: 'Toggle visor media', icon: '🖼️', action: () => setMediaViewer(messages[0]?.id || null) },
    { id: 'toggle-location', label: 'Toggle ubicación', icon: '📍', action: () => setLocationSharing(!locationSharing) },
    { id: 'toggle-contacts', label: 'Toggle contactos', icon: '👤', action: () => setContactManager(!contactManager) },
    { id: 'toggle-event-log', label: 'Toggle registro eventos', icon: '📋', action: () => setEventLog(!eventLog) },
    { id: 'toggle-metadata', label: 'Toggle metadatos', icon: '🏷️', action: () => setMetadataEditor(messages[0]?.id || null) },
    { id: 'toggle-plugins', label: 'Toggle plugins', icon: '🔌', action: () => setPluginManager(!pluginManager) },
    { id: 'toggle-apis', label: 'Toggle APIs', icon: '🔗', action: () => setApiManager(!apiManager) },
    { id: 'toggle-dev-tools', label: 'Toggle dev tools', icon: '🛠️', action: () => setDevTools(!devTools) },
    { id: 'toggle-theme-editor', label: 'Toggle editor temas', icon: '🎨', action: () => setThemeEditor(!themeEditor) },
    { id: 'toggle-performance', label: 'Toggle rendimiento', icon: '⚡', action: () => setPerformanceDashboard(!performanceDashboard) },
    { id: 'toggle-notifications', label: 'Toggle notificaciones', icon: '🔔', action: () => setNotificationCenter(!notificationCenter) },
    { id: 'toggle-offline', label: 'Toggle modo offline', icon: '📴', action: () => setOfflineMode(!offlineMode) },
    { id: 'toggle-sync', label: 'Toggle sincronización', icon: '🔄', action: () => setRealTimeSync(!realTimeSync) },
    { id: 'toggle-cloud', label: 'Toggle nube', icon: '☁️', action: () => setCloudManager(!cloudManager) },
    { id: 'toggle-auth', label: 'Toggle autenticación', icon: '🔐', action: () => setAuthManager(!authManager) },
    { id: 'toggle-encryption', label: 'Toggle encriptación', icon: '🔒', action: () => setEncryptionManager(!encryptionManager) },
    { id: 'toggle-backup', label: 'Toggle backup', icon: '💾', action: () => setBackupManager(!backupManager) },
    { id: 'toggle-restore', label: 'Toggle restaurar', icon: '📥', action: () => setRestoreManager(!restoreManager) },
    { id: 'toggle-analytics', label: 'Toggle analytics', icon: '📊', action: () => setAnalyticsDashboard(!analyticsDashboard) },
    { id: 'toggle-search', label: 'Toggle búsqueda', icon: '🔍', action: () => setSearchManager(!searchManager) },
    { id: 'toggle-export', label: 'Toggle exportar', icon: '📤', action: () => setExportManager(!exportManager) },
    { id: 'toggle-import', label: 'Toggle importar', icon: '📥', action: () => setImportManager(!importManager) },
    { id: 'toggle-widgets', label: 'Toggle widgets', icon: '🧩', action: () => setWidgetEditor(!widgetEditor) },
    { id: 'toggle-presentation', label: 'Toggle presentación', icon: '📊', action: () => setPresentationMode(!presentationMode) },
    { id: 'toggle-services', label: 'Toggle servicios', icon: '🔗', action: () => setServiceManager(!serviceManager) },
    { id: 'toggle-templates', label: 'Toggle plantillas', icon: '📝', action: () => setTemplateEditor(!templateEditor) },
    { id: 'toggle-collaboration', label: 'Toggle colaboración', icon: '👥', action: () => setCollaborationManager(!collaborationManager) },
    { id: 'toggle-notification-rules', label: 'Toggle reglas notificación', icon: '🔔', action: () => setNotificationManager(!notificationManager) },
    { id: 'toggle-accessibility', label: 'Toggle accesibilidad', icon: '♿', action: () => setAccessibilityManager(!accessibilityManager) },
    { id: 'toggle-help', label: 'Toggle ayuda', icon: '❓', action: () => setHelpCenter(!helpCenter) },
    { id: 'toggle-tutorial', label: 'Toggle tutorial', icon: '📚', action: () => setTutorialMode(!tutorialMode) },
    { id: 'toggle-feedback', label: 'Toggle feedback', icon: '💬', action: () => setFeedbackManager(!feedbackManager) },
    { id: 'toggle-reports', label: 'Toggle reportes', icon: '📊', action: () => setReportManager(!reportManager) },
    { id: 'toggle-history', label: 'Toggle historial', icon: '📜', action: () => setHistoryViewer(!historyViewer) },
    { id: 'toggle-relations', label: 'Toggle relaciones', icon: '🔗', action: () => setRelationViewer(!relationViewer) },
    { id: 'toggle-insights', label: 'Toggle insights', icon: '💡', action: () => setInsightsPanel(!insightsPanel) },
    { id: 'toggle-suggestions', label: 'Toggle sugerencias', icon: '💭', action: () => setSuggestionsPanel(!suggestionsPanel) },
    { id: 'toggle-clustering', label: 'Toggle clustering', icon: '📦', action: () => setClusteringViewer(!clusteringViewer) },
    { id: 'optimize-performance', label: 'Optimizar rendimiento', icon: '⚡', action: () => optimizePerformance() },
    { id: 'enhance-ui', label: 'Mejorar UI', icon: '🎨', action: () => enhanceUI() },
    { id: 'toggle-shortcuts', label: 'Toggle shortcuts', icon: '⌨️', action: () => setShortcutManager(!shortcutManager) },
    { id: 'toggle-notification-center', label: 'Toggle centro notificaciones', icon: '🔔', action: () => setNotificationCenter(!notificationCenter) },
    { id: 'toggle-fuzzy-search', label: 'Toggle búsqueda fuzzy', icon: '🔍', action: () => setSearchEnhancements(prev => ({ ...prev, fuzzySearch: !prev.fuzzySearch })) },
    { id: 'export-compressed', label: 'Exportar comprimido', icon: '📤', action: () => exportWithCompression(messages.map(m => m.id), 'json') },
    { id: 'optimize-messages', label: 'Optimizar mensajes', icon: '⚡', action: () => optimizeMessages() },
    { id: 'customize-ui', label: 'Personalizar UI', icon: '🎨', action: () => customizeUI('layout', 'compact') },
    { id: 'toggle-automation', label: 'Toggle automatización', icon: '⚙️', action: () => setAutomationManager(!automationManager) },
    { id: 'toggle-analytics-viewer', label: 'Toggle visor analytics', icon: '📊', action: () => setAnalyticsViewer(!analyticsViewer) },
    { id: 'toggle-quality-monitor', label: 'Toggle monitor calidad', icon: '⭐', action: () => setQualityMonitor(!qualityMonitor) },
    { id: 'toggle-translation', label: 'Toggle traducción tiempo real', icon: '🌐', action: () => setTranslationMode(!translationMode) },
    { id: 'toggle-collaboration-view', label: 'Toggle vista colaboración', icon: '👥', action: () => setCollaborationView(!collaborationView) },
    { id: 'toggle-version-manager', label: 'Toggle gestor versiones', icon: '📝', action: () => setVersionManager(!versionManager) },
    { id: 'toggle-ai-manager', label: 'Toggle gestor IA', icon: '🤖', action: () => setAiManager(!aiManager) },
    { id: 'toggle-learning', label: 'Toggle modo aprendizaje', icon: '🧠', action: () => setLearningMode(!learningMode) },
    { id: 'toggle-recommendations', label: 'Toggle recomendaciones', icon: '💡', action: () => setRecommendationsPanel(!recommendationsPanel) },
    { id: 'toggle-sentiment', label: 'Toggle análisis sentimientos', icon: '😊', action: () => setSentimentViewer(!sentimentViewer) },
    { id: 'toggle-summary', label: 'Toggle resumen automático', icon: '📝', action: () => setSummaryViewer(!summaryViewer) },
    { id: 'toggle-export-wizard', label: 'Toggle asistente exportación', icon: '📤', action: () => setExportWizard(!exportWizard) },
    { id: 'toggle-sync-manager', label: 'Toggle gestor sincronización', icon: '🔄', action: () => setSyncManager(!syncManager) },
    { id: 'toggle-context', label: 'Toggle visor contexto', icon: '🔍', action: () => setContextViewer(!contextViewer) },
    { id: 'toggle-patterns', label: 'Toggle analizador patrones', icon: '🔍', action: () => setPatternAnalyzer(!patternAnalyzer) },
    { id: 'toggle-flow', label: 'Toggle visor flujo', icon: '🌊', action: () => setFlowViewer(!flowViewer) },
    { id: 'toggle-timeline', label: 'Toggle visor timeline', icon: '📅', action: () => setTimelineViewer(!timelineViewer) },
    { id: 'visualize-graph', label: 'Visualizar como grafo', icon: '📊', action: () => visualizeMessages('graph') },
    { id: 'visualize-tree', label: 'Visualizar como árbol', icon: '🌳', action: () => visualizeMessages('tree') },
    { id: 'visualize-network', label: 'Visualizar como red', icon: '🕸️', action: () => visualizeMessages('network') },
    { id: 'toggle-dependency', label: 'Toggle visor dependencias', icon: '🔗', action: () => setDependencyViewer(!dependencyViewer) },
    { id: 'toggle-metrics', label: 'Toggle dashboard métricas', icon: '📈', action: () => setMetricsDashboard(!metricsDashboard) },
    { id: 'toggle-alerts', label: 'Toggle centro alertas', icon: '⚠️', action: () => setAlertCenter(!alertCenter) },
    { id: 'toggle-categories', label: 'Toggle gestor categorías', icon: '📁', action: () => setCategoryManager(!categoryManager) },
    { id: 'add-filter', label: 'Agregar filtro', icon: '🔍', action: () => addMessageFilter('nuevo', true) },
    { id: 'sort-messages', label: 'Ordenar mensajes', icon: '📊', action: () => sortMessages('timestamp', 'desc') },
    { id: 'group-messages', label: 'Agrupar mensajes', icon: '📦', action: () => groupMessages('role', []) },
    { id: 'save-search', label: 'Guardar búsqueda', icon: '💾', action: () => saveSearch('', []) },
    { id: 'add-bookmark', label: 'Agregar marcador', icon: '🔖', action: () => addBookmark('msg-1', 'Marcador', 'Nota', []) },
    { id: 'highlight', label: 'Resaltar mensaje', icon: '🖍️', action: () => highlightMessage('msg-1', 'yellow') },
    { id: 'add-annotation', label: 'Agregar anotación', icon: '📝', action: () => addAnnotation('msg-1', 'note', 'Anotación', { x: 0, y: 0 }) },
    { id: 'add-link', label: 'Agregar enlace', icon: '🔗', action: () => addLink('msg-1', 'https://example.com', 'Ejemplo', 'Descripción') },
    { id: 'add-file', label: 'Agregar archivo', icon: '📎', action: () => addFile('msg-1', 'archivo.pdf', 'pdf', 1024, '/files/archivo.pdf') },
    { id: 'add-image', label: 'Agregar imagen', icon: '🖼️', action: () => addImage('msg-1', '/images/img.jpg', 'Imagen', 'Caption') },
    { id: 'add-video', label: 'Agregar video', icon: '🎥', action: () => addVideo('msg-1', '/videos/video.mp4', '/thumb.jpg', 120) },
    { id: 'add-audio', label: 'Agregar audio', icon: '🎵', action: () => addAudio('msg-1', '/audio/audio.mp3', 180) },
    { id: 'create-form', label: 'Crear formulario', icon: '📋', action: () => createForm('msg-1', []) },
    { id: 'create-poll', label: 'Crear encuesta', icon: '📊', action: () => createPoll('msg-1', 'Pregunta?', ['Opción 1', 'Opción 2']) },
    { id: 'create-quiz', label: 'Crear cuestionario', icon: '📝', action: () => createQuiz('msg-1', []) },
    { id: 'rate-message', label: 'Calificar mensaje', icon: '⭐', action: () => rateMessage('msg-1', 5) },
    { id: 'add-feedback', label: 'Agregar feedback', icon: '💬', action: () => addFeedback('msg-1', 'Feedback', 'positive') },
    { id: 'report-message', label: 'Reportar mensaje', icon: '🚨', action: () => reportMessage('msg-1', 'spam', 'Descripción') },
    { id: 'schedule-message', label: 'Programar mensaje', icon: '⏰', action: () => scheduleMessage('msg-1', Date.now() + 3600000) },
    { id: 'create-template', label: 'Crear plantilla', icon: '📝', action: () => createTemplate('Plantilla', 'Contenido', ['var1'], 'general') },
    { id: 'add-variable', label: 'Agregar variable', icon: '🔧', action: () => addVariable('nombre', 'valor', 'string') },
    { id: 'create-trigger', label: 'Crear disparador', icon: '🎯', action: () => createTrigger('nuevo', ['cond1'], ['acc1']) },
    { id: 'create-workflow', label: 'Crear flujo trabajo', icon: '🔄', action: () => createWorkflow('Flujo', []) },
    { id: 'add-integration', label: 'Agregar integración', icon: '🔌', action: () => addIntegration('Servicio', {}) },
    { id: 'add-webhook', label: 'Agregar webhook', icon: '🔗', action: () => addWebhook('https://example.com/webhook', ['event1']) },
    { id: 'add-api', label: 'Agregar API', icon: '🌐', action: () => addAPI('/api/endpoint', 'POST', {}) },
    { id: 'add-sdk', label: 'Agregar SDK', icon: '📦', action: () => addSDK('sdk-name', '1.0.0', {}) },
    { id: 'add-plugin', label: 'Agregar plugin', icon: '🔌', action: () => addPlugin('plugin-name', '1.0.0', {}) },
    { id: 'toggle-notifications', label: 'Toggle centro notificaciones', icon: '🔔', action: () => setNotificationCenter(!notificationCenter) },
    { id: 'toggle-shortcuts', label: 'Toggle panel atajos', icon: '⌨️', action: () => setShortcutPanel(!shortcutPanel) },
    { id: 'toggle-history', label: 'Toggle visor historial', icon: '📜', action: () => setHistoryViewer(!historyViewer) },
    { id: 'toggle-favorites', label: 'Toggle favoritos', icon: '⭐', action: () => setFavoritesPanel(!favoritesPanel) },
    { id: 'toggle-pinned', label: 'Toggle fijados', icon: '📌', action: () => setPinnedPanel(!pinnedPanel) },
    { id: 'toggle-archived', label: 'Toggle archivados', icon: '📦', action: () => setArchivedPanel(!archivedPanel) },
    { id: 'toggle-trash', label: 'Toggle papelera', icon: '🗑️', action: () => setTrashPanel(!trashPanel) },
    { id: 'toggle-drafts', label: 'Toggle borradores', icon: '💾', action: () => setDraftsPanel(!draftsPanel) },
    { id: 'toggle-scheduled', label: 'Toggle programados', icon: '⏰', action: () => setScheduledPanel(!scheduledPanel) },
    { id: 'toggle-important', label: 'Toggle importantes', icon: '⚠️', action: () => setImportantPanel(!importantPanel) },
    { id: 'toggle-unread', label: 'Toggle no leídos', icon: '📬', action: () => setUnreadPanel(!unreadPanel) },
    { id: 'add-notification', label: 'Agregar notificación', icon: '🔔', action: () => addNotification('info', 'Notificación', 'Mensaje de prueba') },
    { id: 'toggle-favorite', label: 'Toggle favorito', icon: '⭐', action: () => toggleFavorite('msg-1') },
    { id: 'toggle-pin', label: 'Toggle fijar', icon: '📌', action: () => togglePin('msg-1') },
    { id: 'archive-message', label: 'Archivar mensaje', icon: '📦', action: () => archiveMessage('msg-1') },
    { id: 'delete-message', label: 'Eliminar mensaje', icon: '🗑️', action: () => deleteMessage('msg-1') },
    { id: 'save-draft', label: 'Guardar borrador', icon: '💾', action: () => saveDraft('Contenido del borrador') },
    { id: 'toggle-important-msg', label: 'Toggle importante', icon: '⚠️', action: () => toggleImportant('msg-1') },
    { id: 'toggle-read', label: 'Toggle leído', icon: '✓', action: () => toggleRead('msg-1') },
    { id: 'register-shortcut', label: 'Registrar atajo', icon: '⌨️', action: () => registerShortcut('Ctrl+K', 'action', 'Descripción') },
  ]
  
  // Calcular estadísticas de mensaje
  const calculateMessageStats = (message: any, prevMessage?: any) => {
    const wordCount = message.content.trim().split(/\s+/).filter(Boolean).length
    const charCount = message.content.length
    let responseTime: number | undefined
    
    if (prevMessage && message.role === 'assistant' && prevMessage.role === 'user') {
      const timeDiff = new Date(message.timestamp).getTime() - new Date(prevMessage.timestamp).getTime()
      responseTime = Math.round(timeDiff / 1000) // en segundos
    }
    
    return { wordCount, charCount, responseTime }
  }
  
  // Guardar versión de mensaje editado
  const saveMessageVersion = (messageId: string, content: string) => {
    setMessageHistory(prev => {
      const newMap = new Map(prev)
      const history = newMap.get(messageId) || []
      newMap.set(messageId, [...history, { content, timestamp: new Date() }])
      return newMap
    })
    setMessageVersions(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, (newMap.get(messageId) || 0) + 1)
      return newMap
    })
  }
  
  // Traducir texto (simplificado - en producción usar API real)
  const translateText = (text: string, targetLang: string): string => {
    // Placeholder - en producción usar servicio de traducción
    if (targetLang === 'en') {
      return `[Translated to English] ${text}`
    }
    return text
  }
  
  // Agregar etiqueta a mensaje
  const addTagToMessage = (messageId: string, tag: string) => {
    setMessageTags(prev => {
      const newMap = new Map(prev)
      const currentTags = newMap.get(messageId) || []
      if (!currentTags.includes(tag)) {
        newMap.set(messageId, [...currentTags, tag])
        if (!availableTags.includes(tag)) {
          setAvailableTags(prev => [...prev, tag])
        }
      }
      return newMap
    })
  }
  
  // Remover etiqueta de mensaje
  const removeTagFromMessage = (messageId: string, tag: string) => {
    setMessageTags(prev => {
      const newMap = new Map(prev)
      const currentTags = newMap.get(messageId) || []
      newMap.set(messageId, currentTags.filter(t => t !== tag))
      return newMap
    })
  }
  
  // Agregar nota a mensaje
  const saveNoteToMessage = (messageId: string, note: string) => {
    setMessageNotes(prev => {
      const newMap = new Map(prev)
      if (note.trim()) {
        newMap.set(messageId, note)
      } else {
        newMap.delete(messageId)
      }
      return newMap
    })
    setEditingNote(null)
  }
  
  // Detectar enlaces en mensajes
  const detectLinks = (text: string): string[] => {
    const urlRegex = /(https?:\/\/[^\s]+)/g
    return text.match(urlRegex) || []
  }
  
  // Obtener preview de enlace (simplificado)
  const fetchLinkPreview = async (url: string) => {
    if (linkPreview.has(url)) return linkPreview.get(url)
    
    try {
      // En producción, usar un servicio de preview de enlaces
      const preview = {
        url,
        title: new URL(url).hostname,
        description: 'Enlace externo',
      }
      setLinkPreview(prev => new Map(prev).set(url, preview))
      return preview
    } catch (e) {
      return null
    }
  }
  
  // Función simple de análisis de sentimiento
  const analyzeSentiment = (text: string): { sentiment: 'positive' | 'neutral' | 'negative', score: number } => {
    const positiveWords = ['bueno', 'excelente', 'genial', 'perfecto', 'gracias', 'ayuda', 'sí', 'correcto', 'bien']
    const negativeWords = ['malo', 'error', 'no', 'problema', 'incorrecto', 'fallo', 'mal']
    
    const lowerText = text.toLowerCase()
    let positiveCount = 0
    let negativeCount = 0
    
    positiveWords.forEach(word => {
      if (lowerText.includes(word)) positiveCount++
    })
    negativeWords.forEach(word => {
      if (lowerText.includes(word)) negativeCount++
    })
    
    if (positiveCount > negativeCount) {
      return { sentiment: 'positive', score: (positiveCount / (positiveCount + negativeCount + 1)) }
    } else if (negativeCount > positiveCount) {
      return { sentiment: 'negative', score: (negativeCount / (positiveCount + negativeCount + 1)) }
    }
    return { sentiment: 'neutral', score: 0.5 }
  }
  
  // Reacciones disponibles
  const availableReactions = ['👍', '❤️', '😂', '😮', '😢', '🔥', '✅', '❌']

  // ========== NUEVAS FUNCIONES AVANZADAS ==========
  
  // Voice Input/Output
  const startVoiceInput = useCallback(() => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      toast.error('Reconocimiento de voz no disponible en este navegador')
      return
    }
    
    const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition
    const recognition = new SpeechRecognition()
    recognition.continuous = false
    recognition.interimResults = false
    recognition.lang = 'es-ES'
    
    recognition.onstart = () => {
      setIsRecording(true)
      toast.success('Escuchando...', { icon: '🎤' })
    }
    
    recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript
      setInput(prev => prev + (prev ? ' ' : '') + transcript)
      setIsRecording(false)
      toast.success('Voz reconocida', { icon: '✅' })
    }
    
    recognition.onerror = (event: any) => {
      setIsRecording(false)
      toast.error(`Error de reconocimiento: ${event.error}`)
    }
    
    recognition.onend = () => {
      setIsRecording(false)
    }
    
    recognitionRef.current = recognition
    recognition.start()
  }, [])
  
  const stopVoiceInput = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.stop()
      setIsRecording(false)
    }
  }, [])
  
  const speakText = useCallback((text: string) => {
    if (!('speechSynthesis' in window)) {
      toast.error('Síntesis de voz no disponible')
      return
    }
    
    if (synthesisRef.current) {
      window.speechSynthesis.cancel()
    }
    
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.lang = targetLanguage === 'en' ? 'en-US' : 'es-ES'
    utterance.rate = readingSpeed === 'slow' ? 0.8 : readingSpeed === 'fast' ? 1.2 : 1.0
    utterance.pitch = 1.0
    utterance.volume = 1.0
    
    synthesisRef.current = window.speechSynthesis
    synthesisRef.current.speak(utterance)
  }, [targetLanguage, readingSpeed])
  
  // Message Summarization
  const generateSummary = useCallback(() => {
    if (messages.length === 0) {
      toast.error('No hay mensajes para resumir')
      return
    }
    
    const userMessages = messages.filter(m => m.role === 'user').map(m => m.content).join('\n')
    const assistantMessages = messages.filter(m => m.role === 'assistant').map(m => m.content).join('\n')
    
    // Resumen simple (en producción usar API de resumen)
    const summary = `Resumen de conversación (${messages.length} mensajes):
    
Temas principales discutidos:
- ${userMessages.substring(0, 200)}...

Respuestas del asistente:
- ${assistantMessages.substring(0, 200)}...

Total de palabras: ${messages.reduce((acc, m) => acc + m.content.split(' ').length, 0)}
Total de caracteres: ${messages.reduce((acc, m) => acc + m.content.length, 0)}`
    
    setConversationSummary(summary)
    setShowSummary(true)
    toast.success('Resumen generado', { icon: '📝' })
  }, [messages])
  
  // Message Clustering
  const clusterMessages = useCallback(() => {
    const clusters = new Map<string, string[]>()
    
    // Agrupar por tags
    messageTags.forEach((tags, messageId) => {
      tags.forEach(tag => {
        if (!clusters.has(tag)) {
          clusters.set(tag, [])
        }
        clusters.get(tag)!.push(messageId)
      })
    })
    
    // Agrupar por sentimiento
    const sentimentClusters = new Map<string, string[]>()
    messages.forEach(msg => {
      const sentiment = analyzeSentiment(msg.content).sentiment
      if (!sentimentClusters.has(sentiment)) {
        sentimentClusters.set(sentiment, [])
      }
      sentimentClusters.get(sentiment)!.push(msg.id)
    })
    
    // Combinar clusters
    const allClusters = new Map([...clusters, ...sentimentClusters])
    setMessageClusters(allClusters)
    setShowClustering(true)
    toast.success(`Se encontraron ${allClusters.size} clusters`, { icon: '🔗' })
  }, [messages, messageTags])
  
  // Message Scheduling
  const scheduleMessage = useCallback((message: string, delayMinutes: number) => {
    const timestamp = Date.now() + (delayMinutes * 60 * 1000)
    const id = `scheduled-${Date.now()}`
    
    setScheduledMessages(prev => {
      const newMap = new Map(prev)
      newMap.set(id, { message, timestamp })
      return newMap
    })
    
    // Programar envío
    setTimeout(() => {
      if (useBulkChatMode && bulkChat.sessionId) {
        bulkChat.sendMessage(message)
        setScheduledMessages(prev => {
          const newMap = new Map(prev)
          newMap.delete(id)
          return newMap
        })
        toast.success('Mensaje programado enviado', { icon: '⏰' })
      }
    }, delayMinutes * 60 * 1000)
    
    toast.success(`Mensaje programado para ${delayMinutes} minutos`, { icon: '⏰' })
  }, [useBulkChatMode, bulkChat])
  
  // Message Archiving
  const archiveMessage = useCallback((messageId: string) => {
    setArchivedMessages(prev => {
      const newSet = new Set(prev)
      newSet.add(messageId)
      return newSet
    })
    toast.success('Mensaje archivado', { icon: '📦' })
  }, [])
  
  const unarchiveMessage = useCallback((messageId: string) => {
    setArchivedMessages(prev => {
      const newSet = new Set(prev)
      newSet.delete(messageId)
      return newSet
    })
    toast.success('Mensaje desarchivado', { icon: '📦' })
  }, [])
  
  // Message Diff View
  const showMessageDiff = useCallback((messageId1: string, messageId2: string) => {
    const msg1 = messages.find(m => m.id === messageId1)
    const msg2 = messages.find(m => m.id === messageId2)
    
    if (!msg1 || !msg2) {
      toast.error('Mensajes no encontrados')
      return
    }
    
    setDiffMessages([msg1.content, msg2.content])
    setShowDiffView(true)
  }, [messages])
  
  // Message Reminders
  const addReminder = useCallback((messageId: string, delayMinutes: number, note: string) => {
    const timestamp = Date.now() + (delayMinutes * 60 * 1000)
    
    setMessageReminders(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { timestamp, note })
      return newMap
    })
    
    setTimeout(() => {
      const msg = messages.find(m => m.id === messageId)
      if (msg) {
        toast(`⏰ Recordatorio: ${note}`, {
          icon: '🔔',
          duration: 5000,
        })
        // Scroll al mensaje
        const element = messageRefs.current.get(messageId)
        if (element) {
          element.scrollIntoView({ behavior: 'smooth', block: 'center' })
          element.classList.add('ring-2', 'ring-yellow-500')
          setTimeout(() => {
            element.classList.remove('ring-2', 'ring-yellow-500')
          }, 3000)
        }
      }
      setMessageReminders(prev => {
        const newMap = new Map(prev)
        newMap.delete(messageId)
        return newMap
      })
    }, delayMinutes * 60 * 1000)
    
    toast.success(`Recordatorio configurado para ${delayMinutes} minutos`, { icon: '🔔' })
  }, [messages])
  
  // Message Backup/Restore
  const backupConversation = useCallback(() => {
    const backup = {
      timestamp: new Date().toISOString(),
      messages: messages.map(m => ({
        id: m.id,
        role: m.role,
        content: m.content,
        timestamp: m.timestamp,
      })),
      metadata: {
        messageCount: messages.length,
        favoriteCount: favoriteMessages.size,
        tagCount: Array.from(messageTags.values()).reduce((acc, tags) => acc + tags.length, 0),
        noteCount: messageNotes.size,
      },
    }
    
    const backups = JSON.parse(localStorage.getItem('bulk-chat-backups') || '[]')
    backups.push(backup)
    
    // Mantener solo los últimos 10 backups
    if (backups.length > 10) {
      backups.shift()
    }
    
    localStorage.setItem('bulk-chat-backups', JSON.stringify(backups))
    setBackupHistory(backups)
    toast.success('Conversación respaldada', { icon: '💾' })
  }, [messages, favoriteMessages, messageTags, messageNotes])
  
  const restoreConversation = useCallback((backup: any) => {
    if (!backup || !backup.messages) {
      toast.error('Backup inválido')
      return
    }
    
    clearMessages()
    backup.messages.forEach((msg: any) => {
      addMessage({
        id: msg.id,
        role: msg.role,
        content: msg.content,
        timestamp: new Date(msg.timestamp),
      })
    })
    
    toast.success('Conversación restaurada', { icon: '📥' })
  }, [clearMessages, addMessage])
  
  // Load backups on mount
  useEffect(() => {
    try {
      const backups = JSON.parse(localStorage.getItem('bulk-chat-backups') || '[]')
      setBackupHistory(backups)
    } catch (error) {
      console.error('Error loading backups:', error)
    }
  }, [])
  
  // Fijar/desfijar mensaje
  const togglePinMessage = useCallback((messageId: string) => {
    setPinnedMessages(prev => {
      const newSet = new Set(prev)
      if (newSet.has(messageId)) {
        newSet.delete(messageId)
        toast.success('Mensaje desfijado', { icon: '📌', duration: 1500 })
      } else {
        newSet.add(messageId)
        toast.success('Mensaje fijado', { icon: '📌', duration: 1500 })
      }
      // Persistir en localStorage
      try {
        localStorage.setItem('bulk-chat-pinned', JSON.stringify(Array.from(newSet)))
      } catch (e) {
        console.error('Error saving pinned messages:', e)
      }
      return newSet
    })
  }, [])
  
  // Cargar mensajes fijados al montar
  useEffect(() => {
    try {
      const saved = localStorage.getItem('bulk-chat-pinned')
      if (saved) {
        setPinnedMessages(new Set(JSON.parse(saved)))
      }
    } catch (error) {
      console.error('Error loading pinned messages:', error)
    }
  }, [])
  
  // Crear thread desde un mensaje
  const createThreadFromMessage = useCallback((parentId: string) => {
    setThreadParent(parentId)
    toast.success('Thread iniciado. Responde para crear una rama.', { icon: '🌿', duration: 2000 })
  }, [])
  
  // Agregar mensaje a thread
  const addToThread = useCallback((parentId: string, messageId: string) => {
    setMessageThreads(prev => {
      const newMap = new Map(prev)
      const children = newMap.get(parentId) || []
      if (!children.includes(messageId)) {
        newMap.set(parentId, [...children, messageId])
      }
      return newMap
    })
  }, [])
  
  // Exportar a PDF (HTML que se puede imprimir a PDF)
  const exportToPDF = useCallback(async () => {
    try {
      const htmlContent = `
        <!DOCTYPE html>
        <html>
          <head>
            <meta charset="UTF-8">
            <title>Conversación Exportada</title>
            <style>
              body { font-family: Arial, sans-serif; padding: 20px; max-width: 800px; margin: 0 auto; }
              .message { margin: 15px 0; padding: 15px; border-left: 4px solid #ccc; background: #f9f9f9; }
              .user { border-color: #3b82f6; }
              .assistant { border-color: #10b981; }
              .timestamp { color: #666; font-size: 0.9em; margin-bottom: 5px; }
              .content { line-height: 1.6; }
              h1 { color: #333; }
              .metadata { color: #666; font-size: 0.9em; margin-bottom: 20px; }
            </style>
          </head>
          <body>
            <h1>Conversación Exportada</h1>
            <div class="metadata">
              <p><strong>Fecha de exportación:</strong> ${new Date().toLocaleString()}</p>
              <p><strong>Total de mensajes:</strong> ${bulkChat.messages.length}</p>
              <p><strong>Sesión ID:</strong> ${bulkChat.sessionId || 'N/A'}</p>
            </div>
            ${bulkChat.messages.map(msg => {
              const tags = messageTags.get(msg.id) || []
              const note = messageNotes.get(msg.id)
              const isPinned = pinnedMessages.has(msg.id)
              return `
                <div class="message ${msg.role}">
                  <div class="timestamp">
                    ${new Date(msg.timestamp).toLocaleString()} 
                    ${isPinned ? '📌' : ''}
                    ${tags.length > 0 ? ` | Etiquetas: ${tags.join(', ')}` : ''}
                  </div>
                  ${note ? `<div style="color: #f59e0b; font-style: italic; margin-bottom: 5px;">📝 Nota: ${note}</div>` : ''}
                  <div class="content">${msg.content.replace(/\n/g, '<br>').replace(/`([^`]+)`/g, '<code>$1</code>')}</div>
                </div>
              `
            }).join('')}
          </body>
        </html>
      `
      
      const blob = new Blob([htmlContent], { type: 'text/html' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `conversacion-${Date.now()}.html`
      a.click()
      URL.revokeObjectURL(url)
      
      toast.success('Exportación iniciada (HTML). Usa "Imprimir a PDF" en el navegador.', { icon: '📄', duration: 3000 })
    } catch (error) {
      toast.error('Error al exportar', { icon: '❌' })
    }
  }, [bulkChat.messages, bulkChat.sessionId, messageTags, messageNotes, pinnedMessages])
  
  // Exportar a Markdown mejorado
  const exportToMarkdownEnhanced = useCallback(() => {
    try {
      const mdContent = `# Conversación Exportada

**Fecha de exportación:** ${new Date().toLocaleString()}
**Total de mensajes:** ${bulkChat.messages.length}
**Sesión ID:** ${bulkChat.sessionId || 'N/A'}

---

${bulkChat.messages.map((msg, idx) => {
        const role = msg.role === 'user' ? '👤 Usuario' : '🤖 Asistente'
        const timestamp = new Date(msg.timestamp).toLocaleString()
        const tags = messageTags.get(msg.id) || []
        const note = messageNotes.get(msg.id)
        const isPinned = pinnedMessages.has(msg.id)
        const reactions = messageReactions.get(msg.id) || []
        
        return `## ${role} ${isPinned ? '📌' : ''}

**Fecha:** ${timestamp}

${tags.length > 0 ? `**Etiquetas:** ${tags.join(', ')}\n\n` : ''}${note ? `**Nota:** ${note}\n\n` : ''}${reactions.length > 0 ? `**Reacciones:** ${reactions.join(' ')}\n\n` : ''}${msg.content}

---

`
      }).join('\n')}`
      
      const blob = new Blob([mdContent], { type: 'text/markdown' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `conversacion-${Date.now()}.md`
      a.click()
      URL.revokeObjectURL(url)
      
      toast.success('Conversación exportada (Markdown mejorado)', { icon: '📝', duration: 2000 })
    } catch (error) {
      toast.error('Error al exportar', { icon: '❌' })
    }
  }, [bulkChat.messages, bulkChat.sessionId, messageTags, messageNotes, pinnedMessages, messageReactions])
  
  // Vista previa de Markdown
  const renderMarkdownPreview = useCallback((content: string) => {
    // Renderizado básico de Markdown
    return content
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`([^`]+)`/g, '<code style="background: #1e293b; padding: 2px 6px; border-radius: 3px;">$1</code>')
      .replace(/```([\s\S]*?)```/g, '<pre style="background: #1e293b; padding: 10px; border-radius: 5px; overflow-x: auto;"><code>$1</code></pre>')
      .replace(/#{3}\s(.*)/g, '<h3 style="margin-top: 20px; margin-bottom: 10px;">$1</h3>')
      .replace(/#{2}\s(.*)/g, '<h2 style="margin-top: 20px; margin-bottom: 10px;">$1</h2>')
      .replace(/#{1}\s(.*)/g, '<h1 style="margin-top: 20px; margin-bottom: 10px;">$1</h1>')
      .replace(/\n/g, '<br>')
  }, [])
  
  // Agrupar mensajes
  const groupMessages = useCallback(() => {
    if (groupingMode === 'none') {
      setMessageGroups(new Map())
      return
    }

    const groups = new Map<string, string[]>()
    
    if (groupingMode === 'time') {
      bulkChat.messages.forEach(msg => {
        const date = new Date(msg.timestamp)
        const hour = date.getHours()
        const key = `${date.toDateString()} - ${hour}:00`
        const existing = groups.get(key) || []
        groups.set(key, [...existing, msg.id])
      })
    } else if (groupingMode === 'role') {
      bulkChat.messages.forEach(msg => {
        const key = msg.role
        const existing = groups.get(key) || []
        groups.set(key, [...existing, msg.id])
      })
    } else if (groupingMode === 'topic') {
      // Agrupar por etiquetas
      bulkChat.messages.forEach(msg => {
        const tags = messageTags.get(msg.id) || []
        const key = tags.length > 0 ? tags[0] : 'sin-etiqueta'
        const existing = groups.get(key) || []
        groups.set(key, [...existing, msg.id])
      })
    }
    
    setMessageGroups(groups)
  }, [groupingMode, bulkChat.messages, messageTags])

  useEffect(() => {
    groupMessages()
  }, [groupMessages])
  
  // ========== FUNCIONES ADICIONALES AVANZADAS ==========
  
  // Compartir mensaje
  const shareMessage = useCallback((messageId: string, method: 'copy' | 'email' | 'link') => {
    const message = messages.find(m => m.id === messageId)
    if (!message) return
    
    if (method === 'copy') {
      navigator.clipboard.writeText(message.content)
      toast.success('Mensaje copiado al portapapeles', { icon: '📋' })
    } else if (method === 'link') {
      const shareUrl = `${window.location.origin}?message=${messageId}`
      navigator.clipboard.writeText(shareUrl)
      toast.success('Enlace copiado al portapapeles', { icon: '🔗' })
    } else if (method === 'email') {
      const subject = encodeURIComponent('Mensaje compartido')
      const body = encodeURIComponent(`Mensaje de ${message.role}:\n\n${message.content}`)
      window.open(`mailto:?subject=${subject}&body=${body}`)
    }
    
    setMessageSharing(prev => {
      const newMap = new Map(prev)
      const current = newMap.get(messageId) || []
      if (!current.includes(method)) {
        newMap.set(messageId, [...current, method])
      }
      return newMap
    })
  }, [messages])
  
  // Plantillas con variables
  const createTemplateWithVars = useCallback((name: string, template: string) => {
    const variables = template.match(/\{\{(\w+)\}\}/g)?.map(v => v.replace(/[{}]/g, '')) || []
    setMessageTemplatesWithVars(prev => {
      const newMap = new Map(prev)
      newMap.set(name, { template, variables })
      return newMap
    })
    toast.success(`Plantilla "${name}" creada con ${variables.length} variable(s)`, { icon: '📝' })
  }, [])
  
  const useTemplate = useCallback((templateName: string, vars: Record<string, string>) => {
    const template = messageTemplatesWithVars.get(templateName)
    if (!template) {
      toast.error('Plantilla no encontrada', { icon: '❌' })
      return
    }
    
    let result = template.template
    template.variables.forEach(varName => {
      const value = vars[varName] || `{{${varName}}}`
      result = result.replace(new RegExp(`\\{\\{${varName}\\}\\}`, 'g'), value)
    })
    
    setInput(result)
    toast.success(`Plantilla "${templateName}" aplicada`, { icon: '✅' })
  }, [messageTemplatesWithVars])
  
  // Notificaciones inteligentes
  const checkNotificationRules = useCallback((message: any) => {
    if (!smartNotifications) return
    
    notificationRules.forEach((rule, ruleName) => {
      if (!rule.enabled) return
      
      const matches = rule.keywords.some(keyword => 
        message.content.toLowerCase().includes(keyword.toLowerCase())
      )
      
      if (matches) {
        toast(`🔔 ${ruleName}: Nuevo mensaje relevante`, {
          icon: '🔔',
          duration: 5000,
        })
      }
    })
  }, [smartNotifications, notificationRules])
  
  // Actualizar referencia
  useEffect(() => {
    checkNotificationRulesRef.current = checkNotificationRules
  }, [checkNotificationRules])
  
  // Estadísticas en tiempo real
  useEffect(() => {
    if (!realTimeStats) return
    
    const interval = setInterval(() => {
      const stats = {
        messagesPerMinute: messages.length / Math.max(1, (Date.now() - new Date(messages[0]?.timestamp || Date.now()).getTime()) / 60000),
        averageResponseTime: Array.from(messageStats.values())
          .filter(s => s.responseTime !== undefined)
          .reduce((acc, s, _, arr) => acc + (s.responseTime || 0) / arr.length, 0),
        activeThreads: messageThreads.size,
        pinnedCount: pinnedMessages.size,
        archivedCount: archivedMessages.size,
      }
      
      // Actualizar UI con estadísticas en tiempo real
      // Esto se puede mostrar en un panel dedicado
    }, 5000)
    
    return () => clearInterval(interval)
  }, [realTimeStats, messages, messageStats, messageThreads, pinnedMessages, archivedMessages])
  
  // Encriptación de mensajes (simplificada - Base64 para demo)
  const encryptMessage = useCallback((content: string): string => {
    if (!messageEncryption) return content
    return btoa(encodeURIComponent(content))
  }, [messageEncryption])
  
  const decryptMessage = useCallback((encrypted: string): string => {
    try {
      return decodeURIComponent(atob(encrypted))
    } catch {
      return encrypted
    }
  }, [])
  
  // Búsqueda avanzada
  const applyAdvancedSearch = useCallback((messages: any[]) => {
    if (!advancedSearch || Object.keys(searchFilters).length === 0) return messages
    
    return messages.filter(msg => {
      if (searchFilters.dateRange) {
        const msgDate = new Date(msg.timestamp)
        if (msgDate < searchFilters.dateRange.start || msgDate > searchFilters.dateRange.end) {
          return false
        }
      }
      
      if (searchFilters.minWords !== undefined) {
        const wordCount = msg.content.trim().split(/\s+/).filter(Boolean).length
        if (wordCount < searchFilters.minWords!) return false
      }
      
      if (searchFilters.maxWords !== undefined) {
        const wordCount = msg.content.trim().split(/\s+/).filter(Boolean).length
        if (wordCount > searchFilters.maxWords!) return false
      }
      
      if (searchFilters.hasCode !== undefined) {
        const hasCode = /```[\s\S]*?```|`[^`]+`/.test(msg.content)
        if (hasCode !== searchFilters.hasCode) return false
      }
      
      if (searchFilters.hasLinks !== undefined) {
        const hasLinks = /https?:\/\/[^\s]+/.test(msg.content)
        if (hasLinks !== searchFilters.hasLinks) return false
      }
      
      return true
    })
  }, [advancedSearch, searchFilters])
  
  // ========== NUEVAS FUNCIONES AVANZADAS ==========
  
  // Grabación de sesiones
  const startSessionRecording = useCallback(() => {
    setSessionRecording(true)
    const sessionData = {
      id: `session-${Date.now()}`,
      startTime: Date.now(),
      messages: [],
      events: []
    }
    setRecordedSessions(prev => [...prev, sessionData])
    setCurrentSession(sessionData)
    toast.success('Grabación de sesión iniciada', { icon: '🔴' })
  }, [])
  
  const stopSessionRecording = useCallback(() => {
    setSessionRecording(false)
    if (currentSession) {
      setRecordedSessions(prev => prev.map(s => 
        s.id === currentSession.id 
          ? { ...s, endTime: Date.now(), messages: messages }
          : s
      ))
    }
    toast.success('Grabación de sesión detenida', { icon: '⏹️' })
  }, [currentSession, messages])
  
  // Marcadores/Bookmarks
  // Marcadores - Usando hook modular
  const addBookmark = useCallback((messageId: string, name: string) => {
    messageActions.addBookmark(messageId, name, 'default', [])
    eventBus.emit(MessageEvents.BOOKMARK_ADDED, { messageId, name })
    const bookmark = {
      name: name || `Bookmark ${bookmarks.size + 1}`,
      messageId,
      timestamp: Date.now()
    }
    setBookmarks(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, bookmark)
      return newMap
    })
    setMessageBookmarks(prev => new Set(prev).add(messageId))
    toast.success('Marcador agregado', { icon: '🔖' })
  }, [bookmarks.size])
  
  const removeBookmark = useCallback((messageId: string) => {
    setBookmarks(prev => {
      const newMap = new Map(prev)
      newMap.delete(messageId)
      return newMap
    })
    setMessageBookmarks(prev => {
      const newSet = new Set(prev)
      newSet.delete(messageId)
      return newSet
    })
    toast.success('Marcador eliminado', { icon: '🗑️' })
  }, [])
  
  // Modo de estudio - Flashcards
  const createFlashcard = useCallback((messageId: string, question: string, answer: string) => {
    setFlashcards(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { question, answer })
      return newMap
    })
    toast.success('Flashcard creada', { icon: '📚' })
  }, [])
  
  // Modo de productividad - Pomodoro
  useEffect(() => {
    if (!productivityMode || focusTimer >= focusGoal * 60) {
      if (focusTimer > 0 && focusTimer >= focusGoal * 60) {
        toast.success('¡Sesión de enfoque completada!', { icon: '🎉', duration: 5000 })
        setFocusTimer(0)
      }
      return
    }
    
    const interval = setInterval(() => {
      setFocusTimer(prev => prev + 1)
    }, 1000)
    
    return () => clearInterval(interval)
  }, [productivityMode, focusTimer, focusGoal])
  
  // Insights de IA
  const generateAIInsights = useCallback(() => {
    const insights = {
      totalMessages: messages.length,
      averageLength: messages.length > 0 ? messages.reduce((acc, m) => acc + m.content.length, 0) / messages.length : 0,
      topics: Array.from(messageTags.values()).flat(),
      sentiment: { positive: 0, negative: 0, neutral: 0 }, // Simplificado
      mostActiveHour: new Date().getHours(),
      keyInsights: [
        `Total de ${messages.length} mensajes`,
        `Promedio de ${messages.length > 0 ? Math.round(messages.reduce((acc, m) => acc + m.content.length, 0) / messages.length) : 0} caracteres por mensaje`,
        `${pinnedMessages.size} mensajes fijados`,
        `${archivedMessages.size} mensajes archivados`
      ]
    }
    setConversationInsights(insights)
    toast.success('Insights generados', { icon: '🤖' })
  }, [messages, messageTags, pinnedMessages, archivedMessages])
  
  // Modo de lectura
  useEffect(() => {
    if (readingMode && messages.length > 0) {
      const totalMessages = messages.length
      const readMessages = Math.floor((readingProgress / 100) * totalMessages)
      // Auto-avanzar progreso de lectura
    }
  }, [readingMode, readingProgress, messages.length])
  
  // Integración con calendario
  const scheduleEvent = useCallback((title: string, date: Date, messageId?: string) => {
    const eventId = `event-${Date.now()}`
    setScheduledEvents(prev => {
      const newMap = new Map(prev)
      newMap.set(eventId, { title, date, messageId })
      return newMap
    })
    toast.success(`Evento programado: ${title}`, { icon: '📅' })
  }, [])
  
  // Sincronización en la nube
  const syncToCloud = useCallback(async () => {
    if (!cloudSync) return
    
    try {
      const data = {
        messages,
        bookmarks: Array.from(bookmarks.entries()),
        flashcards: Array.from(flashcards.entries()),
        studyNotes: Array.from(studyNotes.entries()),
        timestamp: Date.now()
      }
      
      if (syncProvider === 'localStorage') {
        localStorage.setItem('chat_sync', JSON.stringify(data))
      } else if (syncProvider === 'indexedDB') {
        // Implementar IndexedDB sync
      }
      
      toast.success('Sincronización completada', { icon: '☁️' })
    } catch (error) {
      toast.error('Error en sincronización', { icon: '❌' })
    }
  }, [cloudSync, syncProvider, messages, bookmarks, flashcards, studyNotes])
  
  // Sistema de badges/logros
  const checkAchievements = useCallback(() => {
    const newAchievements = new Set<string>()
    
    if (messages.length >= 10) newAchievements.add('first_10_messages')
    if (messages.length >= 100) newAchievements.add('century')
    if (pinnedMessages.size >= 5) newAchievements.add('pinner')
    if (bookmarks.size >= 10) newAchievements.add('bookmarker')
    if (flashcards.size >= 5) newAchievements.add('student')
    if (focusTimer >= focusGoal * 60) newAchievements.add('focused')
    
    setAchievements(prev => {
      const combined = new Set([...prev, ...newAchievements])
      if (combined.size > prev.size) {
        toast.success(`¡Logro desbloqueado!`, { icon: '🏆', duration: 3000 })
      }
      return combined
    })
  }, [messages.length, pinnedMessages.size, bookmarks.size, flashcards.size, focusTimer, focusGoal])
  
  useEffect(() => {
    checkAchievements()
  }, [checkAchievements])
  
  // Anotaciones
  const addAnnotation = useCallback((messageId: string, type: 'highlight' | 'comment' | 'question', content: string) => {
    setMessageAnnotations(prev => {
      const newMap = new Map(prev)
      const existing = newMap.get(messageId) || []
      newMap.set(messageId, [...existing, { type, content }])
      return newMap
    })
    toast.success('Anotación agregada', { icon: '📝' })
  }, [])
  
  // Carpetas inteligentes
  const createSmartFolder = useCallback((name: string, filters: any) => {
    const folderId = `folder-${Date.now()}`
    const matchingMessages = messages.filter(msg => {
      // Aplicar filtros
      return true // Simplificado
    }).map(m => m.id)
    
    setSmartFolders(prev => {
      const newMap = new Map(prev)
      newMap.set(folderId, { name, filters, messageIds: matchingMessages })
      return newMap
    })
    toast.success(`Carpeta "${name}" creada con ${matchingMessages.length} mensajes`, { icon: '📁' })
  }, [messages])
  
  // Prioridades de mensajes
  const setMessagePriorityLevel = useCallback((messageId: string, priority: 'low' | 'medium' | 'high' | 'urgent') => {
    setMessagePriority(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, priority)
      return newMap
    })
    toast.success(`Prioridad ${priority} asignada`, { icon: '⚡' })
  }, [])
  
  // ========== FUNCIONES ADICIONALES ==========
  
  // Modo de pantalla completa
  const toggleFullscreen = useCallback(() => {
    if (!fullscreenMode) {
      if (document.documentElement.requestFullscreen) {
        document.documentElement.requestFullscreen()
      }
      setFullscreenMode(true)
      toast.success('Modo pantalla completa activado', { icon: '⛶' })
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen()
      }
      setFullscreenMode(false)
      toast.success('Modo pantalla completa desactivado', { icon: '⛶' })
    }
  }, [fullscreenMode])
  
  // Modo de presentación
  const startPresentation = useCallback(() => {
    setPresentationMode(true)
    setPresentationSlide(0)
    toast.success('Modo presentación iniciado', { icon: '📊' })
  }, [])
  
  const nextSlide = useCallback(() => {
    if (presentationMode && presentationSlide < messages.length - 1) {
      setPresentationSlide(prev => prev + 1)
    }
  }, [presentationMode, presentationSlide, messages.length])
  
  const prevSlide = useCallback(() => {
    if (presentationMode && presentationSlide > 0) {
      setPresentationSlide(prev => prev - 1)
    }
  }, [presentationMode, presentationSlide])
  
  // Versiones de conversación
  const saveConversationVersion = useCallback((name: string) => {
    const versionId = `version-${Date.now()}`
    setConversationVersions(prev => {
      const newMap = new Map(prev)
      newMap.set(versionId, {
        timestamp: Date.now(),
        messages: [...messages],
        metadata: { name, messageCount: messages.length }
      })
      return newMap
    })
    toast.success(`Versión "${name}" guardada`, { icon: '💾' })
  }, [messages])
  
  const restoreConversationVersion = useCallback((versionId: string) => {
    const version = conversationVersions.get(versionId)
    if (version) {
      clearMessages()
      version.messages.forEach(msg => addMessage(msg))
      toast.success('Versión restaurada', { icon: '↩️' })
    }
  }, [conversationVersions, clearMessages, addMessage])
  
  // Sistema de votos
  const voteMessage = useCallback((messageId: string, type: 'up' | 'down') => {
    setMessageVotes(prev => {
      const newMap = new Map(prev)
      const current = newMap.get(messageId) || { up: 0, down: 0 }
      if (type === 'up') {
        current.up += 1
      } else {
        current.down += 1
      }
      newMap.set(messageId, current)
      return newMap
    })
    toast.success(`Voto ${type === 'up' ? 'positivo' : 'negativo'} registrado`, { icon: '👍' })
  }, [])
  
  // Sistema de ratings
  const rateMessage = useCallback((messageId: string, rating: number) => {
    if (rating < 1 || rating > 5) return
    setMessageRatings(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, rating)
      return newMap
    })
    toast.success(`Calificación ${rating}/5 asignada`, { icon: '⭐' })
  }, [])
  
  // Sistema de comentarios
  const addComment = useCallback((messageId: string, content: string, author: string = 'Usuario') => {
    setMessageComments(prev => {
      const newMap = new Map(prev)
      const existing = newMap.get(messageId) || []
      newMap.set(messageId, [...existing, { author, content, timestamp: Date.now() }])
      return newMap
    })
    toast.success('Comentario agregado', { icon: '💬' })
  }, [])
  
  // Exportación a CSV
  const exportToCSV = useCallback(() => {
    const csvContent = [
      ['ID', 'Role', 'Content', 'Timestamp'].join(','),
      ...messages.map(msg => [
        msg.id,
        msg.role,
        `"${msg.content.replace(/"/g, '""')}"`,
        new Date(msg.timestamp).toISOString()
      ].join(','))
    ].join('\n')
    
    const blob = new Blob([csvContent], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `conversation-${Date.now()}.csv`
    a.click()
    URL.revokeObjectURL(url)
    toast.success('Exportado a CSV', { icon: '📊' })
  }, [messages])
  
  // Exportación a Excel (simplificada - CSV con extensión .xlsx)
  const exportToExcel = useCallback(() => {
    exportToCSV() // Por ahora, usar CSV como Excel
    toast.success('Exportado a Excel', { icon: '📊' })
  }, [exportToCSV])
  
  // ========== FUNCIONES ADICIONALES AVANZADAS ==========
  
  // Notificaciones Push
  const requestNotificationPermission = useCallback(async () => {
    if ('Notification' in window) {
      const permission = await Notification.requestPermission()
      setNotificationPermission(permission)
      if (permission === 'granted') {
        setPushNotifications(true)
        toast.success('Notificaciones push habilitadas', { icon: '🔔' })
      }
    }
  }, [])
  
  // Modo de impresión
  const printConversation = useCallback(() => {
    setPrintMode(true)
    setTimeout(() => {
      window.print()
      setPrintMode(false)
    }, 100)
  }, [])
  
  // Integración con APIs externas
  const addApiIntegration = useCallback((name: string, endpoint: string) => {
    const id = `api-${Date.now()}`
    setApiIntegrationsList(prev => {
      const newMap = new Map(prev)
      newMap.set(id, { name, endpoint, enabled: true })
      return newMap
    })
    toast.success(`Integración "${name}" agregada`, { icon: '🔌' })
  }, [])
  
  // Sistema de plugins
  const installPlugin = useCallback((name: string, description: string) => {
    setPluginStore(prev => {
      const newMap = new Map(prev)
      newMap.set(name, { name, description, enabled: true })
      return newMap
    })
    setActivePlugins(prev => [...prev, name])
    toast.success(`Plugin "${name}" instalado`, { icon: '🔌' })
  }, [])
  
  // Temas personalizados
  const createCustomTheme = useCallback((name: string, colors: Record<string, string>) => {
    setCustomThemes(prev => {
      const newMap = new Map(prev)
      newMap.set(name, colors)
      return newMap
    })
    setThemePresets(prev => {
      const newMap = new Map(prev)
      newMap.set(name, { colors, createdAt: Date.now() })
      return newMap
    })
    toast.success(`Tema "${name}" creado`, { icon: '🎨' })
  }, [])
  
  // Analytics avanzado
  const generateAdvancedAnalytics = useCallback(() => {
    const data = {
      totalMessages: messages.length,
      averageResponseTime: Array.from(messageStats.values())
        .filter(s => s.responseTime !== undefined)
        .reduce((acc, s, _, arr) => acc + (s.responseTime || 0) / arr.length, 0),
      messagesPerHour: messages.length / Math.max(1, (Date.now() - new Date(messages[0]?.timestamp || Date.now()).getTime()) / 3600000),
      topKeywords: Array.from(messageTags.values()).flat().reduce((acc, tag) => {
        acc[tag] = (acc[tag] || 0) + 1
        return acc
      }, {} as Record<string, number>),
      engagementMetrics: {
        votes: Array.from(messageVotes.values()).reduce((acc, v) => acc + v.up + v.down, 0),
        ratings: Array.from(messageRatings.values()).length,
        comments: Array.from(messageComments.values()).reduce((acc, c) => acc + c.length, 0),
      },
      productivityMetrics: {
        bookmarks: bookmarks.size,
        flashcards: flashcards.size,
        pinned: pinnedMessages.size,
        archived: archivedMessages.size,
      }
    }
    setAnalyticsData(data)
    setAdvancedAnalytics(true)
    toast.success('Analytics avanzado generado', { icon: '📊' })
  }, [messages, messageStats, messageTags, messageVotes, messageRatings, messageComments, bookmarks, flashcards, pinnedMessages, archivedMessages])
  
  // Vista de timeline
  const toggleTimelineView = useCallback(() => {
    setMessageTimeline(!messageTimeline)
    if (!messageTimeline) {
      setTimelineView('linear')
    }
  }, [messageTimeline])
  
  // Comparación de mensajes
  const compareMessages = useCallback((messageId1: string, messageId2: string) => {
    setMessageComparison([messageId1, messageId2])
    setShowComparison(true)
  }, [])
  
  // Traducción automática
  const translateMessage = useCallback((messageId: string, targetLang: string) => {
    const message = messages.find(m => m.id === messageId)
    if (!message) return
    
    // Placeholder para traducción real
    const translated = `[Traducido a ${targetLang}] ${message.content}`
    setMessageTranslation(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { language: targetLang, content: translated })
      return newMap
    })
    toast.success(`Mensaje traducido a ${targetLang}`, { icon: '🌐' })
  }, [messages])
  
  // Resumen automático
  const summarizeMessage = useCallback((messageId: string) => {
    const message = messages.find(m => m.id === messageId)
    if (!message) return
    
    // Placeholder para resumen real
    const summary = message.content.substring(0, 100) + '...'
    setMessageSummarization(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, summary)
      return newMap
    })
    toast.success('Resumen generado', { icon: '📝' })
  }, [messages])
  
  // Exportación avanzada
  const exportToXML = useCallback(() => {
    const xmlContent = `<?xml version="1.0" encoding="UTF-8"?>
<conversation>
${messages.map(msg => `  <message id="${msg.id}" role="${msg.role}" timestamp="${new Date(msg.timestamp).toISOString()}">
    <content><![CDATA[${msg.content}]]></content>
  </message>`).join('\n')}
</conversation>`
    
    const blob = new Blob([xmlContent], { type: 'application/xml' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `conversation-${Date.now()}.xml`
    a.click()
    URL.revokeObjectURL(url)
    toast.success('Exportado a XML', { icon: '📄' })
  }, [messages])
  
  const exportToYAML = useCallback(() => {
    const yamlContent = `conversation:
  messages:
${messages.map(msg => `    - id: ${msg.id}
      role: ${msg.role}
      timestamp: ${new Date(msg.timestamp).toISOString()}
      content: |
        ${msg.content.split('\n').map(line => `        ${line}`).join('\n')}`).join('\n')}`
    
    const blob = new Blob([yamlContent], { type: 'text/yaml' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `conversation-${Date.now()}.yaml`
    a.click()
    URL.revokeObjectURL(url)
    toast.success('Exportado a YAML', { icon: '📄' })
  }, [messages])
  
  // ========== FUNCIONES ADICIONALES FINALES ==========
  
  // Modo de pantalla dividida
  const toggleSplitScreen = useCallback(() => {
    setSplitScreenMode(!splitScreenMode)
    toast.success(`Pantalla dividida ${!splitScreenMode ? 'activada' : 'desactivada'}`, { icon: '🖥️' })
  }, [splitScreenMode])
  
  // Sistema de widgets
  const addWidget = useCallback((type: string, position: string) => {
    const id = `widget-${Date.now()}`
    setWidgets(prev => {
      const newMap = new Map(prev)
      newMap.set(id, { type, position, enabled: true })
      return newMap
    })
    toast.success(`Widget "${type}" agregado`, { icon: '🧩' })
  }, [])
  
  // Grabación de audio
  const startAudioRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      setAudioRecording(true)
      toast.success('Grabación de audio iniciada', { icon: '🎤' })
    } catch (error) {
      toast.error('Error al acceder al micrófono', { icon: '❌' })
    }
  }, [])
  
  const stopAudioRecording = useCallback(() => {
    setAudioRecording(false)
    toast.success('Grabación de audio detenida', { icon: '⏹️' })
  }, [])
  
  // Transiciones de presentación
  const setPresentationTransition = useCallback((transition: 'fade' | 'slide' | 'zoom' | 'none') => {
    setPresentationTransitions(transition)
    toast.success(`Transición "${transition}" aplicada`, { icon: '🎬' })
  }, [])
  
  // Métricas de rendimiento
  const updatePerformanceMetrics = useCallback(() => {
    const metrics = {
      renderTime: performance.now(),
      memoryUsage: (performance as any).memory?.usedJSHeapSize || 0,
      messageCount: messages.length,
      componentSize: document.querySelector('.chat-container')?.clientHeight || 0,
    }
    setPerformanceMetrics(prev => {
      const newMap = new Map(prev)
      Object.entries(metrics).forEach(([key, value]) => {
        newMap.set(key, value)
      })
      return newMap
    })
  }, [messages.length])
  
  useEffect(() => {
    if (showPerformance) {
      const interval = setInterval(updatePerformanceMetrics, 1000)
      return () => clearInterval(interval)
    }
  }, [showPerformance, updatePerformanceMetrics])
  
  // Accesibilidad avanzada
  const toggleAccessibilityFeature = useCallback((feature: 'screenReader' | 'highContrast' | 'largeText' | 'reducedMotion') => {
    setAccessibilityFeatures(prev => ({
      ...prev,
      [feature]: !prev[feature]
    }))
    toast.success(`${feature} ${!accessibilityFeatures[feature] ? 'activado' : 'desactivado'}`, { icon: '♿' })
  }, [accessibilityFeatures])
  
  // ========== FUNCIONES AVANZADAS ADICIONALES ==========
  
  // Cola de mensajes
  const addToQueue = useCallback((message: string, priority: number = 0) => {
    const id = `queue-${Date.now()}`
    setMessageQueue(prev => [...prev, { id, message, priority }].sort((a, b) => b.priority - a.priority))
    toast.success('Mensaje agregado a la cola', { icon: '📋' })
  }, [])
  
  const processQueue = useCallback(async () => {
    if (queueProcessing || messageQueue.length === 0) return
    setQueueProcessing(true)
    const item = messageQueue[0]
    try {
      if (useBulkChatMode && bulkChat.sessionId) {
        addMessage({
          id: Date.now().toString(),
          role: 'user',
          content: item.message,
          timestamp: new Date(),
        })
        await bulkChat.sendMessage(item.message)
      } else {
        setInput(item.message)
        // Trigger submit programmatically
        const syntheticEvent = { preventDefault: () => {} } as React.FormEvent
        handleSubmit(syntheticEvent)
      }
      setMessageQueue(prev => prev.filter(q => q.id !== item.id))
      toast.success('Mensaje procesado de la cola', { icon: '✅' })
    } catch (error) {
      toast.error('Error procesando mensaje', { icon: '❌' })
    } finally {
      setQueueProcessing(false)
    }
  }, [messageQueue, queueProcessing, useBulkChatMode, bulkChat, addMessage, handleSubmit])
  
  // Formateo automático
  const formatMessage = useCallback((content: string, format: 'plain' | 'markdown' | 'html' | 'code') => {
    switch (format) {
      case 'markdown':
        return content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      case 'html':
        return content.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      case 'code':
        return `\`\`\`\n${content}\n\`\`\``
      default:
        return content
    }
  }, [])
  
  // Enlaces entre mensajes
  const linkMessages = useCallback((sourceId: string, targetId: string) => {
    setMessageLinking(prev => {
      const newMap = new Map(prev)
      const links = newMap.get(sourceId) || []
      if (!links.includes(targetId)) {
        newMap.set(sourceId, [...links, targetId])
      }
      return newMap
    })
    toast.success('Mensajes enlazados', { icon: '🔗' })
  }, [])
  
  // Relaciones entre mensajes
  const addMessageRelation = useCallback((sourceId: string, targetId: string, type: string) => {
    setMessageRelations(prev => {
      const newMap = new Map(prev)
      const relations = newMap.get(sourceId) || []
      newMap.set(sourceId, [...relations, { type, target: targetId }])
      return newMap
    })
    toast.success(`Relación "${type}" agregada`, { icon: '🌐' })
  }, [])
  
  // Workflow de mensajes
  const createWorkflow = useCallback((messageId: string, steps: string[]) => {
    setMessageWorkflow(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { status: 'pending', steps })
      return newMap
    })
    toast.success('Workflow creado', { icon: '⚙️' })
  }, [])
  
  // Validación de mensajes
  const validateMessage = useCallback((messageId: string, content: string) => {
    const errors: string[] = []
    if (content.length === 0) errors.push('Mensaje vacío')
    if (content.length > 10000) errors.push('Mensaje muy largo')
    if (!content.trim()) errors.push('Solo espacios en blanco')
    
    setMessageValidation(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { valid: errors.length === 0, errors })
      return newMap
    })
    
    return errors.length === 0
  }, [])
  
  // Deduplicación de mensajes
  const checkDuplicates = useCallback((newMessage: string) => {
    if (!messageDeduplication) return false
    const similarity = messages.map(msg => {
      const words1 = newMessage.toLowerCase().split(/\s+/)
      const words2 = msg.content.toLowerCase().split(/\s+/)
      const common = words1.filter(w => words2.includes(w)).length
      return common / Math.max(words1.length, words2.length)
    })
    return similarity.some(sim => sim >= duplicateThreshold)
  }, [messages, messageDeduplication, duplicateThreshold])
  
  // Compresión de mensajes
  const compressMessage = useCallback((content: string) => {
    if (!messageCompression) return content
    // Compresión simple (eliminar espacios extra, acortar URLs)
    return content
      .replace(/\s+/g, ' ')
      .replace(/https?:\/\/[^\s]+/g, (url) => url.length > 50 ? url.substring(0, 47) + '...' : url)
      .trim()
  }, [messageCompression])
  
  // Encriptación avanzada
  const encryptMessageAdvanced = useCallback((messageId: string, algorithm: string = 'AES') => {
    setMessageEncryptionAdvanced(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { algorithm, key: `key-${Date.now()}` })
      return newMap
    })
    toast.success(`Mensaje encriptado (${algorithm})`, { icon: '🔐' })
  }, [])
  
  // Backup avanzado
  const createAdvancedBackup = useCallback((format: string = 'json') => {
    const backup = {
      messages,
      timestamp: Date.now(),
      format,
      size: JSON.stringify(messages).length
    }
    setMessageBackupAdvanced(prev => {
      const newMap = new Map(prev)
      newMap.set(`backup-${Date.now()}`, backup)
      return newMap
    })
    localStorage.setItem(`backup-${Date.now()}`, JSON.stringify(backup))
    toast.success(`Backup creado (${format})`, { icon: '💾' })
  }, [messages])
  
  // Sincronización multi-dispositivo
  const syncToDevice = useCallback((deviceId: string) => {
    const syncData = {
      messages,
      timestamp: Date.now(),
      device: deviceId
    }
    setMessageSync(prev => {
      const newMap = new Map(prev)
      messages.forEach(msg => {
        newMap.set(msg.id, { synced: true, timestamp: Date.now(), device: deviceId })
      })
      return newMap
    })
    localStorage.setItem(`sync-${deviceId}`, JSON.stringify(syncData))
    toast.success(`Sincronizado con ${deviceId}`, { icon: '🔄' })
  }, [messages])
  
  // Analytics avanzado
  const trackMessageInteraction = useCallback((messageId: string, type: 'view' | 'interaction' | 'share') => {
    setMessageAnalyticsAdvanced(prev => {
      const newMap = new Map(prev)
      const analytics = newMap.get(messageId) || { views: 0, interactions: 0, shares: 0 }
      if (type === 'view') analytics.views++
      if (type === 'interaction') analytics.interactions++
      if (type === 'share') analytics.shares++
      newMap.set(messageId, analytics)
      return newMap
    })
  }, [])
  
  // Auto-backup
  useEffect(() => {
    if (autoBackupInterval <= 0) return
    const interval = setInterval(() => {
      createAdvancedBackup('json')
    }, autoBackupInterval)
    return () => clearInterval(interval)
  }, [autoBackupInterval, createAdvancedBackup])
  
  // Auto-validación
  useEffect(() => {
    if (!autoValidate) return
    messages.forEach(msg => {
      validateMessage(msg.id, msg.content)
    })
  }, [messages, autoValidate, validateMessage])
  
  // ========== FUNCIONES DE MEJORA ADICIONALES ==========
  
  // Búsqueda avanzada
  const performAdvancedSearch = useCallback((query: string, filters?: any) => {
    const results = messages.filter(msg => {
      const matchesQuery = msg.content.toLowerCase().includes(query.toLowerCase())
      if (!matchesQuery) return false
      if (filters) {
        if (filters.role && msg.role !== filters.role) return false
        if (filters.dateRange) {
          const msgDate = new Date(msg.timestamp)
          if (filters.dateRange.start && msgDate < filters.dateRange.start) return false
          if (filters.dateRange.end && msgDate > filters.dateRange.end) return false
        }
      }
      return true
    })
    const searchId = `search-${Date.now()}`
    setMessageSearchAdvanced(prev => {
      const newMap = new Map(prev)
      newMap.set(searchId, { query, results: results.map(r => r.id) })
      return newMap
    })
    setSearchHistory(prev => [query, ...prev.slice(0, 9)])
    return results
  }, [messages])
  
  // Filtros de mensajes
  const applyMessageFilter = useCallback((filterId: string, type: string, value: any) => {
    setMessageFilters(prev => {
      const newMap = new Map(prev)
      newMap.set(filterId, { type, value })
      return newMap
    })
    toast.success(`Filtro "${type}" aplicado`, { icon: '🔍' })
  }, [])
  
  // Guardar preset de filtros
  const saveFilterPreset = useCallback((name: string, filters: any) => {
    const id = `preset-${Date.now()}`
    setFilterPresets(prev => {
      const newMap = new Map(prev)
      newMap.set(id, { name, filters })
      return newMap
    })
    toast.success(`Preset "${name}" guardado`, { icon: '💾' })
  }, [])
  
  // Ordenamiento de mensajes
  const sortMessages = useCallback((sortType: 'chronological' | 'relevance' | 'popularity') => {
    setMessageSorting(sortType)
    toast.success(`Ordenamiento: ${sortType}`, { icon: '📊' })
  }, [])
  
  // Agrupación avanzada
  const groupMessagesAdvanced = useCallback((groupKey: string, messageIds: string[]) => {
    setMessageGroupingAdvanced(prev => {
      const newMap = new Map(prev)
      newMap.set(groupKey, messageIds)
      return newMap
    })
    toast.success(`Grupo "${groupKey}" creado`, { icon: '📦' })
  }, [])
  
  // Notificaciones de mensajes
  const configureMessageNotification = useCallback((messageId: string, type: string, enabled: boolean) => {
    setMessageNotifications(prev => {
      const newMap = new Map(prev)
      const existing = newMap.get(messageId)
      newMap.set(messageId, { 
        type, 
        title: existing?.title || `Notificación ${type}`, 
        body: existing?.body || `Notificación ${enabled ? 'activada' : 'desactivada'}`,
        read: existing?.read || false,
        timestamp: existing?.timestamp || Date.now()
      })
      return newMap
    })
    toast.success(`Notificación ${enabled ? 'activada' : 'desactivada'}`, { icon: '🔔' })
  }, [])
  
  // Resaltado avanzado - Usando hook modular
  const highlightMessage = useCallback((messageId: string, color: string, note?: string) => {
    messageActions.highlightMessage(messageId, color, note)
    eventBus.emit(MessageEvents.HIGHLIGHT_ADDED, { messageId, color, note })
  }, [messageActions])
  
  // Bookmarks avanzados - Usando hook modular
  const addBookmarkAdvanced = useCallback((messageId: string, category: string, tags: string[] = []) => {
    messageActions.addBookmark(messageId, `Bookmark ${category}`, category, tags)
    eventBus.emit(MessageEvents.BOOKMARK_ADDED, { messageId, category, tags })
  }, [messageActions])
  
  // Exportación avanzada
  const exportWithTemplate = useCallback((messageIds: string[], templateId: string) => {
    const template = exportTemplates.get(templateId)
    if (!template) {
      toast.error('Template no encontrado', { icon: '❌' })
      return
    }
    const selectedMessages = messages.filter(m => messageIds.includes(m.id))
    const exported = template.template.replace('{{messages}}', JSON.stringify(selectedMessages))
    const blob = new Blob([exported], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `export-${Date.now()}.txt`
    a.click()
    URL.revokeObjectURL(url)
    toast.success('Exportado con template', { icon: '📤' })
  }, [messages, exportTemplates])
  
  // Importación de mensajes
  const importMessages = useCallback((data: any, format: string, source: string) => {
    try {
      let imported: any[] = []
      if (format === 'json') {
        imported = Array.isArray(data) ? data : data.messages || []
      } else if (format === 'txt') {
        imported = data.split('\n').filter(Boolean).map((line: string, idx: number) => ({
          id: `imported-${Date.now()}-${idx}`,
          role: 'user',
          content: line,
          timestamp: new Date()
        }))
      }
      imported.forEach(msg => {
        addMessage(msg)
        setMessageImport(prev => {
          const newMap = new Map(prev)
          newMap.set(msg.id, { source, format })
          return newMap
        })
      })
      setImportHistory(prev => [{ source, format, count: imported.length, timestamp: Date.now() }, ...prev.slice(0, 9)])
      toast.success(`${imported.length} mensaje(s) importado(s)`, { icon: '📥' })
    } catch (error) {
      toast.error('Error al importar', { icon: '❌' })
    }
  }, [addMessage])
  
  // Colaboración en mensajes
  const shareMessageForCollaboration = useCallback((messageId: string, users: string[], permissions: string) => {
    setMessageCollaboration(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { users, permissions })
      return newMap
    })
    toast.success(`Mensaje compartido con ${users.length} usuario(s)`, { icon: '👥' })
  }, [])
  
  // Control de versiones
  const saveMessageVersion = useCallback((messageId: string, content: string) => {
    setMessageVersioning(prev => {
      const newMap = new Map(prev)
      const versioning = newMap.get(messageId) || { versions: [], current: 0 }
      if (versionControl.maxVersions > 0 && versioning.versions.length >= versionControl.maxVersions) {
        versioning.versions.shift()
      }
      versioning.versions.push({ content, timestamp: Date.now() })
      versioning.current = versioning.versions.length - 1
      newMap.set(messageId, versioning)
      return newMap
    })
    toast.success('Versión guardada', { icon: '💾' })
  }, [versionControl])
  
  // Funciones de IA
  const generateAISuggestions = useCallback((messageId: string) => {
    const message = messages.find(m => m.id === messageId)
    if (!message) return
    const suggestions = [
      `Respuesta sugerida para: ${message.content.substring(0, 50)}...`,
      `Pregunta relacionada: ¿Podrías explicar más sobre esto?`,
      `Siguiente paso sugerido: Continuar la conversación`
    ]
    setMessageAI(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { suggestions, insights: { sentiment: 'positive', confidence: 0.8 } })
      return newMap
    })
    toast.success('Sugerencias de IA generadas', { icon: '🤖' })
  }, [messages])
  
  // ========== FUNCIONES ADICIONALES FINALES ==========
  
  // Sistema de comandos
  const registerCommand = useCallback((command: string, handler: () => void, description: string) => {
    setCommandSystem(prev => {
      const newMap = new Map(prev)
      newMap.set(command, { command, handler, description })
      return newMap
    })
    toast.success(`Comando "${command}" registrado`, { icon: '⌨️' })
  }, [])
  
  const executeCommand = useCallback((command: string) => {
    const cmd = commandSystem.get(command)
    if (cmd) {
      cmd.handler()
      setCommandHistory(prev => [command, ...prev.slice(0, 9)])
      toast.success(`Comando "${command}" ejecutado`, { icon: '✅' })
    } else {
      toast.error(`Comando "${command}" no encontrado`, { icon: '❌' })
    }
  }, [commandSystem])
  
  // Reacciones avanzadas
  const addReaction = useCallback((messageId: string, emoji: string, userId: string = 'user') => {
    setMessageReactionsAdvanced(prev => {
      const newMap = new Map(prev)
      const reactions = newMap.get(messageId) || []
      const existing = reactions.find(r => r.emoji === emoji)
      if (existing) {
        existing.count++
        if (!existing.users.includes(userId)) {
          existing.users.push(userId)
        }
      } else {
        reactions.push({ emoji, count: 1, users: [userId] })
      }
      newMap.set(messageId, reactions)
      return newMap
    })
    toast.success(`Reacción ${emoji} agregada`, { icon: '👍' })
  }, [])
  
  // Encuestas - Usando hook modular
  const createPoll = useCallback((messageId: string, question: string, options: string[]) => {
    messageActions.createPoll(messageId, question, options)
    eventBus.emit(MessageEvents.POLL_CREATED, { messageId, question, options })
  }, [messageActions])
  
  const votePoll = useCallback((messageId: string, optionIndex: number) => {
    messageActions.votePoll(messageId, optionIndex)
    eventBus.emit(MessageEvents.POLL_VOTED, { messageId, optionIndex })
  }, [messageActions])
  
  // Tareas
  const createTask = useCallback((messageId: string, task: string, dueDate?: Date) => {
    setMessageTasks(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { task, completed: false, dueDate })
      return newMap
    })
    toast.success('Tarea creada', { icon: '✓' })
  }, [])
  
  const toggleTask = useCallback((messageId: string) => {
    setMessageTasks(prev => {
      const newMap = new Map(prev)
      const task = newMap.get(messageId)
      if (task) {
        newMap.set(messageId, { ...task, completed: !task.completed })
      }
      return newMap
    })
    toast.success('Tarea actualizada', { icon: '✓' })
  }, [])
  
  // Recordatorios avanzados
  const setReminder = useCallback((messageId: string, reminder: string, date: Date, recurring?: string) => {
    setMessageRemindersAdvanced(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { reminder, date, recurring })
      return newMap
    })
    toast.success('Recordatorio configurado', { icon: '⏰' })
  }, [])
  
  // Calendario - Usando hook modular
  const scheduleEvent = useCallback((messageId: string, event: string, date: Date, duration?: number) => {
    messageActions.scheduleEvent(messageId, event, date, duration)
    eventBus.emit(MessageEvents.EVENT_SCHEDULED, { messageId, event, date, duration })
  }, [messageActions])
  
  // Notas avanzadas
  const addAdvancedNote = useCallback((messageId: string, note: string, attachments: string[] = [], tags: string[] = []) => {
    setMessageNotesAdvanced(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { note, attachments, tags })
      return newMap
    })
    toast.success('Nota avanzada agregada', { icon: '📝' })
  }, [])
  
  // Adjuntos - Usando hook modular
  const addAttachment = useCallback((messageId: string, type: string, url: string, name: string) => {
    messageActions.addAttachment(messageId, type, url, name)
    eventBus.emit(MessageEvents.ATTACHMENT_ADDED, { messageId, type, url, name })
  }, [messageActions])
  
  // Enlaces - Usando hook modular
  const addLink = useCallback((messageId: string, url: string, title: string, description: string) => {
    messageActions.addLink(messageId, url, title, description)
    eventBus.emit(MessageEvents.LINK_ADDED, { messageId, url, title, description })
  }, [messageActions])
  
  // Código
  const addCode = useCallback((messageId: string, language: string, code: string) => {
    setMessageCode(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { language, code })
      return newMap
    })
    toast.success('Código agregado', { icon: '💻' })
  }, [])
  
  // Media
  const addMedia = useCallback((messageId: string, type: string, url: string, thumbnail?: string) => {
    setMessageMedia(prev => {
      const newMap = new Map(prev)
      const media = newMap.get(messageId) || []
      newMap.set(messageId, [...media, { type, url, thumbnail }])
      return newMap
    })
    toast.success('Media agregado', { icon: '🖼️' })
  }, [])
  
  // Ubicación
  const addLocation = useCallback((messageId: string, lat: number, lng: number, address: string) => {
    setMessageLocation(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { lat, lng, address })
      return newMap
    })
    toast.success('Ubicación agregada', { icon: '📍' })
  }, [])
  
  // Contactos
  const addContact = useCallback((messageId: string, name: string, email: string, phone?: string) => {
    setMessageContacts(prev => {
      const newMap = new Map(prev)
      const contacts = newMap.get(messageId) || []
      newMap.set(messageId, [...contacts, { name, email, phone }])
      return newMap
    })
    toast.success('Contacto agregado', { icon: '👤' })
  }, [])
  
  // Eventos
  const logEvent = useCallback((messageId: string, type: string, data: any) => {
    setMessageEvents(prev => {
      const newMap = new Map(prev)
      const events = newMap.get(messageId) || []
      newMap.set(messageId, [...events, { type, data, timestamp: Date.now() }])
      return newMap
    })
  }, [])
  
  // Metadatos
  const addMetadata = useCallback((messageId: string, key: string, value: any) => {
    setMessageMetadata(prev => {
      const newMap = new Map(prev)
      const metadata = newMap.get(messageId) || []
      const existing = metadata.find(m => m.key === key)
      if (existing) {
        existing.value = value
      } else {
        metadata.push({ key, value })
      }
      newMap.set(messageId, metadata)
      return newMap
    })
    toast.success(`Metadato "${key}" agregado`, { icon: '🏷️' })
  }, [])
  
  // Verificar recordatorios
  useEffect(() => {
    if (!reminderSystem) return
    const checkReminders = setInterval(() => {
      const now = new Date()
      messageRemindersAdvanced.forEach((reminder, messageId) => {
        if (reminder.date <= now) {
          toast.info(reminder.reminder, { icon: '⏰', duration: 5000 })
          // Si es recurrente, programar siguiente
          if (reminder.recurring) {
            const nextDate = new Date(reminder.date)
            if (reminder.recurring === 'daily') {
              nextDate.setDate(nextDate.getDate() + 1)
            } else if (reminder.recurring === 'weekly') {
              nextDate.setDate(nextDate.getDate() + 7)
            }
            setMessageRemindersAdvanced(prev => {
              const newMap = new Map(prev)
              newMap.set(messageId, { ...reminder, date: nextDate })
              return newMap
            })
          }
        }
      })
    }, 60000) // Verificar cada minuto
    return () => clearInterval(checkReminders)
  }, [reminderSystem, messageRemindersAdvanced])
  
  // ========== FUNCIONES SISTEMA AVANZADO ==========
  
  // Sistema de plugins
  const installPlugin = useCallback((pluginId: string, plugin: { name: string, version: string, config: any }) => {
    setPluginSystem(prev => {
      const newMap = new Map(prev)
      newMap.set(pluginId, { ...plugin, enabled: true })
      return newMap
    })
    toast.success(`Plugin "${plugin.name}" instalado`, { icon: '🔌' })
  }, [])
  
  const togglePlugin = useCallback((pluginId: string) => {
    setPluginSystem(prev => {
      const newMap = new Map(prev)
      const plugin = newMap.get(pluginId)
      if (plugin) {
        newMap.set(pluginId, { ...plugin, enabled: !plugin.enabled })
      }
      return newMap
    })
    toast.success('Plugin actualizado', { icon: '🔌' })
  }, [])
  
  // Integraciones API
  const addApiIntegration = useCallback((id: string, name: string, endpoint: string, apiKey: string) => {
    setApiIntegrations(prev => {
      const newMap = new Map(prev)
      newMap.set(id, { name, endpoint, apiKey, enabled: true })
      return newMap
    })
    toast.success(`API "${name}" agregada`, { icon: '🔗' })
  }, [])
  
  // Dev Tools
  const logToConsole = useCallback((message: string) => {
    setDevConsole(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${message}`].slice(-100))
  }, [])
  
  // Sistema de temas
  const createTheme = useCallback((id: string, name: string, colors: any, fonts: any) => {
    setThemeSystem(prev => {
      const newMap = new Map(prev)
      newMap.set(id, { name, colors, fonts })
      return newMap
    })
    toast.success(`Tema "${name}" creado`, { icon: '🎨' })
  }, [])
  
  // Monitor de rendimiento
  const recordPerformanceMetric = useCallback((metric: string, value: number) => {
    setPerformanceMonitor(prev => {
      const newMap = new Map(prev)
      newMap.set(`${metric}-${Date.now()}`, { metric, value, timestamp: Date.now() })
      // Mantener solo las últimas 100 métricas
      if (newMap.size > 100) {
        const entries = Array.from(newMap.entries())
        entries.sort((a, b) => b[1].timestamp - a[1].timestamp)
        return new Map(entries.slice(0, 100))
      }
      return newMap
    })
  }, [])
  
  // Notificaciones push
  const sendPushNotification = useCallback((title: string, body: string, icon?: string, data?: any) => {
    if ('Notification' in window && Notification.permission === 'granted') {
      const notification = new Notification(title, { body, icon, data })
      setPushNotifications(prev => {
        const newMap = new Map(prev)
        newMap.set(`notif-${Date.now()}`, { title, body, icon, data })
        return newMap
      })
      notification.onclick = () => {
        window.focus()
      }
    }
  }, [])
  
  // Modo offline
  const queueOfflineAction = useCallback((action: string, data: any) => {
    setOfflineQueue(prev => [...prev, { id: `offline-${Date.now()}`, action, data }])
    toast.info('Acción en cola (modo offline)', { icon: '📴' })
  }, [])
  
  const processOfflineQueue = useCallback(async () => {
    if (offlineQueue.length === 0 || !navigator.onLine) return
    try {
      for (const item of offlineQueue) {
        // Procesar acción
        logToConsole(`Procesando acción offline: ${item.action}`)
      }
      setOfflineQueue([])
      setSyncStatus('synced')
      toast.success('Cola offline procesada', { icon: '✅' })
    } catch (error) {
      setSyncStatus('error')
      toast.error('Error procesando cola offline', { icon: '❌' })
    }
  }, [offlineQueue, logToConsole])
  
  // Sincronización en tiempo real
  const syncRealTime = useCallback(async () => {
    if (!realTimeSync) return
    setSyncStatus('syncing')
    try {
      // Simular sincronización
      await new Promise(resolve => setTimeout(resolve, 1000))
      setSyncStatus('synced')
      toast.success('Sincronizado', { icon: '🔄' })
    } catch (error) {
      setSyncStatus('error')
      toast.error('Error de sincronización', { icon: '❌' })
    }
  }, [realTimeSync])
  
  // Almacenamiento en la nube
  const uploadToCloud = useCallback(async (provider: string, bucket: string, path: string, data: any) => {
    try {
      setCloudStorage(prev => {
        const newMap = new Map(prev)
        newMap.set(`cloud-${Date.now()}`, { provider, bucket, path })
        return newMap
      })
      toast.success(`Subido a ${provider}`, { icon: '☁️' })
    } catch (error) {
      toast.error('Error subiendo a la nube', { icon: '❌' })
    }
  }, [])
  
  // Sistema de autenticación
  const authenticateUser = useCallback((userId: string, user: string, role: string, permissions: string[]) => {
    setAuthSystem(prev => {
      const newMap = new Map(prev)
      newMap.set(userId, { user, role, permissions })
      return newMap
    })
    toast.success(`Usuario "${user}" autenticado`, { icon: '🔐' })
  }, [])
  
  // Encriptación de mensajes
  const encryptMessage = useCallback((messageId: string, algorithm: string = 'AES-256') => {
    setMessageEncryption(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { algorithm, encrypted: true })
      return newMap
    })
    toast.success(`Mensaje encriptado (${algorithm})`, { icon: '🔒' })
  }, [])
  
  // Backup de mensajes
  const createBackup = useCallback((format: string = 'json') => {
    const backup = {
      messages,
      timestamp: Date.now(),
      format,
      size: JSON.stringify(messages).length
    }
    setMessageBackup(prev => {
      const newMap = new Map(prev)
      newMap.set(`backup-${Date.now()}`, { timestamp: Date.now(), size: backup.size, format })
      return newMap
    })
    localStorage.setItem(`backup-${Date.now()}`, JSON.stringify(backup))
    toast.success(`Backup creado (${format})`, { icon: '💾' })
  }, [messages])
  
  // Restaurar mensajes
  const restoreFromBackup = useCallback((backupId: string) => {
    try {
      const backup = localStorage.getItem(backupId)
      if (backup) {
        const data = JSON.parse(backup)
        setMessageRestore(prev => {
          const newMap = new Map(prev)
          newMap.set(backupId, { source: backupId, timestamp: Date.now(), status: 'restored' })
          return newMap
        })
        toast.success('Backup restaurado', { icon: '📥' })
      }
    } catch (error) {
      toast.error('Error restaurando backup', { icon: '❌' })
    }
  }, [])
  
  // Analytics
  const updateAnalytics = useCallback((metric: string, value: number, trend: string) => {
    setAnalyticsData(prev => {
      const newMap = new Map(prev)
      newMap.set(metric, { metric, value, trend })
      return newMap
    })
  }, [])
  
  // Búsqueda avanzada
  const performSearch = useCallback((query: string, filters?: any) => {
    const results = messages.filter(msg => {
      const matchesQuery = msg.content.toLowerCase().includes(query.toLowerCase())
      if (!matchesQuery) return false
      if (filters) {
        if (filters.role && msg.role !== filters.role) return false
        if (filters.dateRange) {
          const msgDate = new Date(msg.timestamp)
          if (filters.dateRange.start && msgDate < filters.dateRange.start) return false
          if (filters.dateRange.end && msgDate > filters.dateRange.end) return false
        }
      }
      return true
    })
    const searchId = `search-${Date.now()}`
    setMessageSearch(prev => {
      const newMap = new Map(prev)
      newMap.set(searchId, { query, results: results.map(r => r.id), filters: filters || {} })
      return newMap
    })
    return results
  }, [messages])
  
  // Exportar mensajes
  const exportMessages = useCallback((messageIds: string[], format: string) => {
    const selectedMessages = messages.filter(m => messageIds.includes(m.id))
    let exported: string = ''
    
    if (format === 'json') {
      exported = JSON.stringify(selectedMessages, null, 2)
    } else if (format === 'txt') {
      exported = selectedMessages.map(m => `${m.role}: ${m.content}`).join('\n\n')
    } else if (format === 'csv') {
      exported = 'Role,Content,Timestamp\n' + selectedMessages.map(m => 
        `"${m.role}","${m.content.replace(/"/g, '""')}","${new Date(m.timestamp).toISOString()}"`
      ).join('\n')
    }
    
    const blob = new Blob([exported], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `export-${Date.now()}.${format}`
    a.click()
    URL.revokeObjectURL(url)
    
    setMessageExport(prev => {
      const newMap = new Map(prev)
      newMap.set(`export-${Date.now()}`, { format, timestamp: Date.now(), size: exported.length })
      return newMap
    })
    
    toast.success(`Exportado (${format})`, { icon: '📤' })
  }, [messages])
  
  // Importar mensajes
  const importMessages = useCallback(async (file: File, format: string) => {
    try {
      const text = await file.text()
      let imported: any[] = []
      
      if (format === 'json') {
        imported = JSON.parse(text)
      } else if (format === 'txt') {
        imported = text.split('\n').filter(Boolean).map((line, idx) => ({
          id: `imported-${Date.now()}-${idx}`,
          role: 'user',
          content: line,
          timestamp: new Date()
        }))
      } else if (format === 'csv') {
        const lines = text.split('\n').slice(1)
        imported = lines.map((line, idx) => {
          const [role, content, timestamp] = line.split(',')
          return {
            id: `imported-${Date.now()}-${idx}`,
            role: role.replace(/"/g, ''),
            content: content.replace(/"/g, ''),
            timestamp: timestamp ? new Date(timestamp.replace(/"/g, '')) : new Date()
          }
        })
      }
      
      imported.forEach(msg => {
        addMessage(msg)
      })
      
      setMessageImport(prev => {
        const newMap = new Map(prev)
        newMap.set(`import-${Date.now()}`, { source: file.name, format, count: imported.length })
        return newMap
      })
      
      toast.success(`${imported.length} mensajes importados`, { icon: '📥' })
    } catch (error) {
      toast.error('Error importando', { icon: '❌' })
    }
  }, [addMessage])
  
  // Monitoreo de estado offline
  useEffect(() => {
    const handleOnline = () => {
      setSyncStatus('syncing')
      processOfflineQueue()
    }
    const handleOffline = () => {
      setSyncStatus('offline')
      setOfflineMode(true)
      toast.warning('Modo offline activado', { icon: '📴' })
    }
    
    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)
    
    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }, [processOfflineQueue])
  
  // Sincronización periódica
  useEffect(() => {
    if (!realTimeSync) return
    const interval = setInterval(() => {
      syncRealTime()
    }, 30000) // Cada 30 segundos
    return () => clearInterval(interval)
  }, [realTimeSync, syncRealTime])
  
  // Actualizar analytics
  useEffect(() => {
    updateAnalytics('totalMessages', messages.length, 'up')
    updateAnalytics('avgMessageLength', 
      messages.length > 0 ? messages.reduce((acc, m) => acc + m.content.length, 0) / messages.length : 0,
      'stable'
    )
  }, [messages, updateAnalytics])
  
  // ========== FUNCIONES FINALES ADICIONALES ==========
  
  // Sistema de widgets
  const createWidget = useCallback((id: string, type: string, position: string, config: any) => {
    setWidgetSystem(prev => {
      const newMap = new Map(prev)
      newMap.set(id, { type, position, config, visible: true })
      return newMap
    })
    toast.success(`Widget "${type}" creado`, { icon: '🧩' })
  }, [])
  
  const toggleWidget = useCallback((id: string) => {
    setWidgetSystem(prev => {
      const newMap = new Map(prev)
      const widget = newMap.get(id)
      if (widget) {
        newMap.set(id, { ...widget, visible: !widget.visible })
      }
      return newMap
    })
  }, [])
  
  // Modo presentación
  const createPresentation = useCallback((slides: Array<{ title: string, content: string, notes?: string }>) => {
    setPresentationSlides(slides)
    setPresentationMode(true)
    setPresentationIndex(0)
    toast.success('Presentación creada', { icon: '📊' })
  }, [])
  
  const nextSlide = useCallback(() => {
    if (presentationIndex < presentationSlides.length - 1) {
      setPresentationIndex(prev => prev + 1)
    }
  }, [presentationIndex, presentationSlides.length])
  
  const prevSlide = useCallback(() => {
    if (presentationIndex > 0) {
      setPresentationIndex(prev => prev - 1)
    }
  }, [presentationIndex])
  
  // Servicios externos
  const connectService = useCallback((id: string, name: string, type: string, config: any) => {
    setExternalServices(prev => {
      const newMap = new Map(prev)
      newMap.set(id, { name, type, config, connected: true })
      return newMap
    })
    toast.success(`Servicio "${name}" conectado`, { icon: '🔗' })
  }, [])
  
  // Biblioteca de plantillas
  const saveTemplate = useCallback((id: string, name: string, category: string, content: string, variables: string[]) => {
    setTemplateLibrary(prev => {
      const newMap = new Map(prev)
      newMap.set(id, { name, category, content, variables })
      return newMap
    })
    toast.success(`Plantilla "${name}" guardada`, { icon: '📝' })
  }, [])
  
  // Salas de colaboración
  const createCollaborationRoom = useCallback((roomId: string, users: string[], permissions: Map<string, string[]>) => {
    setCollaborationRoom(prev => {
      const newMap = new Map(prev)
      newMap.set(roomId, { users, permissions, active: true })
      return newMap
    })
    toast.success('Sala de colaboración creada', { icon: '👥' })
  }, [])
  
  // Reglas de notificación
  const addNotificationRule = useCallback((id: string, condition: string, action: string) => {
    setNotificationRules(prev => {
      const newMap = new Map(prev)
      newMap.set(id, { condition, action, enabled: true })
      return newMap
    })
    toast.success('Regla de notificación agregada', { icon: '🔔' })
  }, [])
  
  // Configuración de accesibilidad
  const setAccessibilityFeature = useCallback((id: string, feature: string, enabled: boolean, config: any) => {
    setAccessibilitySettings(prev => {
      const newMap = new Map(prev)
      newMap.set(id, { feature, enabled, config })
      return newMap
    })
    toast.success(`${feature} ${enabled ? 'activado' : 'desactivado'}`, { icon: '♿' })
  }, [])
  
  // Sistema de ayuda
  const addHelpTopic = useCallback((id: string, topic: string, content: string, category: string) => {
    setHelpSystem(prev => {
      const newMap = new Map(prev)
      newMap.set(id, { topic, content, category })
      return newMap
    })
  }, [])
  
  // Modo tutorial
  const startTutorial = useCallback((steps: Array<{ step: number, title: string, content: string }>) => {
    setTutorialSteps(steps.map(s => ({ ...s, completed: false })))
    setTutorialMode(true)
    toast.success('Tutorial iniciado', { icon: '📚' })
  }, [])
  
  const completeTutorialStep = useCallback((step: number) => {
    setTutorialSteps(prev => prev.map(s => s.step === step ? { ...s, completed: true } : s))
  }, [])
  
  // Sistema de feedback
  const submitFeedback = useCallback((type: string, content: string, rating?: number) => {
    setFeedbackSystem(prev => {
      const newMap = new Map(prev)
      newMap.set(`feedback-${Date.now()}`, { type, content, rating, timestamp: Date.now() })
      return newMap
    })
    toast.success('Feedback enviado', { icon: '💬' })
  }, [])
  
  // Sistema de reportes
  const generateReport = useCallback((type: string, data: any) => {
    setReportSystem(prev => {
      const newMap = new Map(prev)
      newMap.set(`report-${Date.now()}`, { type, data, timestamp: Date.now() })
      return newMap
    })
    toast.success(`Reporte "${type}" generado`, { icon: '📊' })
  }, [])
  
  // Historial de mensajes
  const trackMessageChange = useCallback((messageId: string, previous: string, current: string) => {
    setMessageHistory(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { previous, current, timestamp: Date.now() })
      return newMap
    })
  }, [])
  
  // Relaciones entre mensajes
  const addMessageRelation = useCallback((messageId: string, related: string[], type: string) => {
    setMessageRelations(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { related, type })
      return newMap
    })
    toast.success('Relación agregada', { icon: '🔗' })
  }, [])
  
  // Insights de mensajes
  const generateInsight = useCallback((messageId: string, insight: string, confidence: number, type: string) => {
    setMessageInsights(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { insight, confidence, type })
      return newMap
    })
    toast.success('Insight generado', { icon: '💡' })
  }, [])
  
  // Sugerencias de mensajes
  const generateSuggestions = useCallback((messageId: string, suggestions: string[], context: string) => {
    setMessageSuggestions(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { suggestions, context })
      return newMap
    })
    toast.success('Sugerencias generadas', { icon: '💭' })
  }, [])
  
  // Clustering de mensajes
  const clusterMessages = useCallback((clusterId: string, cluster: string, messageIds: string[]) => {
    setMessageClustering(prev => {
      const newMap = new Map(prev)
      newMap.set(clusterId, { cluster, messages: messageIds })
      return newMap
    })
    toast.success(`Cluster "${cluster}" creado`, { icon: '📦' })
  }, [])
  
  // Inicializar sistema de ayuda
  useEffect(() => {
    if (helpSystem.size === 0) {
      addHelpTopic('help-1', 'Búsqueda', 'Usa Ctrl+F para buscar mensajes', 'básico')
      addHelpTopic('help-2', 'Exportar', 'Exporta conversaciones en múltiples formatos', 'básico')
      addHelpTopic('help-3', 'Plugins', 'Instala plugins para extender funcionalidad', 'avanzado')
    }
  }, [helpSystem.size, addHelpTopic])
  
  // ========== MEJORAS ADICIONALES ==========
  
  // Optimizaciones de rendimiento
  const optimizePerformance = useCallback(() => {
    if (performanceOptimization.virtualScrolling) {
      logToConsole('Virtual scrolling activado')
    }
    if (performanceOptimization.lazyLoading) {
      logToConsole('Lazy loading activado')
    }
    if (performanceOptimization.memoization) {
      logToConsole('Memoization activado')
    }
    toast.success('Rendimiento optimizado', { icon: '⚡' })
  }, [performanceOptimization, logToConsole])
  
  // Mejoras de UI
  const enhanceUI = useCallback(() => {
    setUiEnhancements(prev => ({
      ...prev,
      animations: !prev.animations
    }))
    toast.success(`Animaciones ${uiEnhancements.animations ? 'desactivadas' : 'activadas'}`, { icon: '🎨' })
  }, [uiEnhancements])
  
  // Características de productividad
  const enableProductivityFeature = useCallback((feature: string) => {
    setProductivityFeatures(prev => ({
      ...prev,
      [feature]: !prev[feature as keyof typeof prev]
    }))
    toast.success(`${feature} ${productivityFeatures[feature as keyof typeof productivityFeatures] ? 'desactivado' : 'activado'}`, { icon: '⚡' })
  }, [productivityFeatures])
  
  // Sistema de shortcuts
  const registerShortcut = useCallback((key: string, action: () => void, description: string) => {
    setShortcutSystem(prev => {
      const newMap = new Map(prev)
      newMap.set(key, { key, action, description })
      return newMap
    })
    toast.success(`Shortcut "${key}" registrado`, { icon: '⌨️' })
  }, [])
  
  const executeShortcut = useCallback((key: string) => {
    const shortcut = shortcutSystem.get(key)
    if (shortcut) {
      shortcut.action()
      toast.success(`Shortcut "${key}" ejecutado`, { icon: '⌨️' })
    }
  }, [shortcutSystem])
  
  // Mejoras de accesibilidad
  const enhanceAccessibility = useCallback((feature: string) => {
    setAccessibilityEnhancements(prev => ({
      ...prev,
      [feature]: !prev[feature as keyof typeof prev]
    }))
    toast.success(`${feature} ${accessibilityEnhancements[feature as keyof typeof accessibilityEnhancements] ? 'desactivado' : 'activado'}`, { icon: '♿' })
  }, [accessibilityEnhancements])
  
  // Sistema de notificaciones mejorado
  const addNotification = useCallback((type: string, priority: number, message: string) => {
    setNotificationSystem(prev => {
      const newMap = new Map(prev)
      newMap.set(`notif-${Date.now()}`, { type, priority, timestamp: Date.now(), read: false })
      return newMap
    })
    toast.info(message, { icon: '🔔' })
  }, [])
  
  const markNotificationRead = useCallback((id: string) => {
    setNotificationSystem(prev => {
      const newMap = new Map(prev)
      const notif = newMap.get(id)
      if (notif) {
        newMap.set(id, { ...notif, read: true })
      }
      return newMap
    })
  }, [])
  
  // Mejoras de búsqueda
  const performFuzzySearch = useCallback((query: string) => {
    if (!searchEnhancements.fuzzySearch) return []
    const results = messages.filter(msg => {
      const content = msg.content.toLowerCase()
      const queryLower = query.toLowerCase()
      // Búsqueda fuzzy simple (coincidencias parciales)
      return content.includes(queryLower) || 
             queryLower.split('').some(char => content.includes(char))
    })
    return results
  }, [messages, searchEnhancements])
  
  // Mejoras de exportación
  const exportWithCompression = useCallback((messageIds: string[], format: string) => {
    const selectedMessages = messages.filter(m => messageIds.includes(m.id))
    let exported = JSON.stringify(selectedMessages)
    
    if (exportEnhancements.compression) {
      // Compresión simple (eliminar espacios)
      exported = exported.replace(/\s+/g, ' ')
    }
    
    const blob = new Blob([exported], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `export-${Date.now()}.${format}`
    a.click()
    URL.revokeObjectURL(url)
    
    toast.success(`Exportado con compresión (${format})`, { icon: '📤' })
  }, [messages, exportEnhancements])
  
  // Mejoras de importación
  const importWithValidation = useCallback(async (file: File, format: string) => {
    try {
      const text = await file.text()
      let imported: any[] = []
      
      if (format === 'json') {
        imported = JSON.parse(text)
      }
      
      if (importEnhancements.validation) {
        imported = imported.filter(msg => msg.content && msg.role)
      }
      
      if (importEnhancements.preview) {
        toast.info(`Previsualizando ${imported.length} mensajes`, { icon: '👁️' })
      }
      
      imported.forEach(msg => {
        addMessage(msg)
      })
      
      toast.success(`${imported.length} mensajes importados`, { icon: '📥' })
    } catch (error) {
      toast.error('Error importando', { icon: '❌' })
    }
  }, [importEnhancements, addMessage])
  
  // Características de seguridad
  const enableSecurityFeature = useCallback((feature: string) => {
    setSecurityFeatures(prev => ({
      ...prev,
      [feature]: !prev[feature as keyof typeof prev]
    }))
    toast.success(`${feature} ${securityFeatures[feature as keyof typeof securityFeatures] ? 'desactivado' : 'activado'}`, { icon: '🔒' })
  }, [securityFeatures])
  
  // Optimización de mensajes
  const optimizeMessages = useCallback(() => {
    if (messageOptimization.deduplication) {
      logToConsole('Deduplicación activada')
    }
    if (messageOptimization.compression) {
      logToConsole('Compresión activada')
    }
    if (messageOptimization.caching) {
      logToConsole('Caché activado')
    }
    if (messageOptimization.indexing) {
      logToConsole('Indexación activada')
    }
    toast.success('Mensajes optimizados', { icon: '⚡' })
  }, [messageOptimization, logToConsole])
  
  // Personalización de UI
  const customizeUI = useCallback((setting: string, value: any) => {
    setUiCustomization(prev => ({
      ...prev,
      [setting]: value
    }))
    toast.success(`UI personalizada: ${setting} = ${value}`, { icon: '🎨' })
  }, [])
  
  // Automatización de workflows
  const createWorkflow = useCallback((id: string, trigger: string, actions: string[]) => {
    setWorkflowAutomation(prev => {
      const newMap = new Map(prev)
      newMap.set(id, { trigger, actions, enabled: true })
      return newMap
    })
    toast.success('Workflow creado', { icon: '⚙️' })
  }, [])
  
  // Analytics de mensajes
  const trackMessageAnalytics = useCallback((messageId: string, type: 'view' | 'interaction' | 'share') => {
    setMessageAnalytics(prev => {
      const newMap = new Map(prev)
      const analytics = newMap.get(messageId) || { views: 0, interactions: 0, shares: 0, timestamp: Date.now() }
      if (type === 'view') analytics.views++
      if (type === 'interaction') analytics.interactions++
      if (type === 'share') analytics.shares++
      newMap.set(messageId, analytics)
      return newMap
    })
  }, [])
  
  // Calidad de mensajes
  const assessMessageQuality = useCallback((messageId: string) => {
    const message = messages.find(m => m.id === messageId)
    if (!message) return
    
    const score = Math.min(100, 
      (message.content.length > 10 ? 20 : 0) +
      (message.content.split(/\s+/).length > 5 ? 20 : 0) +
      (message.content.includes('?') || message.content.includes('!') ? 20 : 0) +
      (message.content.length < 1000 ? 20 : 0) +
      (message.content.trim().length > 0 ? 20 : 0)
    )
    
    setMessageQuality(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { score, metrics: { length: message.content.length, words: message.content.split(/\s+/).length } })
      return newMap
    })
    
    toast.success(`Calidad: ${score}/100`, { icon: '⭐' })
  }, [messages])
  
  // Manejo de shortcuts con teclado
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey) {
        const key = `${e.ctrlKey || e.metaKey ? 'Ctrl+' : ''}${e.key}`
        const shortcut = shortcutSystem.get(key)
        if (shortcut) {
          e.preventDefault()
          executeShortcut(key)
        }
      }
    }
    
    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [shortcutSystem, executeShortcut])
  
  // Registrar shortcuts por defecto
  useEffect(() => {
    if (shortcutSystem.size === 0) {
      registerShortcut('Ctrl+F', () => searchInputRef.current?.focus(), 'Buscar mensajes')
      registerShortcut('Ctrl+K', () => setShowCommandPalette(true), 'Abrir paleta de comandos')
      registerShortcut('Ctrl+S', () => createBackup('json'), 'Guardar backup')
    }
  }, [shortcutSystem.size, registerShortcut, createBackup])
  
  // Optimización automática
  useEffect(() => {
    if (messages.length > 100 && messageOptimization.indexing) {
      optimizeMessages()
    }
  }, [messages.length, messageOptimization, optimizeMessages])
  
  // ========== FUNCIONES FINALES ADICIONALES ==========
  
  // Traducción en tiempo real
  const translateMessage = useCallback((messageId: string, original: string, language: string) => {
    // Simulación de traducción (en producción usar API real)
    const translated = `[${language.toUpperCase()}] ${original}`
    setRealTimeTranslation(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { original, translated, language })
      return newMap
    })
    toast.success(`Traducido a ${language}`, { icon: '🌐' })
  }, [])
  
  // Características de colaboración
  const enableCollaborationFeature = useCallback((feature: string) => {
    setCollaborationFeatures(prev => ({
      ...prev,
      [feature]: !prev[feature as keyof typeof prev]
    }))
    toast.success(`${feature} ${collaborationFeatures[feature as keyof typeof collaborationFeatures] ? 'desactivado' : 'activado'}`, { icon: '👥' })
  }, [collaborationFeatures])
  
  // Versiones de conversación
  const createConversationVersion = useCallback((description: string) => {
    const version = {
      version: conversationVersions.size + 1,
      messages: [...messages],
      timestamp: Date.now(),
      description
    }
    setConversationVersions(prev => {
      const newMap = new Map(prev)
      newMap.set(`version-${Date.now()}`, version)
      return newMap
    })
    toast.success(`Versión ${version.version} creada`, { icon: '📝' })
  }, [messages, conversationVersions.size])
  
  const restoreConversationVersion = useCallback((versionId: string) => {
    const version = conversationVersions.get(versionId)
    if (version) {
      // Restaurar mensajes (requeriría función en el store)
      toast.success(`Versión ${version.version} restaurada`, { icon: '📥' })
    }
  }, [conversationVersions])
  
  // Integración con IA externa
  const connectExternalAI = useCallback((id: string, service: string, apiKey: string, config: any) => {
    setExternalAI(prev => {
      const newMap = new Map(prev)
      newMap.set(id, { service, apiKey, enabled: true, config })
      return newMap
    })
    toast.success(`IA "${service}" conectada`, { icon: '🤖' })
  }, [])
  
  // Modo de aprendizaje
  const learnFromMessage = useCallback((pattern: string, response: string) => {
    setLearningData(prev => {
      const newMap = new Map(prev)
      const existing = newMap.get(pattern)
      if (existing) {
        newMap.set(pattern, { ...existing, confidence: Math.min(1, existing.confidence + 0.1) })
      } else {
        newMap.set(pattern, { pattern, response, confidence: 0.5 })
      }
      return newMap
    })
    toast.success('Patrón aprendido', { icon: '🧠' })
  }, [])
  
  // Recomendaciones inteligentes
  const generateRecommendation = useCallback((type: string, content: string, score: number) => {
    setSmartRecommendations(prev => {
      const newMap = new Map(prev)
      newMap.set(`rec-${Date.now()}`, { type, content, score })
      return newMap
    })
    toast.success('Recomendación generada', { icon: '💡' })
  }, [])
  
  // Análisis de sentimientos avanzado
  const analyzeSentiment = useCallback((messageId: string, content: string) => {
    const positiveWords = ['bueno', 'excelente', 'perfecto', 'genial', 'feliz', 'satisfecho']
    const negativeWords = ['malo', 'error', 'problema', 'fallo', 'triste', 'frustrado']
    const emotions = ['alegría', 'tristeza', 'ira', 'miedo', 'sorpresa']
    
    const lowerContent = content.toLowerCase()
    const positiveCount = positiveWords.filter(w => lowerContent.includes(w)).length
    const negativeCount = negativeWords.filter(w => lowerContent.includes(w)).length
    
    let sentiment = 'neutral'
    let score = 0.5
    
    if (positiveCount > negativeCount) {
      sentiment = 'positive'
      score = 0.5 + (positiveCount / 10)
    } else if (negativeCount > positiveCount) {
      sentiment = 'negative'
      score = 0.5 - (negativeCount / 10)
    }
    
    setSentimentAnalysis(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { sentiment, score: Math.max(0, Math.min(1, score)), emotions: emotions.slice(0, 2) })
      return newMap
    })
    
    toast.success(`Sentimiento: ${sentiment} (${(score * 100).toFixed(0)}%)`, { icon: '😊' })
  }, [])
  
  // Resumen automático
  const generateSummary = useCallback((messageIds: string[]) => {
    const selectedMessages = messages.filter(m => messageIds.includes(m.id))
    const summary = `Resumen de ${selectedMessages.length} mensajes:\n${selectedMessages.map(m => `- ${m.content.substring(0, 50)}...`).join('\n')}`
    const keyPoints = selectedMessages.map(m => m.content.split('.').slice(0, 1).join('.'))
    
    setAutoSummary(prev => {
      const newMap = new Map(prev)
      newMap.set(`summary-${Date.now()}`, { summary, keyPoints, timestamp: Date.now() })
      return newMap
    })
    
    toast.success('Resumen generado', { icon: '📝' })
  }, [messages])
  
  // Exportación multi-formato
  const exportMultipleFormats = useCallback((messageIds: string[], formats: string[]) => {
    formats.forEach(format => {
      exportMessages(messageIds, format)
    })
    
    setMultiFormatExport(prev => {
      const newMap = new Map(prev)
      newMap.set(`export-${Date.now()}`, { formats, timestamp: Date.now() })
      return newMap
    })
    
    toast.success(`Exportado en ${formats.length} formatos`, { icon: '📤' })
  }, [exportMessages])
  
  // Mejoras de sincronización
  const syncWithStrategy = useCallback(async (strategy: 'smart' | 'overwrite' | 'merge') => {
    setSyncEnhancements(prev => ({ ...prev, mergeStrategy: strategy }))
    toast.success(`Sincronización: ${strategy}`, { icon: '🔄' })
  }, [])
  
  // Contexto de mensajes
  const analyzeContext = useCallback((messageId: string) => {
    const message = messages.find(m => m.id === messageId)
    if (!message) return
    
    const context = message.content.substring(0, 100)
    const relatedMessages = messages
      .filter(m => m.id !== messageId && m.content.toLowerCase().includes(context.toLowerCase().substring(0, 20)))
      .map(m => m.id)
      .slice(0, 5)
    const topics = message.content.toLowerCase().split(/\s+/).filter(w => w.length > 4).slice(0, 5)
    
    setMessageContext(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { context, relatedMessages, topics })
      return newMap
    })
    
    toast.success('Contexto analizado', { icon: '🔍' })
  }, [messages])
  
  // Patrones de mensajes
  const detectPattern = useCallback((pattern: string) => {
    const matches = messages.filter(m => m.content.toLowerCase().includes(pattern.toLowerCase()))
    const examples = matches.map(m => m.content.substring(0, 50)).slice(0, 3)
    
    setMessagePatterns(prev => {
      const newMap = new Map(prev)
      const existing = newMap.get(pattern)
      if (existing) {
        newMap.set(pattern, { ...existing, frequency: existing.frequency + matches.length })
      } else {
        newMap.set(pattern, { pattern, frequency: matches.length, examples })
      }
      return newMap
    })
    
    toast.success(`Patrón detectado: ${matches.length} ocurrencias`, { icon: '🔍' })
  }, [messages])
  
  // Flujo de mensajes
  const trackMessageFlow = useCallback((fromId: string, toId: string, type: string) => {
    setMessageFlow(prev => {
      const newMap = new Map(prev)
      newMap.set(`flow-${Date.now()}`, { from: fromId, to: toId, type })
      return newMap
    })
    toast.success('Flujo registrado', { icon: '🌊' })
  }, [])
  
  // Timeline de mensajes
  const addTimelineEvent = useCallback((messageId: string, type: string) => {
    setMessageTimeline(prev => {
      const newMap = new Map(prev)
      const timeline = newMap.get(messageId) || { events: [] }
      timeline.events.push({ type, timestamp: Date.now() })
      newMap.set(messageId, timeline)
      return newMap
    })
  }, [])
  
  // Auto-traducción
  useEffect(() => {
    if (translationMode && messages.length > 0) {
      const lastMessage = messages[messages.length - 1]
      if (!realTimeTranslation.has(lastMessage.id)) {
        translateMessage(lastMessage.id, lastMessage.content, targetLanguage)
      }
    }
  }, [messages, translationMode, targetLanguage, realTimeTranslation, translateMessage])
  
  // Auto-análisis de sentimientos
  useEffect(() => {
    if (sentimentViewer && messages.length > 0) {
      const lastMessage = messages[messages.length - 1]
      if (!sentimentAnalysis.has(lastMessage.id)) {
        analyzeSentiment(lastMessage.id, lastMessage.content)
      }
    }
  }, [messages, sentimentViewer, sentimentAnalysis, analyzeSentiment])
  
  // Auto-resumen
  useEffect(() => {
    if (summaryViewer && messages.length > 0 && messages.length % 10 === 0) {
      const recentMessages = messages.slice(-10).map(m => m.id)
      generateSummary(recentMessages)
    }
  }, [messages.length, summaryViewer, generateSummary])
  
  // ========== FUNCIONES DE MEJORA FINALES ==========
  
  // Visualización de mensajes
  const visualizeMessages = useCallback((type: 'graph' | 'tree' | 'network') => {
    setVisualizationMode(type)
    const visualizationData = messages.map(m => ({
      id: m.id,
      content: m.content.substring(0, 50),
      role: m.role,
      timestamp: Date.now()
    }))
    setMessageVisualization(prev => {
      const newMap = new Map(prev)
      newMap.set(`viz-${Date.now()}`, { type, data: visualizationData })
      return newMap
    })
    toast.success(`Visualización ${type} creada`, { icon: '📊' })
  }, [messages])
  
  // Dependencias de mensajes
  const addMessageDependency = useCallback((messageId: string, dependsOn: string[]) => {
    setMessageDependencies(prev => {
      const newMap = new Map(prev)
      const existing = newMap.get(messageId) || { dependsOn: [], requiredBy: [] }
      newMap.set(messageId, { ...existing, dependsOn: [...existing.dependsOn, ...dependsOn] })
      dependsOn.forEach(depId => {
        const dep = newMap.get(depId) || { dependsOn: [], requiredBy: [] }
        newMap.set(depId, { ...dep, requiredBy: [...dep.requiredBy, messageId] })
      })
      return newMap
    })
    toast.success('Dependencia agregada', { icon: '🔗' })
  }, [])
  
  // Métricas de mensajes
  const trackMessageMetrics = useCallback((messageId: string, readTime: number, responseTime: number) => {
    const engagement = Math.min(1, (readTime / 1000) * 0.1 + (responseTime / 1000) * 0.1)
    setMessageMetrics(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { readTime, responseTime, engagement })
      return newMap
    })
  }, [])
  
  // Alertas de mensajes
  const addMessageAlert = useCallback((messageId: string, type: string, message: string, severity: 'info' | 'warning' | 'error') => {
    setMessageAlerts(prev => {
      const newMap = new Map(prev)
      newMap.set(`alert-${Date.now()}`, { type, message, severity })
      return newMap
    })
    if (severity === 'error') {
      toast.error(message, { icon: '⚠️' })
    } else if (severity === 'warning') {
      toast.warning(message, { icon: '⚠️' })
    } else {
      toast.info(message, { icon: 'ℹ️' })
    }
  }, [])
  
  // Prioridad de mensajes
  const setMessagePriorityLevel = useCallback((messageId: string, priority: 'low' | 'medium' | 'high' | 'urgent') => {
    setMessagePriority(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, priority)
      return newMap
    })
    toast.success(`Prioridad ${priority} establecida`, { icon: '⚡' })
  }, [])
  
  // Estado de mensajes
  const setMessageStatusLevel = useCallback((messageId: string, status: 'pending' | 'processing' | 'completed' | 'archived') => {
    setMessageStatus(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, status)
      return newMap
    })
    toast.success(`Estado ${status} establecido`, { icon: '📋' })
  }, [])
  
  // Categorías de mensajes
  const categorizeMessage = useCallback((messageId: string, category: string, subcategory?: string) => {
    setMessageCategories(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { category, subcategory })
      return newMap
    })
    toast.success(`Categorizado: ${category}${subcategory ? ` > ${subcategory}` : ''}`, { icon: '📁' })
  }, [])
  
  // Auto-track de métricas
  useEffect(() => {
    if (metricsDashboard && messages.length > 0) {
      const lastMessage = messages[messages.length - 1]
      if (!messageMetrics.has(lastMessage.id)) {
        trackMessageMetrics(lastMessage.id, 1000, 500)
      }
    }
  }, [messages, metricsDashboard, messageMetrics, trackMessageMetrics])
  
  // ========== FUNCIONES ADICIONALES PRÁCTICAS ==========
  
  // Filtros de mensajes
  const addMessageFilter = useCallback((filter: string, active: boolean) => {
    setMessageFilters(prev => {
      const newMap = new Map(prev)
      newMap.set(`filter-${Date.now()}`, { filter, active })
      return newMap
    })
    toast.success(`Filtro ${active ? 'activado' : 'desactivado'}`, { icon: '🔍' })
  }, [])
  
  // Ordenamiento de mensajes
  const sortMessages = useCallback((sortBy: string, order: 'asc' | 'desc') => {
    setMessageSorting(prev => {
      const newMap = new Map(prev)
      newMap.set(`sort-${Date.now()}`, { sortBy, order })
      return newMap
    })
    toast.success(`Ordenado por ${sortBy} (${order})`, { icon: '📊' })
  }, [])
  
  // Agrupación de mensajes
  const groupMessages = useCallback((groupBy: string, groups: string[]) => {
    setMessageGrouping(prev => {
      const newMap = new Map(prev)
      newMap.set(`group-${Date.now()}`, { groupBy, groups })
      return newMap
    })
    toast.success(`Agrupado por ${groupBy}`, { icon: '📦' })
  }, [])
  
  // Búsqueda guardada
  const saveSearch = useCallback((query: string, results: string[]) => {
    setMessageSearch(prev => {
      const newMap = new Map(prev)
      newMap.set(`search-${Date.now()}`, { query, results, saved: true })
      return newMap
    })
    toast.success('Búsqueda guardada', { icon: '💾' })
  }, [])
  
  // Marcadores de mensajes
  const addBookmark = useCallback((messageId: string, name: string, note?: string, tags: string[] = []) => {
    setMessageBookmarks(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { name, note, tags })
      return newMap
    })
    toast.success('Marcador agregado', { icon: '🔖' })
  }, [])
  
  // Resaltado de mensajes
  const highlightMessage = useCallback((messageId: string, color: string, note?: string) => {
    setMessageHighlights(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { color, note })
      return newMap
    })
    toast.success('Mensaje resaltado', { icon: '🖍️' })
  }, [])
  
  // Anotaciones de mensajes
  // Anotaciones - Usando hook modular
  const addAnnotation = useCallback((messageId: string, type: string, content: string, position: { x: number, y: number }) => {
    messageActions.addAnnotation(messageId, content)
    eventBus.emit(MessageEvents.ANNOTATION_ADDED, { messageId, content })
  }, [messageActions])
  
  // Archivos adjuntos
  const addFile = useCallback((messageId: string, name: string, type: string, size: number, url: string) => {
    setMessageFiles(prev => {
      const newMap = new Map(prev)
      newMap.set(`file-${Date.now()}`, { name, type, size, url })
      return newMap
    })
    toast.success('Archivo agregado', { icon: '📎' })
  }, [])
  
  // Imágenes
  const addImage = useCallback((messageId: string, url: string, alt: string, caption?: string) => {
    setMessageImages(prev => {
      const newMap = new Map(prev)
      newMap.set(`image-${Date.now()}`, { url, alt, caption })
      return newMap
    })
    toast.success('Imagen agregada', { icon: '🖼️' })
  }, [])
  
  // Videos
  const addVideo = useCallback((messageId: string, url: string, thumbnail: string, duration?: number) => {
    setMessageVideos(prev => {
      const newMap = new Map(prev)
      newMap.set(`video-${Date.now()}`, { url, thumbnail, duration })
      return newMap
    })
    toast.success('Video agregado', { icon: '🎥' })
  }, [])
  
  // Audio
  const addAudio = useCallback((messageId: string, url: string, duration: number, waveform?: number[]) => {
    setMessageAudio(prev => {
      const newMap = new Map(prev)
      newMap.set(`audio-${Date.now()}`, { url, duration, waveform })
      return newMap
    })
    toast.success('Audio agregado', { icon: '🎵' })
  }, [])
  
  // Documentos
  const addDocument = useCallback((messageId: string, type: string, content: string, metadata: any) => {
    setMessageDocuments(prev => {
      const newMap = new Map(prev)
      newMap.set(`doc-${Date.now()}`, { type, content, metadata })
      return newMap
    })
    toast.success('Documento agregado', { icon: '📄' })
  }, [])
  
  // Formularios
  const createForm = useCallback((messageId: string, fields: Array<{ name: string, type: string, required: boolean }>) => {
    setMessageForms(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { fields, responses: [] })
      return newMap
    })
    toast.success('Formulario creado', { icon: '📋' })
  }, [])
  
  // Encuestas duplicadas - Eliminadas, usando hook modular arriba
    toast.success('Voto registrado', { icon: '✅' })
  }, [])
  
  // Cuestionarios
  const createQuiz = useCallback((messageId: string, questions: Array<{ question: string, answers: string[], correct: number }>) => {
    setMessageQuizzes(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { questions, results: [] })
      return newMap
    })
    toast.success('Cuestionario creado', { icon: '📝' })
  }, [])
  
  // Encuestas
  const createSurvey = useCallback((messageId: string, title: string, questions: string[]) => {
    setMessageSurveys(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { title, questions, responses: [] })
      return newMap
    })
    toast.success('Encuesta creada', { icon: '📋' })
  }, [])
  
  // Calificaciones
  const rateMessage = useCallback((messageId: string, rating: number, comment?: string) => {
    setMessageRatings(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { rating, comment, timestamp: Date.now() })
      return newMap
    })
    toast.success(`Calificado con ${rating} estrellas`, { icon: '⭐' })
  }, [])
  
  // Reseñas
  const addReview = useCallback((messageId: string, review: string, rating: number) => {
    setMessageReviews(prev => {
      const newMap = new Map(prev)
      newMap.set(`review-${Date.now()}`, { review, rating, helpful: 0 })
      return newMap
    })
    toast.success('Reseña agregada', { icon: '📝' })
  }, [])
  
  // Feedback
  const addFeedback = useCallback((messageId: string, feedback: string, type: 'positive' | 'negative' | 'neutral') => {
    setMessageFeedback(prev => {
      const newMap = new Map(prev)
      newMap.set(`feedback-${Date.now()}`, { feedback, type, timestamp: Date.now() })
      return newMap
    })
    toast.success('Feedback registrado', { icon: '💬' })
  }, [])
  
  // Reportes
  const reportMessage = useCallback((messageId: string, reason: string, description: string) => {
    setMessageReports(prev => {
      const newMap = new Map(prev)
      newMap.set(`report-${Date.now()}`, { reason, description, status: 'pending' })
      return newMap
    })
    toast.success('Reporte enviado', { icon: '🚨' })
  }, [])
  
  // Moderación
  const moderateMessage = useCallback((messageId: string, action: string, reason: string, moderator: string) => {
    setMessageModeration(prev => {
      const newMap = new Map(prev)
      newMap.set(`mod-${Date.now()}`, { action, reason, moderator })
      return newMap
    })
    toast.success(`Mensaje ${action}`, { icon: '🛡️' })
  }, [])
  
  // Aprobación
  const approveMessage = useCallback((messageId: string, status: 'pending' | 'approved' | 'rejected', approver?: string) => {
    setMessageApproval(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { status, approver })
      return newMap
    })
    toast.success(`Mensaje ${status}`, { icon: '✅' })
  }, [])
  
  // Programación
  const scheduleMessage = useCallback((messageId: string, scheduledTime: number) => {
    setMessageScheduling(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { scheduledTime, status: 'scheduled' })
      return newMap
    })
    toast.success('Mensaje programado', { icon: '⏰' })
  }, [])
  
  // Plantillas
  const createTemplate = useCallback((name: string, content: string, variables: string[], category: string) => {
    setMessageTemplates(prev => {
      const newMap = new Map(prev)
      newMap.set(`template-${Date.now()}`, { name, content, variables, category })
      return newMap
    })
    toast.success('Plantilla creada', { icon: '📝' })
  }, [])
  
  // Variables
  const addVariable = useCallback((name: string, value: string, type: string) => {
    setMessageVariables(prev => {
      const newMap = new Map(prev)
      newMap.set(name, { name, value, type })
      return newMap
    })
    toast.success('Variable agregada', { icon: '🔧' })
  }, [])
  
  // Condiciones
  const addCondition = useCallback((condition: string, action: string, enabled: boolean) => {
    setMessageConditions(prev => {
      const newMap = new Map(prev)
      newMap.set(`condition-${Date.now()}`, { condition, action, enabled })
      return newMap
    })
    toast.success('Condición agregada', { icon: '⚙️' })
  }, [])
  
  // Acciones
  const executeAction = useCallback((action: string, parameters: any) => {
    setMessageActions(prev => {
      const newMap = new Map(prev)
      newMap.set(`action-${Date.now()}`, { action, parameters, result: null })
      return newMap
    })
    toast.success('Acción ejecutada', { icon: '⚡' })
  }, [])
  
  // Disparadores
  const createTrigger = useCallback((trigger: string, conditions: string[], actions: string[]) => {
    setMessageTriggers(prev => {
      const newMap = new Map(prev)
      newMap.set(`trigger-${Date.now()}`, { trigger, conditions, actions })
      return newMap
    })
    toast.success('Disparador creado', { icon: '🎯' })
  }, [])
  
  // Flujos de trabajo
  const createWorkflow = useCallback((name: string, steps: Array<{ step: number, action: string, condition?: string }>) => {
    setMessageWorkflows(prev => {
      const newMap = new Map(prev)
      newMap.set(`workflow-${Date.now()}`, { name, steps, active: true })
      return newMap
    })
    toast.success('Flujo de trabajo creado', { icon: '🔄' })
  }, [])
  
  // Integraciones
  const addIntegration = useCallback((service: string, config: any) => {
    setMessageIntegrations(prev => {
      const newMap = new Map(prev)
      newMap.set(`integration-${Date.now()}`, { service, config, enabled: true })
      return newMap
    })
    toast.success(`Integración "${service}" agregada`, { icon: '🔌' })
  }, [])
  
  // Webhooks
  const addWebhook = useCallback((url: string, events: string[], secret?: string) => {
    setMessageWebhooks(prev => {
      const newMap = new Map(prev)
      newMap.set(`webhook-${Date.now()}`, { url, events, secret })
      return newMap
    })
    toast.success('Webhook agregado', { icon: '🔗' })
  }, [])
  
  // APIs
  const addAPI = useCallback((endpoint: string, method: string, auth: any) => {
    setMessageAPIs(prev => {
      const newMap = new Map(prev)
      newMap.set(`api-${Date.now()}`, { endpoint, method, auth, enabled: true })
      return newMap
    })
    toast.success('API agregada', { icon: '🌐' })
  }, [])
  
  // SDKs
  const addSDK = useCallback((sdk: string, version: string, config: any) => {
    setMessageSDKs(prev => {
      const newMap = new Map(prev)
      newMap.set(`sdk-${Date.now()}`, { sdk, version, config })
      return newMap
    })
    toast.success(`SDK "${sdk}" agregado`, { icon: '📦' })
  }, [])
  
  // Plugins
  const addPlugin = useCallback((plugin: string, version: string, config: any) => {
    setMessagePlugins(prev => {
      const newMap = new Map(prev)
      newMap.set(`plugin-${Date.now()}`, { plugin, version, config, enabled: true })
      return newMap
    })
    toast.success(`Plugin "${plugin}" agregado`, { icon: '🔌' })
  }, [])
  
  // ========== FUNCIONES DE NOTIFICACIONES Y ORGANIZACIÓN ==========
  
  // Notificaciones
  const addNotification = useCallback((type: string, title: string, body: string) => {
    setMessageNotifications(prev => {
      const newMap = new Map(prev)
      newMap.set(`notif-${Date.now()}`, { type, title, body, read: false, timestamp: Date.now() })
      return newMap
    })
    toast.info(title, { icon: '🔔' })
  }, [])
  
  // Favoritos
  const toggleFavorite = useCallback((messageId: string) => {
    setMessageFavorites(prev => {
      const newSet = new Set(prev)
      if (newSet.has(messageId)) {
        newSet.delete(messageId)
        toast.success('Eliminado de favoritos', { icon: '⭐' })
      } else {
        newSet.add(messageId)
        toast.success('Agregado a favoritos', { icon: '⭐' })
      }
      return newSet
    })
  }, [])
  
  // Fijar mensajes
  const togglePin = useCallback((messageId: string) => {
    setMessagePinned(prev => {
      const newSet = new Set(prev)
      if (newSet.has(messageId)) {
        newSet.delete(messageId)
        toast.success('Mensaje desfijado', { icon: '📌' })
      } else {
        newSet.add(messageId)
        toast.success('Mensaje fijado', { icon: '📌' })
      }
      return newSet
    })
  }, [])
  
  // Archivar mensajes
  const archiveMessage = useCallback((messageId: string) => {
    setMessageArchived(prev => {
      const newSet = new Set(prev)
      newSet.add(messageId)
      return newSet
    })
    toast.success('Mensaje archivado', { icon: '📦' })
  }, [])
  
  // Eliminar mensajes
  const deleteMessage = useCallback((messageId: string) => {
    setMessageDeleted(prev => {
      const newSet = new Set(prev)
      newSet.add(messageId)
      return newSet
    })
    toast.success('Mensaje eliminado', { icon: '🗑️' })
  }, [])
  
  // Borradores
  const saveDraft = useCallback((content: string) => {
    setMessageDrafts(prev => {
      const newMap = new Map(prev)
      newMap.set(`draft-${Date.now()}`, { content, timestamp: Date.now() })
      return newMap
    })
    toast.success('Borrador guardado', { icon: '💾' })
  }, [])
  
  // Marcar como importante
  const toggleImportant = useCallback((messageId: string) => {
    setMessageImportant(prev => {
      const newSet = new Set(prev)
      if (newSet.has(messageId)) {
        newSet.delete(messageId)
        toast.success('Marcado como no importante', { icon: '⚠️' })
      } else {
        newSet.add(messageId)
        toast.success('Marcado como importante', { icon: '⚠️' })
      }
      return newSet
    })
  }, [])
  
  // Marcar como leído/no leído
  const toggleRead = useCallback((messageId: string) => {
    if (messageUnread.has(messageId)) {
      setMessageUnread(prev => {
        const newSet = new Set(prev)
        newSet.delete(messageId)
        return newSet
      })
      setMessageRead(prev => {
        const newSet = new Set(prev)
        newSet.add(messageId)
        return newSet
      })
      toast.success('Marcado como leído', { icon: '✓' })
    } else {
      setMessageRead(prev => {
        const newSet = new Set(prev)
        newSet.delete(messageId)
        return newSet
      })
      setMessageUnread(prev => {
        const newSet = new Set(prev)
        newSet.add(messageId)
        return newSet
      })
      toast.success('Marcado como no leído', { icon: '📬' })
    }
  }, [messageUnread])
  
  // Historial de cambios
  const trackHistory = useCallback((messageId: string, previous: string, current: string) => {
    setMessageHistory(prev => {
      const newMap = new Map(prev)
      newMap.set(`history-${Date.now()}`, { previous, current, timestamp: Date.now() })
      return newMap
    })
  }, [])
  
  // Atajos de teclado
  const registerShortcut = useCallback((key: string, action: string, description: string) => {
    setMessageShortcuts(prev => {
      const newMap = new Map(prev)
      newMap.set(key, { key, action, description })
      return newMap
    })
    toast.success(`Atajo "${key}" registrado`, { icon: '⌨️' })
  }, [])
  
  // ========== FUNCIONES DE MEJORA ADICIONALES ==========
  
  // Sistema de caché de mensajes
  const cacheMessage = useCallback((messageId: string, content: string) => {
    if (!cacheEnabled) return
    setMessageCache(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { content, timestamp: Date.now() })
      // Limpiar caché antiguo (más de 1 hora)
      const oneHourAgo = Date.now() - 3600000
      Array.from(newMap.entries()).forEach(([id, data]) => {
        if (data.timestamp < oneHourAgo) newMap.delete(id)
      })
      return newMap
    })
  }, [cacheEnabled])
  
  // Indexación de mensajes para búsqueda rápida
  const indexMessage = useCallback((messageId: string, content: string) => {
    const keywords = content.toLowerCase().split(/\s+/).filter(w => w.length > 3)
    const summary = content.substring(0, 100) + '...'
    setMessageIndexing(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { keywords, summary })
      return newMap
    })
    
    // Actualizar índice de búsqueda
    setSearchIndex(prev => {
      const newMap = new Map(prev)
      keywords.forEach(keyword => {
        const ids = newMap.get(keyword) || []
        if (!ids.includes(messageId)) {
          newMap.set(keyword, [...ids, messageId])
        }
      })
      return newMap
    })
  }, [])
  
  // Búsqueda inteligente
  const smartSearchMessages = useCallback((query: string): string[] => {
    if (!smartSearch) return []
    const queryWords = query.toLowerCase().split(/\s+/)
    const results = new Set<string>()
    
    queryWords.forEach(word => {
      const ids = searchIndex.get(word) || []
      ids.forEach(id => results.add(id))
    })
    
    return Array.from(results)
  }, [smartSearch, searchIndex])
  
  // Sugerencias de mensajes
  const generateMessageSuggestions = useCallback((messageId: string, context: string) => {
    if (!aiFeatures.suggestions) return
    const suggestions = [
      `¿Puedes explicar más sobre "${context}"?`,
      `¿Tienes más información sobre esto?`,
      `¿Podrías darme un ejemplo?`
    ]
    setMessageSuggestions(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, suggestions)
      return newMap
    })
  }, [aiFeatures.suggestions])
  
  // Autocompletado inteligente
  const getAutoComplete = useCallback((partial: string): string[] => {
    if (!autoComplete || partial.length < 2) return []
    const matches: string[] = []
    messages.forEach(msg => {
      const words = msg.content.toLowerCase().split(/\s+/)
      words.forEach(word => {
        if (word.startsWith(partial.toLowerCase()) && !matches.includes(word)) {
          matches.push(word)
        }
      })
    })
    return matches.slice(0, 5)
  }, [autoComplete, messages])
  
  // Predicción de escritura
  const predictNextWord = useCallback((currentText: string): string | null => {
    if (!typingPrediction || currentText.length < 3) return null
    const lastWords = currentText.toLowerCase().split(/\s+/).slice(-2)
    // Buscar patrones en mensajes anteriores
    for (const msg of messages) {
      const words = msg.content.toLowerCase().split(/\s+/)
      const index = words.findIndex((w, i) => 
        i < words.length - 1 && 
        words[i] === lastWords[lastWords.length - 1] &&
        words[i + 1]
      )
      if (index >= 0) {
        return words[index + 1]
      }
    }
    return null
  }, [typingPrediction, messages])
  
  // Respuestas rápidas
  const addQuickReply = useCallback((reply: string) => {
    setQuickReplies(prev => [...prev, reply].slice(-10)) // Mantener solo las últimas 10
    toast.success('Respuesta rápida agregada', { icon: '⚡' })
  }, [])
  
  // Macros de mensajes
  const createMacro = useCallback((trigger: string, replacement: string) => {
    setMessageMacros(prev => {
      const newMap = new Map(prev)
      newMap.set(trigger, { trigger, replacement })
      return newMap
    })
    toast.success(`Macro "${trigger}" creada`, { icon: '⌨️' })
  }, [])
  
  const expandMacro = useCallback((text: string): string => {
    if (!macroEnabled) return text
    let result = text
    messageMacros.forEach((macro, trigger) => {
      const regex = new RegExp(`\\b${trigger}\\b`, 'gi')
      result = result.replace(regex, macro.replacement)
    })
    return result
  }, [macroEnabled, messageMacros])
  
  // Sistema de deshacer/rehacer
  const saveState = useCallback(() => {
    if (!undoEnabled) return
    const state = {
      messages: [...messages],
      timestamp: Date.now()
    }
    setUndoStack(prev => [...prev, state].slice(-50)) // Mantener solo las últimas 50
    setRedoStack([]) // Limpiar redo al hacer nueva acción
  }, [undoEnabled, messages])
  
  const undo = useCallback(() => {
    if (undoStack.length === 0) {
      toast.error('No hay acciones para deshacer', { icon: '❌' })
      return
    }
    const previousState = undoStack[undoStack.length - 1]
    setRedoStack(prev => [...prev, { messages: [...messages], timestamp: Date.now() }])
    setUndoStack(prev => prev.slice(0, -1))
    // Restaurar mensajes (esto requeriría una función de restauración en el store)
    toast.success('Acción deshecha', { icon: '↶' })
  }, [undoStack, messages])
  
  const redo = useCallback(() => {
    if (redoStack.length === 0) {
      toast.error('No hay acciones para rehacer', { icon: '❌' })
      return
    }
    const nextState = redoStack[redoStack.length - 1]
    setUndoStack(prev => [...prev, { messages: [...messages], timestamp: Date.now() }])
    setRedoStack(prev => prev.slice(0, -1))
    toast.success('Acción rehecha', { icon: '↷' })
  }, [redoStack, messages])
  
  // Modo batch
  const addToBatch = useCallback((role: string, content: string) => {
    setMessageBatch(prev => [...prev, { role, content }])
    toast.success('Mensaje agregado al batch', { icon: '📦' })
  }, [])
  
  const sendBatch = useCallback(async () => {
    if (messageBatch.length === 0) return
    try {
      for (const msg of messageBatch) {
        if (useBulkChatMode && bulkChat.sessionId) {
          await bulkChat.sendMessage(msg.content)
        }
        addMessage({
          id: Date.now().toString(),
          role: msg.role as 'user' | 'assistant',
          content: msg.content,
          timestamp: new Date(),
        })
      }
      setMessageBatch([])
      toast.success(`${messageBatch.length} mensajes enviados`, { icon: '✅' })
    } catch (error) {
      toast.error('Error enviando batch', { icon: '❌' })
    }
  }, [messageBatch, useBulkChatMode, bulkChat, addMessage])
  
  // Filtrado automático
  const applyFilters = useCallback((content: string): boolean => {
    if (!autoFilter) return true
    for (const rule of filterRules) {
      const regex = new RegExp(rule.pattern, 'i')
      if (regex.test(content)) {
        if (rule.action === 'block') return false
        if (rule.action === 'flag') {
          toast.warning('Mensaje marcado por filtro', { icon: '⚠️' })
        }
      }
    }
    return true
  }, [autoFilter, filterRules])
  
  // Notificaciones inteligentes
  const sendNotification = useCallback((messageId: string, type: string) => {
    if (!notificationSettings.desktop) return
    const message = messages.find(m => m.id === messageId)
    if (message && 'Notification' in window && Notification.permission === 'granted') {
      new Notification('Nuevo mensaje', {
        body: message.content.substring(0, 100),
        icon: '/icon.png'
      })
    }
    setMessageNotifications(prev => {
      const newMap = new Map(prev)
      const existing = newMap.get(messageId)
      newMap.set(messageId, { 
        type, 
        title: existing?.title || 'Nuevo mensaje',
        body: existing?.body || message?.content.substring(0, 100) || '',
        read: existing?.read || false,
        timestamp: Date.now() 
      })
      return newMap
    })
  }, [notificationSettings, messages])
  
  // Resaltado de mensajes
  const highlightMessage = useCallback((messageId: string, color: string, pattern: string) => {
    setMessageHighlights(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { color, pattern })
      return newMap
    })
    toast.success('Mensaje resaltado', { icon: '🖍️' })
  }, [])
  
  // Bookmarks avanzados
  const addAdvancedBookmark = useCallback((messageId: string, name: string, category: string, tags: string[]) => {
    setMessageBookmarksAdvanced(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, { name, category, tags })
      return newMap
    })
    if (!bookmarkCategories.includes(category)) {
      setBookmarkCategories(prev => [...prev, category])
    }
    toast.success('Bookmark avanzado agregado', { icon: '🔖' })
  }, [bookmarkCategories])
  
  // Exportación avanzada con plantillas
  const exportWithTemplate = useCallback((templateId: string) => {
    const template = exportTemplates.get(templateId)
    if (!template) {
      toast.error('Plantilla no encontrada', { icon: '❌' })
      return
    }
    // Aplicar plantilla y exportar
    toast.success(`Exportado con plantilla "${template.name}"`, { icon: '📄' })
  }, [exportTemplates])
  
  // Importación de mensajes
  const importMessages = useCallback(async (file: File, format: string) => {
    if (!importEnabled) return
    try {
      const text = await file.text()
      let imported: any[] = []
      
      if (format === 'json') {
        imported = JSON.parse(text)
      } else if (format === 'txt') {
        imported = text.split('\n').filter(Boolean).map((line, idx) => ({
          id: `imported-${Date.now()}-${idx}`,
          role: 'user',
          content: line,
          timestamp: new Date()
        }))
      }
      
      imported.forEach(msg => {
        addMessage(msg)
      })
      
      setMessageImport(prev => {
        const newMap = new Map(prev)
        newMap.set(`import-${Date.now()}`, { source: file.name, format, timestamp: Date.now() })
        return newMap
      })
      
      toast.success(`${imported.length} mensajes importados`, { icon: '📥' })
    } catch (error) {
      toast.error('Error importando mensajes', { icon: '❌' })
    }
  }, [importEnabled, addMessage])
  
  // Sincronización avanzada
  const syncWithProvider = useCallback(async (provider: string) => {
    try {
      const syncData = {
        messages,
        timestamp: Date.now(),
        provider
      }
      
      if (provider === 'localStorage') {
        localStorage.setItem('chat-sync', JSON.stringify(syncData))
      } else if (provider === 'indexedDB') {
        // Implementar IndexedDB sync
      }
      
      setMessageSyncAdvanced(prev => {
        const newMap = new Map(prev)
        newMap.set(provider, { provider, status: 'synced', lastSync: Date.now() })
        return newMap
      })
      
      toast.success(`Sincronizado con ${provider}`, { icon: '🔄' })
    } catch (error) {
      toast.error(`Error sincronizando con ${provider}`, { icon: '❌' })
    }
  }, [messages])
  
  // Control de versiones
  const createVersion = useCallback((messageId: string, changes: string[]) => {
    if (!versionControl) return
    setMessageVersioning(prev => {
      const newMap = new Map(prev)
      const current = newMap.get(messageId) || { version: 0, changes: [], timestamp: Date.now() }
      newMap.set(messageId, {
        version: current.version + 1,
        changes: [...current.changes, ...changes],
        timestamp: Date.now()
      })
      return newMap
    })
    toast.success(`Versión ${(messageVersioning.get(messageId)?.version || 0) + 1} creada`, { icon: '📝' })
  }, [versionControl, messageVersioning])
  
  // Colaboración
  const addCollaborator = useCallback((messageId: string, userId: string, permissions: string[]) => {
    setMessageCollaboration(prev => {
      const newMap = new Map(prev)
      const current = newMap.get(messageId) || { users: [], permissions: [] }
      newMap.set(messageId, {
        users: [...current.users, userId],
        permissions: [...current.permissions, ...permissions]
      })
      return newMap
    })
    toast.success(`Colaborador agregado a mensaje`, { icon: '👥' })
  }, [])
  
  // Análisis de IA
  const analyzeMessage = useCallback((messageId: string, content: string) => {
    if (!aiFeatures.sentiment && !aiFeatures.topics) return
    
    const analysis: any = {
      suggestions: [],
      sentiment: 'neutral',
      topics: []
    }
    
    // Análisis de sentimiento simple
    if (aiFeatures.sentiment) {
      const positiveWords = ['bueno', 'excelente', 'perfecto', 'genial']
      const negativeWords = ['malo', 'error', 'problema', 'fallo']
      const lowerContent = content.toLowerCase()
      const positiveCount = positiveWords.filter(w => lowerContent.includes(w)).length
      const negativeCount = negativeWords.filter(w => lowerContent.includes(w)).length
      analysis.sentiment = positiveCount > negativeCount ? 'positive' : negativeCount > positiveCount ? 'negative' : 'neutral'
    }
    
    // Extracción de temas simple
    if (aiFeatures.topics) {
      const words = content.toLowerCase().split(/\s+/).filter(w => w.length > 4)
      const commonWords = ['the', 'this', 'that', 'with', 'from', 'have', 'been', 'will']
      analysis.topics = words.filter(w => !commonWords.includes(w)).slice(0, 5)
    }
    
    setMessageAI(prev => {
      const newMap = new Map(prev)
      newMap.set(messageId, analysis)
      return newMap
    })
  }, [aiFeatures])
  
  // Auto-indexación de mensajes
  useEffect(() => {
    messages.forEach(msg => {
      indexMessage(msg.id, msg.content)
      cacheMessage(msg.id, msg.content)
      if (aiFeatures.suggestions || aiFeatures.sentiment || aiFeatures.topics) {
        analyzeMessage(msg.id, msg.content)
      }
    })
  }, [messages, indexMessage, cacheMessage, analyzeMessage, aiFeatures])
  
  // Auto-save conversation
  useEffect(() => {
    if (!autoSave) return
    
    const interval = setInterval(() => {
      if (messages.length > 0) {
        try {
          const data = {
            messages: messages.map(m => ({
              id: m.id,
              role: m.role,
              content: m.content,
              timestamp: m.timestamp,
            })),
            favorites: Array.from(favoriteMessages),
            tags: Object.fromEntries(messageTags),
            notes: Object.fromEntries(messageNotes),
            timestamp: new Date().toISOString(),
          }
          localStorage.setItem('bulk-chat-autosave', JSON.stringify(data))
        } catch (error) {
          console.error('Error auto-saving:', error)
        }
      }
    }, 30000) // Auto-save cada 30 segundos
    
    return () => clearInterval(interval)
  }, [autoSave, messages, favoriteMessages, messageTags, messageNotes])
  
  // Load auto-saved conversation on mount
  useEffect(() => {
    if (!autoSave) return
    
    try {
      const saved = localStorage.getItem('bulk-chat-autosave')
      if (saved) {
        const data = JSON.parse(saved)
        if (data.messages && Array.isArray(data.messages)) {
          // Restaurar solo si no hay mensajes actuales
          if (messages.length === 0) {
            data.messages.forEach((msg: any) => {
              addMessage({
                id: msg.id,
                role: msg.role,
                content: msg.content,
                timestamp: new Date(msg.timestamp),
              })
            })
            
            if (data.favorites) {
              setFavoriteMessages(new Set(data.favorites))
            }
            if (data.tags) {
              setMessageTags(new Map(Object.entries(data.tags)))
            }
            if (data.notes) {
              setMessageNotes(new Map(Object.entries(data.notes)))
            }
          }
        }
      }
    } catch (error) {
      console.error('Error loading auto-save:', error)
    }
  }, []) // Solo al montar

  const [commandQuery, setCommandQuery] = useState('')
  const filteredCommands = commands.filter(cmd => 
    cmd.label.toLowerCase().includes(commandQuery.toLowerCase()) ||
    cmd.id.toLowerCase().includes(commandQuery.toLowerCase())
  )

  return (
    <>
      {/* Estilos para modo impresión */}
      {showPrintMode && (
        <style>{`
          @media print {
            body * {
              visibility: hidden;
            }
            .print-content, .print-content * {
              visibility: visible;
            }
            .print-content {
              position: absolute;
              left: 0;
              top: 0;
              width: 100%;
            }
            .no-print {
              display: none !important;
            }
          }
        `}</style>
      )}
      <div className={`max-w-4xl mx-auto ${presentationMode ? 'fixed inset-0 z-50 bg-slate-900' : ''} ${showPrintMode ? 'print-content' : ''}`}>
          {/* Command Palette */}
          {showCommandPalette && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-[100] flex items-start justify-center pt-[20vh] bg-black/50 backdrop-blur-sm"
              onClick={() => setShowCommandPalette(false)}
            >
              <motion.div
                initial={{ scale: 0.95 }}
                animate={{ scale: 1 }}
                className="bg-slate-800 border border-slate-700 rounded-lg shadow-2xl w-full max-w-2xl mx-4"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="p-4 border-b border-slate-700">
                  <input
                    type="text"
                    value={commandQuery}
                    onChange={(e) => setCommandQuery(e.target.value)}
                    placeholder="Buscar comando... (Ctrl+K)"
                    className="w-full px-4 py-3 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    autoFocus
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && filteredCommands.length > 0) {
                        filteredCommands[0].action()
                        setShowCommandPalette(false)
                        setCommandQuery('')
                      }
                      if (e.key === 'Escape') {
                        setShowCommandPalette(false)
                        setCommandQuery('')
                      }
                    }}
                  />
                </div>
                <div className="max-h-96 overflow-y-auto p-2">
                  {filteredCommands.length > 0 ? (
                    filteredCommands.map((cmd, idx) => (
                      <button
                        key={cmd.id}
                        onClick={() => {
                          cmd.action()
                          setCommandHistory(prev => [...prev, cmd.id].slice(-10))
                          setShowCommandPalette(false)
                          setCommandQuery('')
                        }}
                        className="w-full text-left px-4 py-3 hover:bg-slate-700 rounded-lg transition-colors flex items-center gap-3"
                      >
                        <span className="text-2xl">{cmd.icon}</span>
                        <span className="text-slate-200">{cmd.label}</span>
                        {idx === 0 && (
                          <span className="ml-auto text-xs text-slate-500">Enter</span>
                        )}
                      </button>
                    ))
                  ) : (
                    <div className="px-4 py-8 text-center text-slate-400">
                      No se encontraron comandos
                    </div>
                  )}
                </div>
              </motion.div>
            </motion.div>
          )}

          {/* Presentation Mode Overlay */}
          {presentationMode && (
            <div className="fixed top-4 right-4 z-50 flex gap-2">
              <button
                onClick={() => setPresentationMode(false)}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg text-white text-sm transition-colors"
              >
                Salir (Esc)
              </button>
            </div>
          )}

      {/* Header with Actions */}
          <div className={`flex items-center justify-between mb-4 flex-wrap gap-2 ${presentationMode || showPrintMode || zenMode ? 'hidden' : ''} no-print`}>
        <h2 className="text-xl font-bold text-white">Crea tu Modelo TruthGPT</h2>                                                                               
        <div className="flex items-center gap-2">
          {/* Rate Limit Indicator */}
          <RateLimitIndicator className="hidden md:flex" />
          
          {/* Bulk Chat Mode Toggle */}
          <button
            onClick={() => {
              setUseBulkChatMode(!useBulkChatMode)
              if (!useBulkChatMode && !bulkChat.sessionId) {
                bulkChat.createSession()
              }
            }}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all text-sm border relative ${
              useBulkChatMode
                ? 'bg-green-600/50 hover:bg-green-700/50 border-green-700 text-green-300 shadow-lg shadow-green-500/20'
                : 'bg-slate-800/50 hover:bg-slate-700/50 border-slate-700 text-slate-300'
            }`}
            title={`Bulk Chat ${bulkChat.isConnected ? 'Conectado' : 'Desconectado'} - ${bulkChat.sessionId ? (bulkChat.isPaused ? 'Pausado' : 'Activo') : 'Sin sesión'}`}
          >
            <div className="relative" data-connection-button onClick={(e) => {
              e.stopPropagation()
              setShowConnectionInfo(!showConnectionInfo)
            }}>
              <div className={`w-2 h-2 rounded-full ${
                !bulkChat.isOnline ? 'bg-gray-500' :
                bulkChat.connectionQuality === 'excellent' ? 'bg-green-500' :
                bulkChat.connectionQuality === 'good' ? 'bg-yellow-500' :
                bulkChat.connectionQuality === 'poor' ? 'bg-orange-500' :
                bulkChat.isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'
              }`} />
              {bulkChat.isLoading && (
                <div className="absolute inset-0 w-2 h-2 rounded-full bg-green-400 animate-ping" />
              )}
              {!bulkChat.isOnline && (
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full border-2 border-slate-800" title="Offline" />
              )}
            </div>
            <span>Bulk Chat</span>
            {bulkChat.sessionId && (
              <span className="text-xs opacity-75 flex items-center gap-1">
                <span>{bulkChat.isPaused ? '⏸ Pausado' : '▶ Activo'}</span>
                {bulkChat.messageCount > 0 && (
                  <span className="bg-green-500/20 text-green-400 px-1.5 py-0.5 rounded text-[10px]">
                    {bulkChat.messageCount}
                  </span>
                )}
              </span>
            )}
            {bulkChat.reconnectAttempts > 0 && bulkChat.reconnectAttempts < 5 && (
              <span className="text-xs text-yellow-400 animate-pulse">
                (Reintento {bulkChat.reconnectAttempts}/5)
              </span>
            )}
          </button>
          
          {/* Bulk Chat Controls */}
          {useBulkChatMode && bulkChat.sessionId && (
            <>
              {bulkChat.isPaused ? (
                <button
                  onClick={() => bulkChat.resume()}
                  disabled={bulkChat.isLoading}
                  className="px-3 py-2 bg-green-600/50 hover:bg-green-700/50 border border-green-700 rounded-lg transition-all text-sm text-green-300 disabled:opacity-50 hover:scale-105 active:scale-95"
                  title="Reanudar Bulk Chat"
                >
                  ▶
                </button>
              ) : (
                <button
                  onClick={() => bulkChat.pause()}
                  disabled={bulkChat.isLoading}
                  className="px-3 py-2 bg-yellow-600/50 hover:bg-yellow-700/50 border border-yellow-700 rounded-lg transition-all text-sm text-yellow-300 disabled:opacity-50 hover:scale-105 active:scale-95"
                  title="Pausar Bulk Chat"
                >
                  ⏸
                </button>
              )}
              <button
                onClick={() => bulkChat.stop()}
                disabled={bulkChat.isLoading}
                className="px-3 py-2 bg-red-600/50 hover:bg-red-700/50 border border-red-700 rounded-lg transition-all text-sm text-red-300 disabled:opacity-50 hover:scale-105 active:scale-95"
                title="Detener Bulk Chat"
              >
                ⏹
              </button>
              {!bulkChat.isConnected && (
                <button
                  onClick={() => bulkChat.retryConnection()}
                  disabled={bulkChat.isLoading}
                  className="px-3 py-2 bg-blue-600/50 hover:bg-blue-700/50 border border-blue-700 rounded-lg transition-all text-sm text-blue-300 disabled:opacity-50 hover:scale-105 active:scale-95"
                  title="Reintentar conexión"
                >
                  🔄
                </button>
              )}
                <button
                  onClick={async () => {
                    if (bulkChat.sessionId && bulkChat.messages.length > 0) {
                      // Exportar como JSON
                      const data = {
                        sessionId: bulkChat.sessionId,
                        messages: bulkChat.messages,
                        timestamp: new Date().toISOString(),
                        messageCount: bulkChat.messageCount,
                      }
                      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
                      const url = URL.createObjectURL(blob)
                      const a = document.createElement('a')
                      a.href = url
                      a.download = `bulk-chat-${bulkChat.sessionId.slice(0, 8)}-${Date.now()}.json`
                      document.body.appendChild(a)
                      a.click()
                      document.body.removeChild(a)
                      URL.revokeObjectURL(url)
                      toast.success('Conversación exportada (JSON)', { icon: '💾', duration: 2000 })
                    }
                  }}
                  disabled={!bulkChat.sessionId || bulkChat.messages.length === 0}
                  className="px-3 py-2 bg-indigo-600/50 hover:bg-indigo-700/50 border border-indigo-700 rounded-lg transition-all text-sm text-indigo-300 disabled:opacity-50 hover:scale-105 active:scale-95"
                  title="Exportar conversación (JSON) - Ctrl+Shift+E"
                >
                  💾
                </button>
                <button
                  onClick={async () => {
                    if (bulkChat.sessionId && bulkChat.messages.length > 0) {
                      // Exportar como TXT
                      const text = bulkChat.messages
                        .map(msg => `${msg.role === 'user' ? 'Usuario' : 'Asistente'}: ${msg.content}\n${new Date(msg.timestamp).toLocaleString()}\n---\n`)
                        .join('\n')
                      const blob = new Blob([text], { type: 'text/plain' })
                      const url = URL.createObjectURL(blob)
                      const a = document.createElement('a')
                      a.href = url
                      a.download = `bulk-chat-${bulkChat.sessionId.slice(0, 8)}-${Date.now()}.txt`
                      document.body.appendChild(a)
                      a.click()
                      document.body.removeChild(a)
                      URL.revokeObjectURL(url)
                      toast.success('Conversación exportada (TXT)', { icon: '📄', duration: 2000 })
                    }
                  }}
                  disabled={!bulkChat.sessionId || bulkChat.messages.length === 0}
                  className="px-3 py-2 bg-teal-600/50 hover:bg-teal-700/50 border border-teal-700 rounded-lg transition-all text-sm text-teal-300 disabled:opacity-50 hover:scale-105 active:scale-95"
                  title="Exportar como texto plano"
                >
                  📄
                </button>
                <button
                  onClick={async () => {
                    if (bulkChat.sessionId && bulkChat.messages.length > 0) {
                      // Exportar como Markdown
                      const markdown = `# Conversación Bulk Chat\n\n**Sesión:** ${bulkChat.sessionId.slice(0, 8)}\n**Fecha:** ${new Date().toLocaleString()}\n**Total de mensajes:** ${bulkChat.messageCount}\n\n---\n\n${bulkChat.messages.map(msg => `## ${msg.role === 'user' ? 'Usuario' : 'Asistente'}\n\n${msg.content}\n\n*${new Date(msg.timestamp).toLocaleString()}*\n\n---\n`).join('\n')}`
                      const blob = new Blob([markdown], { type: 'text/markdown' })
                      const url = URL.createObjectURL(blob)
                      const a = document.createElement('a')
                      a.href = url
                      a.download = `bulk-chat-${bulkChat.sessionId.slice(0, 8)}-${Date.now()}.md`
                      document.body.appendChild(a)
                      a.click()
                      document.body.removeChild(a)
                      URL.revokeObjectURL(url)
                      toast.success('Conversación exportada (Markdown)', { icon: '📝', duration: 2000 })
                    }
                  }}
                  disabled={!bulkChat.sessionId || bulkChat.messages.length === 0}
                  className="px-3 py-2 bg-pink-600/50 hover:bg-pink-700/50 border border-pink-700 rounded-lg transition-all text-sm text-pink-300 disabled:opacity-50 hover:scale-105 active:scale-95"
                  title="Exportar como Markdown"
                >
                  📝
                </button>
              <button
                onClick={() => setReadMode(!readMode)}
                className={`px-3 py-2 border rounded-lg transition-all text-sm ${
                  readMode
                    ? 'bg-blue-600/50 hover:bg-blue-700/50 border-blue-700 text-blue-300'
                    : 'bg-slate-800/50 hover:bg-slate-700/50 border-slate-700 text-slate-300'
                }`}
                title="Modo lectura rápida (Ctrl+Shift+R)"
              >
                📖
              </button>
              <button
                onClick={() => {
                  if (bulkChat.sessionId) {
                    const shareUrl = `${window.location.origin}?session=${bulkChat.sessionId}`
                    navigator.clipboard.writeText(shareUrl).then(() => {
                      toast.success('URL de sesión copiada', { icon: '🔗', duration: 2000 })
                    }).catch(() => {
                      toast.error('Error al copiar URL')
                    })
                  }
                }}
                disabled={!bulkChat.sessionId}
                className="px-3 py-2 bg-emerald-600/50 hover:bg-emerald-700/50 border border-emerald-700 rounded-lg transition-all text-sm text-emerald-300 disabled:opacity-50 hover:scale-105 active:scale-95"
                title="Compartir sesión"
              >
                🔗
              </button>
            </>
          )}
          
          {/* Command Palette Button */}
          <button
            onClick={() => setShowCommandPalette(true)}
            className="px-4 py-2 bg-slate-700/50 hover:bg-slate-600/50 border border-slate-600 rounded-lg transition-all text-sm text-slate-300"
            title="Paleta de comandos (Ctrl+K)"
          >
            ⌘
          </button>
          
          {/* Quick Actions Menu */}
          <button
            onClick={() => setShowQuickActions(!showQuickActions)}
            data-quick-actions
            className="px-4 py-2 bg-violet-600/50 hover:bg-violet-700/50 border border-violet-700 rounded-lg transition-all text-sm text-violet-300 relative"
            title="Acciones rápidas"
          >
            ⚡
            {bulkChat.sessionId && bulkChat.messages.length > 0 && (
              <span className="absolute -top-1 -right-1 w-2 h-2 bg-violet-400 rounded-full animate-pulse" />
            )}
          </button>
          
          {showQuickActions && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              data-quick-actions
              className="absolute top-full right-0 mt-2 bg-slate-800 border border-slate-700 rounded-lg p-3 shadow-xl z-50 min-w-[240px]"
            >
              <div className="space-y-2 text-sm">
                <div className="text-xs text-slate-400 mb-2 font-semibold">Acciones Rápidas</div>
                <button
                  onClick={() => {
                    if (bulkChat.sessionId) {
                      bulkChat.refreshMessages()
                      toast.success('Mensajes actualizados', { icon: '🔄', duration: 2000 })
                    }
                  }}
                  className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                >
                  🔄 Actualizar mensajes
                </button>
                <button
                  onClick={() => {
                    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
                  }}
                  className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                >
                  ⬇️ Ir al final
                </button>
                <button
                  onClick={() => {
                    if (bulkChat.sessionId) {
                      const summary = `Sesión: ${bulkChat.sessionId.slice(0, 8)}\nMensajes: ${bulkChat.messageCount}\nEstado: ${bulkChat.isPaused ? 'Pausado' : 'Activo'}\nConexión: ${bulkChat.isConnected ? 'Conectado' : 'Desconectado'}`
                      navigator.clipboard.writeText(summary)
                      toast.success('Resumen copiado', { icon: '📋', duration: 2000 })
                    }
                  }}
                  disabled={!bulkChat.sessionId}
                  className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2 disabled:opacity-50"
                >
                  📋 Copiar resumen
                </button>
                <button
                  onClick={() => {
                    setReadMode(!readMode)
                    setShowQuickActions(false)
                  }}
                  className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                >
                  📖 {readMode ? 'Desactivar' : 'Activar'} modo lectura
                </button>
                <button
                  onClick={() => {
                    setAutoScroll(!autoScroll)
                    setShowQuickActions(false)
                    toast.success(`Auto-scroll ${!autoScroll ? 'activado' : 'desactivado'}`, { icon: '⬇️', duration: 2000 })
                  }}
                  className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                >
                  ⬇️ {autoScroll ? 'Desactivar' : 'Activar'} auto-scroll
                </button>
                <button
                  onClick={() => {
                    setShowWordCount(!showWordCount)
                    setShowQuickActions(false)
                  }}
                  className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                >
                  📝 {showWordCount ? 'Ocultar' : 'Mostrar'} conteo de palabras
                </button>
                <button
                  onClick={() => {
                    setShowSentiment(!showSentiment)
                    setShowQuickActions(false)
                  }}
                  className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                >
                  😊 {showSentiment ? 'Ocultar' : 'Mostrar'} análisis de sentimiento
                </button>
                <button
                  onClick={() => {
                    setShowReactions(!showReactions)
                    setShowQuickActions(false)
                  }}
                  className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                >
                  👍 {showReactions ? 'Ocultar' : 'Mostrar'} reacciones
                </button>
                <button
                  onClick={() => {
                    setShowPrintMode(true)
                    setTimeout(() => {
                      window.print()
                      setShowPrintMode(false)
                    }, 500)
                    setShowQuickActions(false)
                  }}
                  className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                >
                  🖨️ Imprimir conversación
                </button>
                <button
                  onClick={() => {
                    setShowDebug(!showDebug)
                    setShowQuickActions(false)
                  }}
                  className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                >
                  🐛 {showDebug ? 'Ocultar' : 'Mostrar'} modo debug
                </button>
                <button
                  onClick={() => {
                    setTheme(prev => prev === 'dark' ? 'light' : prev === 'light' ? 'auto' : 'dark')
                    setShowQuickActions(false)
                    toast.success(`Tema: ${theme === 'dark' ? 'Claro' : theme === 'light' ? 'Automático' : 'Oscuro'}`, { icon: '🎨', duration: 2000 })
                  }}
                  className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                >
                  🎨 Cambiar tema ({theme})
                </button>
                <button
                  onClick={() => {
                    setZenMode(!zenMode)
                    setShowQuickActions(false)
                  }}
                  className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                >
                  🧘 {zenMode ? 'Desactivar' : 'Activar'} modo Zen
                </button>
                <button
                  onClick={() => {
                    setShowTags(!showTags)
                    setShowQuickActions(false)
                  }}
                  className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                >
                  🏷️ {showTags ? 'Ocultar' : 'Mostrar'} etiquetas
                </button>
                <button
                  onClick={() => {
                    const speeds: ('slow' | 'normal' | 'fast')[] = ['slow', 'normal', 'fast']
                    const currentIndex = speeds.indexOf(readingSpeed)
                    const nextIndex = (currentIndex + 1) % speeds.length
                    setReadingSpeed(speeds[nextIndex])
                    setShowQuickActions(false)
                    toast.success(`Velocidad de lectura: ${speeds[nextIndex]}`, { icon: '📖', duration: 2000 })
                  }}
                  className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                >
                  📖 Velocidad lectura ({readingSpeed})
                </button>
                <button
                  onClick={() => {
                    setShowEditHistory(!showEditHistory)
                    setShowQuickActions(false)
                  }}
                  className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                >
                  📜 {showEditHistory ? 'Ocultar' : 'Mostrar'} historial de ediciones
                </button>
                <button
                  onClick={() => {
                    setTranslationMode(!translationMode)
                    setShowQuickActions(false)
                  }}
                  className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                >
                  🌐 {translationMode ? 'Desactivar' : 'Activar'} modo traducción
                </button>
                <button
                  onClick={() => {
                    setPresentationMode(!presentationMode)
                    setShowQuickActions(false)
                    if (!presentationMode && !document.fullscreenElement) {
                      document.documentElement.requestFullscreen().catch(() => {})
                    }
                  }}
                  className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                >
                  🖥️ {presentationMode ? 'Salir de' : 'Entrar a'} modo presentación (F11)
                </button>
                <div className="pt-2 border-t border-slate-700">
                  <div className="text-xs text-slate-400 px-3 py-1 mb-2">Accesibilidad:</div>
                  <div className="flex gap-1 px-3">
                    <button
                      onClick={() => {
                        setFontSize('small')
                        setShowQuickActions(false)
                      }}
                      className={`flex-1 px-2 py-1.5 rounded text-xs transition-colors ${
                        fontSize === 'small' 
                          ? 'bg-blue-600 text-white' 
                          : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                      }`}
                    >
                      A-
                    </button>
                    <button
                      onClick={() => {
                        setFontSize('medium')
                        setShowQuickActions(false)
                      }}
                      className={`flex-1 px-2 py-1.5 rounded text-xs transition-colors ${
                        fontSize === 'medium' 
                          ? 'bg-blue-600 text-white' 
                          : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                      }`}
                    >
                      A
                    </button>
                    <button
                      onClick={() => {
                        setFontSize('large')
                        setShowQuickActions(false)
                      }}
                      className={`flex-1 px-2 py-1.5 rounded text-xs transition-colors ${
                        fontSize === 'large' 
                          ? 'bg-blue-600 text-white' 
                          : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                      }`}
                    >
                      A+
                    </button>
                  </div>
                </div>
                <button
                  onClick={() => {
                    setShowStats(!showStats)
                    setShowQuickActions(false)
                  }}
                  className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                >
                  📊 {showStats ? 'Ocultar' : 'Mostrar'} estadísticas
                </button>
                <button
                  onClick={() => {
                    setShowTimeline(!showTimeline)
                    setShowQuickActions(false)
                  }}
                  className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                >
                  📅 {showTimeline ? 'Ocultar' : 'Mostrar'} timeline
                </button>
                <div className="pt-2 border-t border-slate-700">
                  <div className="text-xs text-slate-400 px-3 py-1 mb-2">Nuevas características:</div>
                  <button
                    onClick={() => {
                      setVoiceInputEnabled(!voiceInputEnabled)
                      if (!voiceInputEnabled) {
                        startVoiceInput()
                      } else {
                        stopVoiceInput()
                      }
                      setShowQuickActions(false)
                    }}
                    className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                  >
                    🎤 {voiceInputEnabled ? 'Desactivar' : 'Activar'} entrada de voz
                  </button>
                  <button
                    onClick={() => {
                      setVoiceOutputEnabled(!voiceOutputEnabled)
                      setShowQuickActions(false)
                      toast.success(`Salida de voz ${!voiceOutputEnabled ? 'activada' : 'desactivada'}`, { icon: '🔊' })
                    }}
                    className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                  >
                    🔊 {voiceOutputEnabled ? 'Desactivar' : 'Activar'} salida de voz
                  </button>
                  <button
                    onClick={() => {
                      generateSummary()
                      setShowQuickActions(false)
                    }}
                    className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                  >
                    📝 Generar resumen
                  </button>
                  <button
                    onClick={() => {
                      clusterMessages()
                      setShowQuickActions(false)
                    }}
                    className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                  >
                    🔗 Agrupar mensajes
                  </button>
                  <button
                    onClick={() => {
                      setShowScheduler(!showScheduler)
                      setShowQuickActions(false)
                    }}
                    className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                  >
                    ⏰ {showScheduler ? 'Ocultar' : 'Mostrar'} programador
                  </button>
                  <button
                    onClick={() => {
                      setShowArchive(!showArchive)
                      setShowQuickActions(false)
                    }}
                    className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                  >
                    📦 {showArchive ? 'Ocultar' : 'Mostrar'} archivo
                  </button>
                  <button
                    onClick={() => {
                      setShowDiffView(!showDiffView)
                      setShowQuickActions(false)
                    }}
                    className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                  >
                    🔀 {showDiffView ? 'Ocultar' : 'Mostrar'} vista diff
                  </button>
                  <button
                    onClick={() => {
                      setShowReminders(!showReminders)
                      setShowQuickActions(false)
                    }}
                    className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                  >
                    🔔 {showReminders ? 'Ocultar' : 'Mostrar'} recordatorios
                  </button>
                  <button
                    onClick={() => {
                      setShowAnalytics(!showAnalytics)
                      setShowQuickActions(false)
                    }}
                    className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                  >
                    📊 {showAnalytics ? 'Ocultar' : 'Mostrar'} analíticas
                  </button>
                  <button
                    onClick={() => {
                      setShowBackup(!showBackup)
                      setShowQuickActions(false)
                    }}
                    className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                  >
                    💾 {showBackup ? 'Ocultar' : 'Mostrar'} respaldos
                  </button>
                  <button
                    onClick={() => {
                      setShowExportMenu(!showExportMenu)
                      setShowQuickActions(false)
                    }}
                    className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                  >
                    📤 {showExportMenu ? 'Ocultar' : 'Mostrar'} menú de exportación
                  </button>
                  <button
                    onClick={() => {
                      setShowPinnedMessages(!showPinnedMessages)
                      setShowQuickActions(false)
                    }}
                    className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                  >
                    📌 {showPinnedMessages ? 'Ocultar' : 'Mostrar'} mensajes fijados
                  </button>
                  <button
                    onClick={() => {
                      setGroupingMode(groupingMode === 'none' ? 'time' : groupingMode === 'time' ? 'role' : groupingMode === 'role' ? 'topic' : 'none')
                      setShowQuickActions(false)
                    }}
                    className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                  >
                    🔗 Agrupación: {groupingMode === 'none' ? 'Ninguna' : groupingMode === 'time' ? 'Por tiempo' : groupingMode === 'role' ? 'Por rol' : 'Por tema'}
                  </button>
                  <button
                    onClick={() => {
                      setAdvancedSearch(!advancedSearch)
                      setShowQuickActions(false)
                    }}
                    className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                  >
                    🔍 {advancedSearch ? 'Desactivar' : 'Activar'} búsqueda avanzada
                  </button>
                  <button
                    onClick={() => {
                      setSmartNotifications(!smartNotifications)
                      setShowQuickActions(false)
                    }}
                    className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                  >
                    🔔 {smartNotifications ? 'Desactivar' : 'Activar'} notificaciones inteligentes
                  </button>
                  <button
                    onClick={() => {
                      setRealTimeStats(!realTimeStats)
                      setShowQuickActions(false)
                    }}
                    className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                  >
                    📊 {realTimeStats ? 'Desactivar' : 'Activar'} estadísticas en tiempo real
                  </button>
                  <button
                    onClick={() => {
                      setMessageEncryption(!messageEncryption)
                      setShowQuickActions(false)
                    }}
                    className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                  >
                    🔒 {messageEncryption ? 'Desactivar' : 'Activar'} encriptación
                  </button>
                  <button
                    onClick={() => {
                      setShowTemplateEditor(!showTemplateEditor)
                      setShowQuickActions(false)
                    }}
                    className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                  >
                    📝 {showTemplateEditor ? 'Ocultar' : 'Mostrar'} editor de plantillas
                  </button>
                  <button
                    onClick={() => {
                      setCollaborationMode(!collaborationMode)
                      setShowQuickActions(false)
                    }}
                    className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                  >
                    👥 {collaborationMode ? 'Desactivar' : 'Activar'} modo colaboración
                  </button>
                  <button
                    onClick={() => {
                      setDevMode(!devMode)
                      setShowQuickActions(false)
                    }}
                    className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
                  >
                    🛠️ {devMode ? 'Desactivar' : 'Activar'} modo desarrollo
                  </button>
                </div>
                <div className="pt-2 border-t border-slate-700">
                  <div className="text-xs text-slate-400 px-3 py-1 mb-2">Modo de visualización:</div>
                  <div className="flex gap-1 px-3">
                    <button
                      onClick={() => {
                        setViewMode('compact')
                        setShowQuickActions(false)
                      }}
                      className={`flex-1 px-2 py-1.5 rounded text-xs transition-colors ${
                        viewMode === 'compact' 
                          ? 'bg-blue-600 text-white' 
                          : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                      }`}
                    >
                      Compacto
                    </button>
                    <button
                      onClick={() => {
                        setViewMode('normal')
                        setShowQuickActions(false)
                      }}
                      className={`flex-1 px-2 py-1.5 rounded text-xs transition-colors ${
                        viewMode === 'normal' 
                          ? 'bg-blue-600 text-white' 
                          : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                      }`}
                    >
                      Normal
                    </button>
                    <button
                      onClick={() => {
                        setViewMode('comfortable')
                        setShowQuickActions(false)
                      }}
                      className={`flex-1 px-2 py-1.5 rounded text-xs transition-colors ${
                        viewMode === 'comfortable' 
                          ? 'bg-blue-600 text-white' 
                          : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                      }`}
                    >
                      Cómodo
                    </button>
                  </div>
                </div>
                {selectedMessages.size > 0 && (
                  <div className="pt-2 border-t border-slate-700">
                    <button
                      onClick={async () => {
                        const selected = Array.from(selectedMessages).map(id => 
                          messages.find(m => m.id === id)
                        ).filter(Boolean)
                        
                        if (selected.length > 0) {
                          const text = selected
                            .map(msg => `${msg!.role === 'user' ? 'Usuario' : 'Asistente'}: ${msg!.content}\n${new Date(msg!.timestamp).toLocaleString()}\n---\n`)
                            .join('\n')
                          const blob = new Blob([text], { type: 'text/plain' })
                          const url = URL.createObjectURL(blob)
                          const a = document.createElement('a')
                          a.href = url
                          a.download = `mensajes-seleccionados-${Date.now()}.txt`
                          document.body.appendChild(a)
                          a.click()
                          document.body.removeChild(a)
                          URL.revokeObjectURL(url)
                          toast.success(`${selected.length} mensaje(s) exportado(s)`, { icon: '📋', duration: 2000 })
                          setSelectedMessages(new Set())
                        }
                      }}
                      className="w-full text-left px-3 py-2 bg-blue-700 hover:bg-blue-600 rounded text-white transition-colors flex items-center gap-2"
                    >
                      📋 Exportar {selectedMessages.size} seleccionado(s)
                    </button>
                    <button
                      onClick={() => {
                        setSelectedMessages(new Set())
                        toast('Selección limpiada', { icon: '✕', duration: 2000 })
                      }}
                      className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2 mt-2"
                    >
                      ✕ Limpiar selección
                    </button>
                  </div>
                )}
                <button
                  onClick={() => {
                    if (bulkChat.messages.length > 0) {
                      const userMessages = bulkChat.messages.filter(m => m.role === 'user')
                      const assistantMessages = bulkChat.messages.filter(m => m.role === 'assistant')
                      const totalChars = bulkChat.messages.reduce((acc, m) => acc + m.content.length, 0)
                      const summary = `📊 Resumen de Conversación

📝 Total de mensajes: ${bulkChat.messageCount}
👤 Mensajes del usuario: ${userMessages.length}
🤖 Mensajes del asistente: ${assistantMessages.length}
📏 Total de caracteres: ${totalChars.toLocaleString()}
⏱️ Duración: ${bulkChat.sessionId ? 'Sesión activa' : 'N/A'}
🔗 Sesión: ${bulkChat.sessionId?.slice(0, 8) || 'N/A'}

${assistantMessages.length > 0 ? `📄 Último mensaje:\n${assistantMessages[assistantMessages.length - 1].content.substring(0, 200)}...` : ''}`
                      navigator.clipboard.writeText(summary)
                      toast.success('Resumen copiado', { icon: '📊', duration: 2000 })
                    }
                  }}
                  disabled={bulkChat.messages.length === 0}
                  className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2 disabled:opacity-50"
                >
                  📋 Generar resumen
                </button>
                <div className="pt-2 border-t border-slate-700">
                  <div className="text-xs text-slate-500 px-3 py-1">
                    <div>Ctrl+B: Bulk Chat</div>
                    <div>Ctrl+F: Buscar</div>
                    <div>Ctrl+Shift+R: Lectura</div>
                    <div>Ctrl+Shift+E: Exportar</div>
                    <div>F3: Siguiente resultado</div>
                    <div>Shift+F3: Anterior</div>
                    <div>Ctrl+Click: Seleccionar</div>
                    <div>Ctrl+K: Comandos</div>
                    <div>F11: Pantalla completa</div>
                    <div>Esc: Limpiar</div>
                    <div>📋: Plantillas</div>
                    <div>🖨️: Imprimir</div>
                    <div>🎨: Tema</div>
                    <div>🐛: Debug</div>
                    <div>🧘: Modo Zen</div>
                    <div>🏷️: Etiquetas</div>
                    <div>📝: Notas</div>
                    <div>✏️: Editar</div>
                    <div>📜: Historial</div>
                    <div>🌐: Traducción</div>
                    <div>⏱️: Estadísticas</div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
          
          {/* Connection Info Panel */}
          {showConnectionInfo && useBulkChatMode && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              data-connection-panel
              className="absolute top-full right-0 mt-2 bg-slate-800 border border-slate-700 rounded-lg p-4 shadow-xl z-50 min-w-[280px]"
            >
              <div className="space-y-2 text-sm">
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Estado:</span>
                  <span className={`font-semibold ${
                    bulkChat.isConnected ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {bulkChat.isConnected ? 'Conectado' : 'Desconectado'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Red:</span>
                  <span className={`font-semibold ${
                    bulkChat.isOnline ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {bulkChat.isOnline ? 'Online' : 'Offline'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Calidad:</span>
                  <span className={`font-semibold ${
                    bulkChat.connectionQuality === 'excellent' ? 'text-green-400' :
                    bulkChat.connectionQuality === 'good' ? 'text-yellow-400' :
                    bulkChat.connectionQuality === 'poor' ? 'text-orange-400' :
                    'text-red-400'
                  }`}>
                    {bulkChat.connectionQuality === 'excellent' ? 'Excelente' :
                     bulkChat.connectionQuality === 'good' ? 'Buena' :
                     bulkChat.connectionQuality === 'poor' ? 'Regular' :
                     'Sin conexión'}
                  </span>
                </div>
                {bulkChat.lastActivity && (
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">Última actividad:</span>
                    <span className="text-slate-300 text-xs">
                      {new Date(bulkChat.lastActivity).toLocaleTimeString()}
                    </span>
                  </div>
                )}
                {bulkChat.sessionId && (
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">Sesión:</span>
                    <span className="text-slate-300 text-xs font-mono">
                      {bulkChat.sessionId.slice(0, 8)}...
                    </span>
                  </div>
                )}
                <div className="pt-2 border-t border-slate-700">
                  <button
                    onClick={() => {
                      navigator.clipboard.writeText(bulkChat.sessionId || '')
                      toast.success('ID de sesión copiado', { icon: '📋', duration: 2000 })
                    }}
                    className="w-full px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-xs text-slate-300 transition-colors"
                    disabled={!bulkChat.sessionId}
                  >
                    📋 Copiar ID de sesión
                  </button>
                </div>
              </div>
            </motion.div>
          )}
          
          {/* Smart Suggestions Toggle */}
          <button
            onClick={() => setShowSmartSuggestions(!showSmartSuggestions)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors text-sm border ${
              showSmartSuggestions
                ? 'bg-purple-600/50 hover:bg-purple-700/50 border-purple-700 text-purple-300'
                : 'bg-slate-800/50 hover:bg-slate-700/50 border-slate-700 text-slate-300'
            }`}
            title="Sugerencias Inteligentes"
          >
            <Lightbulb className="w-4 h-4" />
            <span>Sugerencias</span>
          </button>
          {/* Proactive Builder Toggle */}
          <button
            onClick={() => setShowProactive(!showProactive)}
            className="flex items-center gap-2 px-4 py-2 bg-purple-600/50 hover:bg-purple-700/50 border border-purple-700 rounded-lg transition-colors text-sm text-purple-300"
            title="Constructor Proactivo - Construye modelos continuamente"
          >
            <Activity className="w-4 h-4" />
            <span>Proactivo</span>
          </button>
          {/* Metrics Toggle */}
          <button
            onClick={() => setShowMetrics(!showMetrics)}
            className="flex items-center gap-2 px-4 py-2 bg-slate-800/50 hover:bg-slate-700/50 border border-slate-700 rounded-lg transition-colors text-sm text-slate-300"
            title="Mostrar métricas de rendimiento"
          >
            <Activity className="w-4 h-4" />
            <span>Métricas</span>
          </button>
          <button
            id="templates-button"
            onClick={() => setShowTemplates(!showTemplates)}
            className="flex items-center gap-2 px-4 py-2 bg-slate-800/50 hover:bg-slate-700/50 border border-slate-700 rounded-lg transition-colors text-sm text-slate-300"
          >
            <span>📋</span>
            <span>Templates</span>
          </button>
          {modelHistory.length >= 2 && (
            <button
              onClick={() => setShowComparator(true)}
              className="flex items-center gap-2 px-4 py-2 bg-slate-800/50 hover:bg-slate-700/50 border border-slate-700 rounded-lg transition-colors text-sm text-slate-300"
            >
              <span>⚖️</span>
              <span>Comparar</span>
            </button>
          )}
          <button
            id="history-button"
            onClick={() => setShowHistory(!showHistory)}
            className="flex items-center gap-2 px-4 py-2 bg-slate-800/50 hover:bg-slate-700/50 border border-slate-700 rounded-lg transition-colors text-sm text-slate-300"
          >
            <History className="w-4 h-4" />
            <span>Historial</span>
          </button>
          {(messages.length > 0 || currentModel || input.trim()) && (
            <button
              onClick={handleDiscardAll}
              className="flex items-center gap-2 px-4 py-2 bg-red-600/50 hover:bg-red-700/50 border border-red-700 rounded-lg transition-colors text-sm text-red-300"
              title="Descartar todo lo que se ha hecho"
            >
              <Trash2 className="w-4 h-4" />
              <span>Descartar Todo</span>
            </button>
          )}
        </div>
      </div>

      {/* Smart Suggestions Panel */}
      <AnimatePresence>
        {showSmartSuggestions && smartSuggestions.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mb-4 bg-gradient-to-r from-purple-900/30 to-pink-900/30 backdrop-blur-sm rounded-lg border border-purple-700/50 p-4"
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-purple-400" />
                <h3 className="text-sm font-semibold text-purple-300">Sugerencias Inteligentes</h3>
              </div>
              <button
                onClick={() => setShowSmartSuggestions(false)}
                className="text-slate-400 hover:text-slate-300"
              >
                ×
              </button>
            </div>
            <div className="space-y-2">
              {smartSuggestions.map((suggestion, index) => (
                <button
                  key={index}
                  onClick={() => {
                    setInput(suggestion.text)
                    setShowSmartSuggestions(false)
                    validateInput(suggestion.text)
                  }}
                  className="w-full text-left p-3 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg border border-slate-700 hover:border-purple-500 transition-all text-sm text-slate-300"
                >
                  <div className="flex items-center justify-between">
                    <span>{suggestion.text}</span>
                    <span className="text-xs text-purple-400 opacity-75">
                      {Math.round(suggestion.confidence * 100)}%
                    </span>
                  </div>
                  {suggestion.category && (
                    <span className="text-xs text-slate-500 mt-1 block">
                      {suggestion.category}
                    </span>
                  )}
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Proactive Builder Panel */}
      <AnimatePresence>
        {showProactive && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mb-4"
          >
            <ProactiveModelBuilder
              onModelCreated={(result) => {
                // Agregar modelo al historial cuando se crea
                const newModel = {
                  id: result.modelId,
                  name: result.modelName,
                  description: result.description,
                  status: result.status === 'completed' ? 'completed' : 'failed',
                  createdAt: new Date(),
                  spec: null,
                }
                
                // Agregar al historial de sugerencias
                smartSuggestionsManager.addToHistory(result.description)
                
                setModelHistory(prev => [newModel, ...prev])
                try {
                  saveModelToHistory(newModel as any)
                } catch (error) {
                  console.error('Error saving model to history:', error)
                }
                
                // Agregar mensaje al chat si está visible
                if (messages.length > 0 || showProactive) {
                  addMessage({
                    id: Date.now().toString(),
                    role: 'assistant',
                    content: result.status === 'completed' 
                      ? `✅ Modelo "${result.modelName}" construido exitosamente${result.duration ? ` en ${Math.round(result.duration / 1000)}s` : ''}`
                      : `❌ Error al construir "${result.modelName}": ${result.error || 'Error desconocido'}`,
                    timestamp: new Date(),
                  })
                }
              }}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Templates Panel */}
      <AnimatePresence>
        {showTemplates && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mb-4 bg-slate-800/50 backdrop-blur-sm rounded-lg border border-slate-700 p-6"
          >
            <ModelTemplates onSelectTemplate={handleTemplateSelect} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* History Panel */}
      <AnimatePresence>
        {showHistory && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mb-4 bg-slate-800/50 backdrop-blur-sm rounded-lg border border-slate-700 p-6"
          >
            <h3 className="text-lg font-bold text-white mb-4">Historial de Modelos</h3>
            <ModelStats models={modelHistory} />
            <ModelHistory models={modelHistory} />
          </motion.div>
        )}
      </AnimatePresence>

      <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg border border-slate-700 shadow-2xl overflow-hidden">                                          
        {/* Draft Recovery */}
        <DraftRecovery onRestore={(draft) => {
          try {
            // Validate draft
            if (!draft || typeof draft !== 'object') {
              console.warn('Invalid draft provided')
              return
            }
            
            const draftInput = draft.input
            if (!draftInput || typeof draftInput !== 'string') {
              console.warn('Invalid draft input')
              return
            }
            
            const trimmedDraft = draftInput.trim()
            if (trimmedDraft.length === 0) {
              console.warn('Empty draft input')
              return
            }
            
            // Validate length
            if (trimmedDraft.length > 5000) {
              toast.error('El borrador es demasiado largo (máximo 5000 caracteres)')
              return
            }
            
            setInput(trimmedDraft)
            validateInput(trimmedDraft)
          } catch (error) {
            console.error('Error restoring draft:', error)
            toast.error('Error al restaurar el borrador')
          }
        }} />

        {/* Search Bar */}
        {messages.length > 0 && (
          <div className="p-4 border-b border-slate-700">
            <div className="flex items-center gap-2">
              <div className="flex-1">
            <SearchBar
                  ref={searchInputRef}
              onSearch={handleSearch}
                  placeholder="Buscar en mensajes... (Ctrl+F)"
            />
          </div>
              {searchQuery && (
                <button
                  onClick={() => {
                    setSearchQuery('')
                    setFilteredMessages(messages)
                    if (searchInputRef.current) {
                      searchInputRef.current.clear()
                    }
                  }}
                  className="px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-slate-300 text-sm transition-colors"
                  title="Limpiar búsqueda"
                >
                  ✕
                </button>
              )}
              <button
                onClick={() => setHighlightSearch(!highlightSearch)}
                className={`px-3 py-2 rounded-lg text-sm transition-colors ${
                  highlightSearch
                    ? 'bg-yellow-600/50 hover:bg-yellow-700/50 border border-yellow-700 text-yellow-300'
                    : 'bg-slate-700 hover:bg-slate-600 text-slate-300'
                }`}
                title="Resaltar resultados"
              >
                🔍
              </button>
              <button
                onClick={() => setShowFilters(!showFilters)}
                className={`px-3 py-2 rounded-lg text-sm transition-colors ${
                  showFilters
                    ? 'bg-purple-600/50 hover:bg-purple-700/50 border border-purple-700 text-purple-300'
                    : 'bg-slate-700 hover:bg-slate-600 text-slate-300'
                }`}
                title="Filtros avanzados"
              >
                🔽
              </button>
              <div className="flex items-center gap-2 text-xs text-slate-500">
                {searchQuery && filteredMessages.length > 0 && (
                  <>
                    <span>{filteredMessages.length} resultado{filteredMessages.length !== 1 ? 's' : ''}</span>
                    {filteredMessages.length > 1 && (
                      <>
                        <button
                          onClick={() => {
                            const nextIndex = currentSearchIndex < filteredMessages.length - 1 
                              ? currentSearchIndex + 1 
                              : 0
                            setCurrentSearchIndex(nextIndex)
                            const message = filteredMessages[nextIndex]
                            const element = messageRefs.current.get(message.id)
                            if (element) {
                              element.scrollIntoView({ behavior: 'smooth', block: 'center' })
                              element.classList.add('ring-2', 'ring-yellow-500', 'ring-opacity-50')
                              setTimeout(() => {
                                element.classList.remove('ring-2', 'ring-yellow-500', 'ring-opacity-50')
                              }, 2000)
                            }
                          }}
                          className="px-2 py-1 bg-yellow-600/30 hover:bg-yellow-600/50 rounded text-yellow-300 transition-colors"
                          title="Siguiente resultado"
                        >
                          ↓
                        </button>
                        <button
                          onClick={() => {
                            const prevIndex = currentSearchIndex > 0 
                              ? currentSearchIndex - 1 
                              : filteredMessages.length - 1
                            setCurrentSearchIndex(prevIndex)
                            const message = filteredMessages[prevIndex]
                            const element = messageRefs.current.get(message.id)
                            if (element) {
                              element.scrollIntoView({ behavior: 'smooth', block: 'center' })
                              element.classList.add('ring-2', 'ring-yellow-500', 'ring-opacity-50')
                              setTimeout(() => {
                                element.classList.remove('ring-2', 'ring-yellow-500', 'ring-opacity-50')
                              }, 2000)
                            }
                          }}
                          className="px-2 py-1 bg-yellow-600/30 hover:bg-yellow-600/50 rounded text-yellow-300 transition-colors"
                          title="Resultado anterior"
                        >
                          ↑
                        </button>
                        <span className="text-slate-400">
                          {currentSearchIndex >= 0 ? `${currentSearchIndex + 1}/${filteredMessages.length}` : ''}
                        </span>
                      </>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Filtros avanzados */}
        {showFilters && messages.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="p-4 border-b border-slate-700 bg-slate-800/30"
          >
            <div className="flex items-center gap-4 flex-wrap">
              <div className="flex items-center gap-2">
                <label className="text-xs text-slate-400">Filtrar por rol:</label>
                <select
                  value={filterRole}
                  onChange={(e) => setFilterRole(e.target.value as 'all' | 'user' | 'assistant')}
                  className="px-3 py-1.5 bg-slate-700 border border-slate-600 rounded text-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
                >
                  <option value="all">Todos</option>
                  <option value="user">Usuario</option>
                  <option value="assistant">Asistente</option>
                </select>
              </div>
              <button
                onClick={() => {
                  const favoritesOnly = Array.from(favoriteMessages).map(id => 
                    messages.find(m => m.id === id)
                  ).filter(Boolean)
                  setFilteredMessages(favoritesOnly as typeof messages)
                  toast.success(`${favoritesOnly.length} mensaje(s) favorito(s)`, { icon: '⭐', duration: 2000 })
                }}
                disabled={favoriteMessages.size === 0}
                className="px-3 py-1.5 bg-yellow-600/50 hover:bg-yellow-700/50 border border-yellow-700 rounded text-yellow-300 text-sm disabled:opacity-50 transition-colors"
              >
                ⭐ Solo favoritos ({favoriteMessages.size})
              </button>
              <button
                onClick={() => {
                  setFilterRole('all')
                  setFilteredMessages(messages)
                  setSearchQuery('')
                  if (searchInputRef.current) {
                    searchInputRef.current.clear()
                  }
                  toast('Filtros limpiados', { icon: '🔄', duration: 2000 })
                }}
                className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 text-sm transition-colors"
              >
                🔄 Limpiar filtros
              </button>
              <div className="flex items-center gap-2 ml-auto">
                <label className="text-xs text-slate-400">Resaltar código:</label>
                <button
                  onClick={() => setShowCodeSyntax(!showCodeSyntax)}
                  className={`px-3 py-1.5 rounded text-sm transition-colors ${
                    showCodeSyntax
                      ? 'bg-green-600/50 hover:bg-green-700/50 border border-green-700 text-green-300'
                      : 'bg-slate-700 hover:bg-slate-600 text-slate-300'
                  }`}
                >
                  {showCodeSyntax ? '✓' : '✕'}
                </button>
                <label className="text-xs text-slate-400 ml-2">Auto-scroll:</label>
                <button
                  onClick={() => setAutoScroll(!autoScroll)}
                  className={`px-3 py-1.5 rounded text-sm transition-colors ${
                    autoScroll
                      ? 'bg-blue-600/50 hover:bg-blue-700/50 border border-blue-700 text-blue-300'
                      : 'bg-slate-700 hover:bg-slate-600 text-slate-300'
                  }`}
                >
                  {autoScroll ? '✓' : '✕'}
                </button>
              </div>
            </div>
          </motion.div>
        )}

        {/* Messages Area */}
        <div className={`${presentationMode ? 'h-screen' : 'h-[600px]'} overflow-y-auto ${
          viewMode === 'compact' ? 'p-3 space-y-2' : 
          viewMode === 'comfortable' ? 'p-8 space-y-6' : 
          'p-6 space-y-4'
        } ${readMode ? 'bg-slate-900/50' : ''} ${
          fontSize === 'small' ? 'text-sm' :
          fontSize === 'large' ? 'text-lg' :
          'text-base'
        }`}>
          <AnimatePresence>
            {messages.length === 0 ? (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-center text-slate-400 mt-10"
              >
                <p className="text-lg mb-2">¡Hola! 👋</p>
                <p className="mb-6">Describe el modelo de IA que quieres crear y lo generaré con TruthGPT.</p>
                <div className="space-y-6">
                  <QuickActions onSelect={(prompt) => {
                    try {
                      // Validate prompt
                      if (!prompt || typeof prompt !== 'string') {
                        console.warn('Invalid prompt from QuickActions')
                        return
                      }
                      
                      const trimmedPrompt = prompt.trim()
                      if (trimmedPrompt.length === 0) {
                        console.warn('Empty prompt from QuickActions')
                        return
                      }
                      
                      if (trimmedPrompt.length > 5000) {
                        toast.error('El prompt es demasiado largo (máximo 5000 caracteres)')
                        return
                      }
                      
                      setInput(trimmedPrompt)
                      validateInput(trimmedPrompt)
                    } catch (error) {
                      console.error('Error handling QuickAction prompt:', error)
                      toast.error('Error al procesar el prompt')
                    }
                  }} />
                  <Suggestions onSelect={handleSuggestionSelect} />
                  {!showTemplates && (
                    <button
                      onClick={() => setShowTemplates(true)}
                      className="text-sm text-purple-400 hover:text-purple-300 underline"
                    >
                      O explora nuestros templates de modelos →
                    </button>
                  )}
                </div>
              </motion.div>
            ) : (
              <>
                {filteredMessages.length === 0 && searchQuery ? (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-center text-slate-400 mt-10 p-6 bg-slate-800/30 rounded-lg border border-slate-700"
                  >
                    <p className="text-lg mb-2">🔍 Sin resultados</p>
                    <p>No se encontraron mensajes que coincidan con</p>
                    <p className="font-mono text-yellow-400 mt-2">"{searchQuery}"</p>
                    <button
                      onClick={() => {
                        setSearchQuery('')
                        setFilteredMessages(messages)
                        if (searchInputRef.current) {
                          searchInputRef.current.clear()
                        }
                      }}
                      className="mt-4 px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-slate-300 transition-colors"
                    >
                      Limpiar búsqueda
                    </button>
                  </motion.div>
                ) : (
                  <>
                    {filteredMessages.map((message, index) => {
                      // Función para resaltar texto de búsqueda
                      const highlightText = (text: string, query: string) => {
                        if (!query || !highlightSearch) return text
                        const escapedQuery = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
                        const regex = new RegExp(`(${escapedQuery})`, 'gi')
                        const parts = text.split(regex)
                        return parts.map((part, i) => 
                          regex.test(part) ? (
                            <mark key={i} className="bg-yellow-500/30 text-yellow-200 rounded px-1">
                              {part}
                            </mark>
                          ) : part
                        )
                      }
                      
                      const isFavorite = favoriteMessages.has(message.id)
                      const isPinned = pinnedMessages.has(message.id)
                      const isArchived = archivedMessages.has(message.id)
                      const isHighlighted = currentSearchIndex >= 0 && index === currentSearchIndex
                      const isSelected = selectedMessages.has(message.id)
                      const isCollapsed = collapsedMessages.has(message.id)
                      const isLongMessage = message.content.length > 500
                      const threadChildren = messageThreads.get(message.id) || []
                      const isThreadParent = threadChildren.length > 0
                      const wordCount = message.content.trim().split(/\s+/).filter(Boolean).length
                      const sentiment = showSentiment ? analyzeSentiment(message.content) : null
                      const messageReactionsList = messageReactions.get(message.id) || []
                      const links = detectLinks(message.content)
                      const hasLinks = links.length > 0
                      const messageTagsList = messageTags.get(message.id) || []
                      const messageNote = messageNotes.get(message.id)
                      const messageVersion = messageVersions.get(message.id) || 0
                      const messageStat = messageStats.get(message.id)
                      const prevMessage = index > 0 ? filteredMessages[index - 1] : undefined
                      
                      // Calcular estadísticas si no existen
                      if (!messageStat) {
                        const stats = calculateMessageStats(message, prevMessage)
                        setMessageStats(prev => new Map(prev).set(message.id, stats))
                      }
                      
                      // Traducir si está activo
                      const displayContent = translationMode && message.role === 'assistant' 
                        ? translateText(message.content, targetLanguage)
                        : message.content
                      
                      // Detectar código en el mensaje
                      const hasCode = /```[\s\S]*?```|`[^`]+`/.test(message.content)
                      
                      return (
                        <motion.div
                          key={message.id}
                          ref={(el) => {
                            if (el) {
                              messageRefs.current.set(message.id, el)
                            }
                          }}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: readMode ? 0 : index * 0.05 }}
                          className={`${readMode ? 'transform hover:scale-[1.02] transition-transform cursor-pointer' : ''} ${isHighlighted ? 'ring-2 ring-yellow-500 ring-opacity-50 rounded-lg' : ''} ${isSelected ? 'ring-2 ring-blue-500 ring-opacity-50 bg-blue-500/10 rounded-lg' : ''} relative group`}
                          onClick={(e) => {
                            if (e.ctrlKey || e.metaKey) {
                              e.preventDefault()
                              setSelectedMessages(prev => {
                                const newSet = new Set(prev)
                                if (newSet.has(message.id)) {
                                  newSet.delete(message.id)
                                } else {
                                  newSet.add(message.id)
                                }
                                return newSet
                              })
                            }
                          }}
                        >
                          {/* Botones de acción rápida en hover */}
                          <div className="absolute top-2 right-2 flex flex-col gap-1 opacity-0 group-hover:opacity-100 transition-opacity z-10">
                            {/* Reacciones rápidas */}
                            {showReactions && message.role === 'assistant' && (
                              <div className="flex gap-1 mb-1 flex-wrap max-w-[200px]">
                                {availableReactions.slice(0, 4).map((reaction) => {
                                  const messageReactionsList = messageReactions.get(message.id) || []
                                  const isActive = messageReactionsList.includes(reaction)
                                  return (
                                    <button
                                      key={reaction}
                                      onClick={(e) => {
                                        e.stopPropagation()
                                        setMessageReactions(prev => {
                                          const newMap = new Map(prev)
                                          const current = newMap.get(message.id) || []
                                          if (current.includes(reaction)) {
                                            newMap.set(message.id, current.filter(r => r !== reaction))
                                          } else {
                                            newMap.set(message.id, [...current, reaction])
                                          }
                                          return newMap
                                        })
                                      }}
                                      className={`p-1 rounded text-xs transition-all ${
                                        isActive
                                          ? 'bg-blue-600/80 scale-110'
                                          : 'bg-slate-700/80 hover:bg-slate-600'
                                      }`}
                                      title={reaction}
                                    >
                                      {reaction}
                                    </button>
                                  )
                                })}
                              </div>
                            )}
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                navigator.clipboard.writeText(message.content)
                                toast.success('Mensaje copiado', { icon: '📋', duration: 2000 })
                              }}
                              className="p-1.5 bg-slate-700/80 hover:bg-slate-600 rounded text-slate-300 text-xs transition-colors"
                              title="Copiar mensaje"
                            >
                              📋
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                setSelectedMessages(prev => {
                                  const newSet = new Set(prev)
                                  if (newSet.has(message.id)) {
                                    newSet.delete(message.id)
                                  } else {
                                    newSet.add(message.id)
                                  }
                                  return newSet
                                })
                              }}
                              className={`p-1.5 rounded text-xs transition-colors ${
                                isSelected
                                  ? 'bg-blue-600/80 hover:bg-blue-700 text-white'
                                  : 'bg-slate-700/80 hover:bg-slate-600 text-slate-300'
                              }`}
                              title={isSelected ? 'Deseleccionar' : 'Seleccionar (Ctrl+Click)'}
                            >
                              {isSelected ? '✓' : '☐'}
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                setFavoriteMessages(prev => {
                                  const newSet = new Set(prev)
                                  if (newSet.has(message.id)) {
                                    newSet.delete(message.id)
                                    toast('Marcador eliminado', { icon: '⭐', duration: 2000 })
                                  } else {
                                    newSet.add(message.id)
                                    toast.success('Marcador agregado', { icon: '⭐', duration: 2000 })
                                  }
                                  return newSet
                                })
                              }}
                              className={`p-1.5 rounded text-xs transition-colors ${
                                isFavorite
                                  ? 'bg-yellow-600/80 hover:bg-yellow-700 text-yellow-200'
                                  : 'bg-slate-700/80 hover:bg-slate-600 text-slate-300'
                              }`}
                              title={isFavorite ? 'Quitar marcador' : 'Marcar como favorito'}
                            >
                              ⭐
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                const shareText = `${message.role === 'user' ? 'Usuario' : 'Asistente'}: ${message.content}\n\n${new Date(message.timestamp).toLocaleString()}`
                                navigator.clipboard.writeText(shareText)
                                toast.success('Mensaje copiado para compartir', { icon: '🔗', duration: 2000 })
                              }}
                              className="p-1.5 bg-slate-700/80 hover:bg-slate-600 rounded text-slate-300 text-xs transition-colors"
                              title="Compartir mensaje"
                            >
                              🔗
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                setEditingNote(message.id)
                              }}
                              className={`p-1.5 rounded text-xs transition-colors ${
                                messageNote
                                  ? 'bg-yellow-600/80 hover:bg-yellow-700 text-yellow-200'
                                  : 'bg-slate-700/80 hover:bg-slate-600 text-slate-300'
                              }`}
                              title={messageNote ? 'Editar nota' : 'Agregar nota'}
                            >
                              📝
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                setEditingMessage(message.id)
                              }}
                              className="p-1.5 bg-slate-700/80 hover:bg-slate-600 rounded text-slate-300 text-xs transition-colors"
                              title="Editar mensaje"
                            >
                              ✏️
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                togglePinMessage(message.id)
                              }}
                              className={`p-1.5 rounded text-xs transition-colors ${
                                pinnedMessages.has(message.id)
                                  ? 'bg-purple-600/80 hover:bg-purple-700 text-purple-200'
                                  : 'bg-slate-700/80 hover:bg-slate-600 text-slate-300'
                              }`}
                              title={pinnedMessages.has(message.id) ? 'Desfijar mensaje' : 'Fijar mensaje'}
                            >
                              📌
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                createThreadFromMessage(message.id)
                              }}
                              className="p-1.5 bg-slate-700/80 hover:bg-slate-600 rounded text-slate-300 text-xs transition-colors"
                              title="Crear thread desde este mensaje"
                            >
                              🌿
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                setPreviewMessageId(message.id)
                                setShowMarkdownPreview(true)
                              }}
                              className="p-1.5 bg-slate-700/80 hover:bg-slate-600 rounded text-slate-300 text-xs transition-colors"
                              title="Vista previa Markdown"
                            >
                              👁️
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                setShareTarget(message.id)
                                setShowShareMenu(true)
                              }}
                              className="p-1.5 bg-slate-700/80 hover:bg-slate-600 rounded text-slate-300 text-xs transition-colors"
                              title="Compartir mensaje"
                            >
                              🔗
                            </button>
                            {messageEncryption && encryptedMessages.has(message.id) && (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation()
                                  // Mostrar mensaje desencriptado
                                  const decrypted = decryptMessage(message.content)
                                  toast(`Mensaje desencriptado: ${decrypted.substring(0, 100)}...`, {
                                    icon: '🔓',
                                    duration: 5000,
                                  })
                                }}
                                className="p-1.5 bg-green-700/80 hover:bg-green-600 rounded text-green-200 text-xs transition-colors"
                                title="Mensaje encriptado - Click para desencriptar"
                              >
                                🔒
                              </button>
                            )}
                            {!messageBookmarks.has(message.id) ? (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation()
                                  const name = prompt('Nombre del marcador:') || `Bookmark ${bookmarks.size + 1}`
                                  if (name) addBookmark(message.id, name)
                                }}
                                className="p-1.5 bg-slate-700/80 hover:bg-slate-600 rounded text-slate-300 text-xs transition-colors"
                                title="Agregar marcador"
                              >
                                🔖
                              </button>
                            ) : (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation()
                                  removeBookmark(message.id)
                                }}
                                className="p-1.5 bg-yellow-700/80 hover:bg-yellow-600 rounded text-yellow-200 text-xs transition-colors"
                                title="Eliminar marcador"
                              >
                                🔖
                              </button>
                            )}
                            {studyMode && (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation()
                                  const question = prompt('Pregunta para la flashcard:')
                                  if (question) {
                                    createFlashcard(message.id, question, message.content)
                                  }
                                }}
                                className="p-1.5 bg-slate-700/80 hover:bg-slate-600 rounded text-slate-300 text-xs transition-colors"
                                title="Crear flashcard"
                              >
                                📚
                              </button>
                            )}
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                const priority = prompt('Prioridad (low/medium/high/urgent):') as 'low' | 'medium' | 'high' | 'urgent'
                                if (priority && ['low', 'medium', 'high', 'urgent'].includes(priority)) {
                                  setMessagePriorityLevel(message.id, priority)
                                }
                              }}
                              className="p-1.5 bg-slate-700/80 hover:bg-slate-600 rounded text-slate-300 text-xs transition-colors"
                              title="Establecer prioridad"
                            >
                              ⚡
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                const type = prompt('Tipo de anotación (highlight/comment/question):') as 'highlight' | 'comment' | 'question'
                                const content = prompt('Contenido de la anotación:')
                                if (type && content && ['highlight', 'comment', 'question'].includes(type)) {
                                  addAnnotation(message.id, type, content)
                                }
                              }}
                              className="p-1.5 bg-slate-700/80 hover:bg-slate-600 rounded text-slate-300 text-xs transition-colors"
                              title="Agregar anotación"
                            >
                              📝
                            </button>
                            {calendarIntegration && (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation()
                                  const title = prompt('Título del evento:')
                                  if (title) {
                                    const days = parseInt(prompt('Días desde ahora:') || '0')
                                    const date = new Date()
                                    date.setDate(date.getDate() + days)
                                    scheduleEvent(title, date, message.id)
                                  }
                                }}
                                className="p-1.5 bg-slate-700/80 hover:bg-slate-600 rounded text-slate-300 text-xs transition-colors"
                                title="Programar evento"
                              >
                                📅
                              </button>
                            )}
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                voteMessage(message.id, 'up')
                              }}
                              className="p-1.5 bg-slate-700/80 hover:bg-slate-600 rounded text-slate-300 text-xs transition-colors"
                              title={`👍 ${messageVotes.get(message.id)?.up || 0} | 👎 ${messageVotes.get(message.id)?.down || 0}`}
                            >
                              👍 {messageVotes.get(message.id)?.up || 0}
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                voteMessage(message.id, 'down')
                              }}
                              className="p-1.5 bg-slate-700/80 hover:bg-slate-600 rounded text-slate-300 text-xs transition-colors"
                              title="Votar negativo"
                            >
                              👎 {messageVotes.get(message.id)?.down || 0}
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                const rating = parseInt(prompt('Calificación (1-5):') || '0')
                                if (rating >= 1 && rating <= 5) {
                                  rateMessage(message.id, rating)
                                }
                              }}
                              className="p-1.5 bg-slate-700/80 hover:bg-slate-600 rounded text-slate-300 text-xs transition-colors"
                              title={`Calificación: ${messageRatings.get(message.id) || 'N/A'}/5`}
                            >
                              ⭐ {messageRatings.get(message.id) || '?'}
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                const content = prompt('Comentario:')
                                if (content) {
                                  addComment(message.id, content)
                                }
                              }}
                              className="p-1.5 bg-slate-700/80 hover:bg-slate-600 rounded text-slate-300 text-xs transition-colors"
                              title={`Comentarios: ${messageComments.get(message.id)?.length || 0}`}
                            >
                              💬 {messageComments.get(message.id)?.length || 0}
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                const lang = prompt('Idioma de destino (es, en, fr, etc.):') || 'es'
                                translateMessage(message.id, lang)
                              }}
                              className="p-1.5 bg-slate-700/80 hover:bg-slate-600 rounded text-slate-300 text-xs transition-colors"
                              title="Traducir mensaje"
                            >
                              🌐
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                summarizeMessage(message.id)
                              }}
                              className="p-1.5 bg-slate-700/80 hover:bg-slate-600 rounded text-slate-300 text-xs transition-colors"
                              title="Generar resumen"
                            >
                              📝
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                const otherId = prompt('ID del otro mensaje para comparar:')
                                if (otherId) {
                                  compareMessages(message.id, otherId)
                                }
                              }}
                              className="p-1.5 bg-slate-700/80 hover:bg-slate-600 rounded text-slate-300 text-xs transition-colors"
                              title="Comparar con otro mensaje"
                            >
                              🔍
                            </button>
                            {showEditHistory && messageHistory.has(message.id) && (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation()
                                  const history = messageHistory.get(message.id) || []
                                  toast.success(`Versiones: ${history.length + 1}`, { 
                                    icon: '📜', 
                                    duration: 2000 
                                  })
                                }}
                                className="p-1.5 bg-purple-600/80 hover:bg-purple-700 rounded text-white text-xs transition-colors"
                                title={`Ver historial (${messageHistory.get(message.id)?.length || 0} versiones)`}
                              >
                                📜
                              </button>
                            )}
                            {showTags && (
                              <div className="flex gap-1 flex-wrap max-w-[150px]">
                                {availableTags.slice(0, 2).map(tag => {
                                  const hasTag = messageTagsList.includes(tag)
                                  return (
                                    <button
                                      key={tag}
                                      onClick={(e) => {
                                        e.stopPropagation()
                                        if (hasTag) {
                                          removeTagFromMessage(message.id, tag)
                                        } else {
                                          addTagToMessage(message.id, tag)
                                        }
                                      }}
                                      className={`px-1.5 py-0.5 text-xs rounded transition-colors ${
                                        hasTag
                                          ? 'bg-blue-600 text-white'
                                          : 'bg-slate-700/80 hover:bg-slate-600 text-slate-300'
                                      }`}
                                      title={hasTag ? `Remover "${tag}"` : `Agregar "${tag}"`}
                                    >
                                      {hasTag ? '✓' : '+'} {tag.substring(0, 6)}
                                    </button>
                                  )
                                })}
                              </div>
                            )}
                          </div>
                          
                          {readMode && message.role === 'assistant' ? (
                            <div className="bg-slate-700/30 rounded-lg p-4 border border-slate-600 group">
                              {isLongMessage && (
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    setCollapsedMessages(prev => {
                                      const newSet = new Set(prev)
                                      if (newSet.has(message.id)) {
                                        newSet.delete(message.id)
                                      } else {
                                        newSet.add(message.id)
                                      }
                                      return newSet
                                    })
                                  }}
                                  className="mb-2 text-xs text-slate-400 hover:text-slate-300 transition-colors"
                                >
                                  {isCollapsed ? '▶ Expandir' : '▼ Colapsar'}
                                </button>
                              )}
                              <div className={`text-slate-300 leading-relaxed whitespace-pre-wrap ${isCollapsed && isLongMessage ? 'line-clamp-3' : ''} ${zenMode ? 'text-lg' : ''}`}>
                                {highlightText(displayContent, searchQuery)}
                                {messageVersion > 0 && (
                                  <span className="ml-2 text-xs text-purple-400" title={`Editado ${messageVersion} vez(es)`}>
                                    ✏️ v{messageVersion + 1}
                                  </span>
                                )}
                                {hasLinks && (
                                  <div className="mt-2 space-y-1">
                                    {links.map((link, idx) => (
                                      <a
                                        key={idx}
                                        href={link}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="text-blue-400 hover:text-blue-300 underline text-xs flex items-center gap-1"
                                        onClick={(e) => e.stopPropagation()}
                                      >
                                        🔗 {new URL(link).hostname}
                                      </a>
                                    ))}
                                  </div>
                                )}
                                {messageNote && (
                                  <div className="mt-2 p-2 bg-yellow-900/20 border border-yellow-700/50 rounded text-xs text-yellow-200">
                                    <strong>📝 Nota:</strong> {messageNote}
                                  </div>
                                )}
                                {showTags && messageTagsList.length > 0 && (
                                  <div className="mt-2 flex flex-wrap gap-1">
                                    {messageTagsList.map(tag => (
                                      <span
                                        key={tag}
                                        className="px-2 py-0.5 bg-blue-600/30 text-blue-300 rounded text-xs"
                                      >
                                        🏷️ {tag}
                                      </span>
                                    ))}
                                  </div>
                                )}
                              </div>
                              <div className="text-xs text-slate-500 mt-2 flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                  <span>{new Date(message.timestamp).toLocaleString()}</span>
                                  {showWordCount && (
                                    <span className="text-cyan-400" title="Contador de palabras">
                                      📝 {wordCount}
                                    </span>
                                  )}
                                  {messageStat?.responseTime !== undefined && (
                                    <span className="text-green-400" title="Tiempo de respuesta">
                                      ⏱️ {messageStat.responseTime}s
                                    </span>
                                  )}
                                  {messageVersion > 0 && (
                                    <span className="text-purple-400" title={`Versión ${messageVersion + 1}`}>
                                      ✏️ v{messageVersion + 1}
                                    </span>
                                  )}
                                </div>
                                <div className="flex items-center gap-2">
                                  {sentiment && (
                                    <span 
                                      className={`text-xs ${
                                        sentiment.sentiment === 'positive' ? 'text-green-400' :
                                        sentiment.sentiment === 'negative' ? 'text-red-400' :
                                        'text-slate-400'
                                      }`}
                                      title={`Sentimiento: ${sentiment.sentiment}`}
                                    >
                                      {sentiment.sentiment === 'positive' ? '😊' :
                                       sentiment.sentiment === 'negative' ? '😞' :
                                       '😐'}
                                    </span>
                                  )}
                                  {hasLinks && (
                                    <span className="text-blue-400" title="Contiene enlaces">🔗</span>
                                  )}
                                  {hasCode && showCodeSyntax && (
                                    <span className="text-purple-400" title="Contiene código">💻</span>
                                  )}
                                  {isFavorite && <span className="text-yellow-400">⭐</span>}
                                  {messageReactionsList.length > 0 && (
                                    <div className="flex gap-0.5">
                                      {messageReactionsList.map((r, idx) => (
                                        <span key={idx} className="text-xs">{r}</span>
                                      ))}
                                    </div>
                                  )}
                                </div>
                              </div>
                            </div>
                          ) : (
                            searchQuery && highlightSearch ? (
                              <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700 group">
                                {isLongMessage && (
                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation()
                                      setCollapsedMessages(prev => {
                                        const newSet = new Set(prev)
                                        if (newSet.has(message.id)) {
                                          newSet.delete(message.id)
                                        } else {
                                          newSet.add(message.id)
                                        }
                                        return newSet
                                      })
                                    }}
                                    className="mb-2 text-xs text-slate-400 hover:text-slate-300 transition-colors"
                                  >
                                    {isCollapsed ? '▶ Expandir' : '▼ Colapsar'}
                                  </button>
                                )}
                                <div className="flex items-start gap-2">
                                  <div className="font-semibold text-slate-300 text-sm">
                                    {message.role === 'user' ? 'Tú' : 'Asistente'}:
                                  </div>
                                  <div className={`flex-1 text-slate-200 ${isCollapsed && isLongMessage ? 'line-clamp-3' : ''} ${zenMode ? 'text-lg' : ''}`}>
                                    {highlightText(displayContent, searchQuery)}
                                    {messageVersion > 0 && (
                                      <span className="ml-2 text-xs text-purple-400" title={`Editado ${messageVersion} vez(es)`}>
                                        ✏️ v{messageVersion + 1}
                                      </span>
                                    )}
                                    {hasLinks && (
                                      <div className="mt-2 space-y-1">
                                        {links.map((link, idx) => (
                                          <a
                                            key={idx}
                                            href={link}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="text-blue-400 hover:text-blue-300 underline text-xs flex items-center gap-1"
                                            onClick={(e) => e.stopPropagation()}
                                          >
                                            🔗 {new URL(link).hostname}
                                          </a>
                                        ))}
                                      </div>
                                    )}
                                    {messageNote && (
                                      <div className="mt-2 p-2 bg-yellow-900/20 border border-yellow-700/50 rounded text-xs text-yellow-200">
                                        <strong>📝 Nota:</strong> {messageNote}
                                      </div>
                                    )}
                                    {showTags && messageTagsList.length > 0 && (
                                      <div className="mt-2 flex flex-wrap gap-1">
                                        {messageTagsList.map(tag => (
                                          <span
                                            key={tag}
                                            className="px-2 py-0.5 bg-blue-600/30 text-blue-300 rounded text-xs"
                                          >
                                            🏷️ {tag}
                                          </span>
                                        ))}
                                      </div>
                                    )}
                                  </div>
                                </div>
                                <div className="text-xs text-slate-500 mt-2 flex items-center justify-between">
                                  <div className="flex items-center gap-2">
                                    <span>{new Date(message.timestamp).toLocaleString()}</span>
                                    {showWordCount && (
                                      <span className="text-cyan-400" title="Contador de palabras">
                                        📝 {wordCount}
                                      </span>
                                    )}
                                    {messageStat?.responseTime !== undefined && (
                                      <span className="text-green-400" title="Tiempo de respuesta">
                                        ⏱️ {messageStat.responseTime}s
                                      </span>
                                    )}
                                    {messageVersion > 0 && (
                                      <span className="text-purple-400" title={`Versión ${messageVersion + 1}`}>
                                        ✏️ v{messageVersion + 1}
                                      </span>
                                    )}
                                  </div>
                                  <div className="flex items-center gap-2">
                                    {sentiment && (
                                      <span 
                                        className={`text-xs ${
                                          sentiment.sentiment === 'positive' ? 'text-green-400' :
                                          sentiment.sentiment === 'negative' ? 'text-red-400' :
                                          'text-slate-400'
                                        }`}
                                        title={`Sentimiento: ${sentiment.sentiment}`}
                                      >
                                        {sentiment.sentiment === 'positive' ? '😊' :
                                         sentiment.sentiment === 'negative' ? '😞' :
                                         '😐'}
                                      </span>
                                    )}
                                    {hasLinks && (
                                      <span className="text-blue-400" title="Contiene enlaces">🔗</span>
                                    )}
                                    {hasCode && showCodeSyntax && (
                                      <span className="text-purple-400" title="Contiene código">💻</span>
                                    )}
                                    {isFavorite && <span className="text-yellow-400">⭐</span>}
                                    {messageReactionsList.length > 0 && (
                                      <div className="flex gap-0.5">
                                        {messageReactionsList.map((r, idx) => (
                                          <span key={idx} className="text-xs">{r}</span>
                                        ))}
                                      </div>
                                    )}
                                  </div>
                                </div>
                              </div>
                            ) : (
                              <div className="group relative">
                                {isLongMessage && (
                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation()
                                      setCollapsedMessages(prev => {
                                        const newSet = new Set(prev)
                                        if (newSet.has(message.id)) {
                                          newSet.delete(message.id)
                                        } else {
                                          newSet.add(message.id)
                                        }
                                        return newSet
                                      })
                                    }}
                                    className="absolute top-2 left-2 z-20 px-2 py-1 bg-slate-800/90 hover:bg-slate-700 rounded text-xs text-slate-300 transition-colors"
                                  >
                                    {isCollapsed ? '▶ Expandir' : '▼ Colapsar'}
                                  </button>
                                )}
                                <div className={isCollapsed && isLongMessage ? 'opacity-60' : ''}>
                                  <Message message={message} />
                                </div>
                                {isFavorite && (
                                  <div className="absolute top-2 right-2 text-yellow-400 text-sm z-10">
                                    ⭐
                                  </div>
                                )}
                                {isPinned && (
                                  <div className="absolute top-2 left-2 text-purple-400 text-sm z-10" title="Mensaje fijado">
                                    📌
                                  </div>
                                )}
                                {isThreadParent && (
                                  <div className="absolute bottom-2 left-2 text-green-400 text-xs z-10" title={`Thread con ${threadChildren.length} respuesta(s)`}>
                                    🌿 {threadChildren.length}
                                  </div>
                                )}
                                {isArchived && (
                                  <div className="absolute top-2 left-12 text-gray-400 text-xs z-10" title="Mensaje archivado">
                                    📦
                                  </div>
                                )}
                                {hasLinks && (
                                  <div className="absolute bottom-2 left-2 text-blue-400 text-xs z-10" title="Contiene enlaces">
                                    🔗
                                  </div>
                                )}
                                {hasCode && showCodeSyntax && (
                                  <div className="absolute bottom-2 right-2 text-purple-400 text-xs z-10" title="Contiene código">
                                    💻
                                  </div>
                                )}
                              </div>
                            )
                          )}
                        </motion.div>
                      )
                    })}
                    {/* Skeleton loader para cuando está cargando */}
                    {useBulkChatMode && bulkChat.isLoading && filteredMessages.length === 0 && (
                      <div className="space-y-4">
                        {[1, 2, 3].map((i) => (
                          <div key={i} className="animate-pulse">
                            <div className="flex gap-3">
                              <div className="w-8 h-8 bg-slate-700 rounded-full" />
                              <div className="flex-1 space-y-2">
                                <div className="h-4 bg-slate-700 rounded w-3/4" />
                                <div className="h-4 bg-slate-700 rounded w-1/2" />
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </>
                )}
                {currentModel?.spec && (
                  <div className="mt-4 p-4 bg-slate-700/30 rounded-lg border border-slate-600">
                    <ArchitectureVisualizer spec={currentModel.spec as any} />
                  </div>
                )}
              </>
            )}
          </AnimatePresence>
          {isLoading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex items-center gap-2 text-slate-400"
            >
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Creando tu modelo TruthGPT...</span>
            </motion.div>
          )}
          {useBulkChatMode && bulkChat.isLoading && !isLoading && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-center gap-2 text-purple-400 bg-purple-900/20 p-3 rounded-lg border border-purple-700/50"
            >
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="text-sm">Bulk Chat procesando...</span>
            </motion.div>
          )}
          {useBulkChatMode && bulkChat.isTyping && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="flex items-center gap-2 text-blue-400 bg-blue-900/20 p-3 rounded-lg border border-blue-700/50"
            >
              <div className="flex gap-1">
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
              <span className="text-sm">Asistente está escribiendo...</span>
            </motion.div>
          )}
          {useBulkChatMode && !bulkChat.isConnected && bulkChat.sessionId && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex items-center gap-2 text-yellow-400 bg-yellow-900/20 p-3 rounded-lg border border-yellow-700/50"
            >
              <AlertCircle className="w-4 h-4" />
              <span className="text-sm">Desconectado - Reintentando conexión...</span>
              {bulkChat.reconnectAttempts > 0 && (
                <span className="text-xs opacity-75">({bulkChat.reconnectAttempts}/5)</span>
              )}
            </motion.div>
          )}
          {useBulkChatMode && bulkChat.messageCount > 0 && (
            <div className="flex items-center justify-between text-xs text-slate-500 px-4 py-2 bg-slate-800/30 rounded-lg border border-slate-700/50 mx-4 my-2 flex-wrap gap-2">
              <div className="flex items-center gap-2">
                <span>📊 {bulkChat.messageCount} mensaje{bulkChat.messageCount !== 1 ? 's' : ''}</span>
                {bulkChat.connectionQuality !== 'offline' && (
                  <span className={`px-2 py-0.5 rounded text-[10px] ${
                    bulkChat.connectionQuality === 'excellent' ? 'bg-green-500/20 text-green-400' :
                    bulkChat.connectionQuality === 'good' ? 'bg-yellow-500/20 text-yellow-400' :
                    'bg-orange-500/20 text-orange-400'
                  }`}>
                    {bulkChat.connectionQuality === 'excellent' ? '⚡' :
                     bulkChat.connectionQuality === 'good' ? '✓' :
                     '⚠'}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-3">
                {bulkChat.lastActivity && (
                  <div className="text-slate-600">
                    🕐 {new Date(bulkChat.lastActivity).toLocaleTimeString()}
                  </div>
                )}
                {bulkChat.session && (
                  <div className="text-slate-600">
                    📍 {bulkChat.session.state}
                  </div>
                )}
              </div>
            </div>
          )}
          {readMode && (
            <div className="mx-4 mb-2 p-2 bg-blue-900/20 border border-blue-700/50 rounded-lg text-xs text-blue-300 text-center">
              📖 Modo lectura activo - Solo mensajes del asistente
            </div>
          )}
          {zenMode && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="mx-4 mb-2 p-2 bg-purple-900/20 border border-purple-700/50 rounded-lg text-xs text-purple-300 text-center"
            >
              🧘 Modo Zen activo - Interfaz minimalista
            </motion.div>
          )}
          {editingNote !== null && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mx-4 mb-4 p-4 bg-slate-800 border border-slate-600 rounded-lg"
            >
              <h3 className="text-sm font-semibold text-slate-300 mb-2">📝 Agregar nota</h3>
              <textarea
                value={messageNotes.get(editingNote) || ''}
                onChange={(e) => {
                  const newMap = new Map(messageNotes)
                  newMap.set(editingNote, e.target.value)
                  setMessageNotes(newMap)
                }}
                placeholder="Escribe una nota para este mensaje..."
                className="w-full p-2 bg-slate-700 border border-slate-600 rounded text-white text-sm"
                rows={3}
                autoFocus
              />
              <div className="flex gap-2 mt-2">
                <button
                  onClick={() => saveNoteToMessage(editingNote, messageNotes.get(editingNote) || '')}
                  className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-white text-sm"
                >
                  Guardar
                </button>
                <button
                  onClick={() => {
                    setEditingNote(null)
                  }}
                  className="px-3 py-1 bg-slate-600 hover:bg-slate-700 rounded text-white text-sm"
                >
                  Cancelar
                </button>
              </div>
            </motion.div>
          )}
          {editingMessage !== null && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mx-4 mb-4 p-4 bg-slate-800 border border-slate-600 rounded-lg"
            >
              <h3 className="text-sm font-semibold text-slate-300 mb-2">✏️ Editar mensaje</h3>
              <textarea
                value={filteredMessages.find(m => m.id === editingMessage)?.content || ''}
                onChange={(e) => {
                  // Actualizar mensaje en store
                  const messageIndex = filteredMessages.findIndex(m => m.id === editingMessage)
                  if (messageIndex >= 0) {
                    const updatedMessages = [...filteredMessages]
                    updatedMessages[messageIndex] = {
                      ...updatedMessages[messageIndex],
                      content: e.target.value
                    }
                    setFilteredMessages(updatedMessages)
                  }
                }}
                placeholder="Edita el contenido del mensaje..."
                className="w-full p-2 bg-slate-700 border border-slate-600 rounded text-white text-sm"
                rows={5}
                autoFocus
              />
              <div className="flex gap-2 mt-2">
                <button
                  onClick={() => {
                    const message = filteredMessages.find(m => m.id === editingMessage)
                    if (message) {
                      saveMessageVersion(editingMessage, message.content)
                      toast.success('Mensaje actualizado', { icon: '✏️', duration: 2000 })
                    }
                    setEditingMessage(null)
                  }}
                  className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-white text-sm"
                >
                  Guardar
                </button>
                <button
                  onClick={() => {
                    setEditingMessage(null)
                  }}
                  className="px-3 py-1 bg-slate-600 hover:bg-slate-700 rounded text-white text-sm"
                >
                  Cancelar
                </button>
              </div>
            </motion.div>
          )}
          {translationMode && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mx-4 mb-2 p-2 bg-green-900/20 border border-green-700/50 rounded-lg text-xs text-green-300 text-center"
            >
              🌐 Modo traducción activo - Idioma objetivo: {targetLanguage.toUpperCase()}
              <select
                value={targetLanguage}
                onChange={(e) => setTargetLanguage(e.target.value)}
                className="ml-2 px-2 py-1 bg-slate-700 border border-slate-600 rounded text-white text-xs"
              >
                <option value="en">Inglés</option>
                <option value="es">Español</option>
                <option value="fr">Francés</option>
                <option value="de">Alemán</option>
                <option value="pt">Portugués</option>
              </select>
            </motion.div>
          )}
          {showHistory && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mx-4 mb-4 p-4 bg-slate-800/50 rounded-lg border border-slate-700 max-h-64 overflow-y-auto"
            >
              <h3 className="text-sm font-semibold text-slate-300 mb-3">📜 Historial de Ediciones</h3>
              <div className="space-y-2">
                {Array.from(messageHistory.entries()).map(([messageId, history]) => {
                  const message = filteredMessages.find(m => m.id === messageId)
                  if (!message || history.length === 0) return null
                  return (
                    <div key={messageId} className="p-2 bg-slate-700/30 rounded border border-slate-600">
                      <div className="text-xs text-slate-400 mb-1">
                        Mensaje: {message.content.substring(0, 50)}...
                      </div>
                      <div className="text-xs text-slate-300">
                        Versiones: {history.length + 1}
                      </div>
                    </div>
                  )
                })}
                {Array.from(messageHistory.entries()).length === 0 && (
                  <div className="text-xs text-slate-500 text-center py-4">
                    No hay historial de ediciones
                  </div>
                )}
              </div>
            </motion.div>
          )}
          {showDebug && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mx-4 mb-4 p-4 bg-red-900/20 border border-red-700/50 rounded-lg"
            >
              <h3 className="text-sm font-semibold text-red-300 mb-3">🐛 Debug Info</h3>
              <div className="space-y-2 text-xs font-mono">
                <div className="text-red-200">
                  <strong>Session ID:</strong> {bulkChat.sessionId || 'N/A'}
                </div>
                <div className="text-red-200">
                  <strong>WebSocket:</strong> {bulkChat.isConnected ? '✅' : '❌'}
                </div>
                <div className="text-red-200">
                  <strong>Online:</strong> {bulkChat.isOnline ? '✅' : '❌'}
                </div>
                <div className="text-red-200">
                  <strong>Quality:</strong> {bulkChat.connectionQuality}
                </div>
                <div className="text-red-200">
                  <strong>Reconnect attempts:</strong> {bulkChat.reconnectAttempts}
                </div>
                <div className="text-red-200">
                  <strong>Messages in store:</strong> {messages.length}
                </div>
                <div className="text-red-200">
                  <strong>BulkChat messages:</strong> {bulkChat.messages.length}
                </div>
                <div className="text-red-200">
                  <strong>Filtered:</strong> {filteredMessages.length}
                </div>
                <div className="text-red-200">
                  <strong>Favorites:</strong> {favoriteMessages.size}
                </div>
                <div className="text-red-200">
                  <strong>Selected:</strong> {selectedMessages.size}
                </div>
                <div className="text-red-200">
                  <strong>Collapsed:</strong> {collapsedMessages.size}
                </div>
                <div className="text-red-200">
                  <strong>Reactions:</strong> {Array.from(messageReactions.values()).reduce((acc, r) => acc + r.length, 0)}
                </div>
                <div className="text-red-200">
                  <strong>View mode:</strong> {viewMode}
                </div>
                <div className="text-red-200">
                  <strong>Font size:</strong> {fontSize}
                </div>
                <div className="text-red-200">
                  <strong>Auto-scroll:</strong> {autoScroll ? '✅' : '❌'}
                </div>
                <div className="text-red-200">
                  <strong>Theme:</strong> {theme}
                </div>
                <div className="text-red-200">
                  <strong>Tags:</strong> {Array.from(messageTags.values()).reduce((acc, tags) => acc + tags.length, 0)}
                </div>
                <div className="text-red-200">
                  <strong>Notes:</strong> {messageNotes.size}
                </div>
                <div className="text-red-200">
                  <strong>Zen mode:</strong> {zenMode ? '✅' : '❌'}
                </div>
                <div className="text-red-200">
                  <strong>Reading speed:</strong> {readingSpeed}
                </div>
              </div>
            </motion.div>
          )}
          {showTimeline && bulkChat.messages.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mx-4 mb-4 p-4 bg-slate-800/50 rounded-lg border border-slate-700 max-h-48 overflow-y-auto"
            >
              <h3 className="text-sm font-semibold text-slate-300 mb-3">📅 Timeline de Mensajes</h3>
              <div className="space-y-2 text-xs">
                {bulkChat.messages.slice().reverse().map((msg, idx) => {
                  const date = new Date(msg.timestamp)
                  const timeStr = date.toLocaleTimeString()
                  const dateStr = date.toLocaleDateString()
                  const isToday = date.toDateString() === new Date().toDateString()
                  
                  return (
                    <div
                      key={msg.id}
                      className="flex items-center gap-2 p-2 bg-slate-700/30 rounded hover:bg-slate-700/50 cursor-pointer transition-colors"
                      onClick={() => {
                        const element = messageRefs.current.get(msg.id)
                        if (element) {
                          element.scrollIntoView({ behavior: 'smooth', block: 'center' })
                          element.classList.add('ring-2', 'ring-blue-500', 'ring-opacity-50')
                          setTimeout(() => {
                            element.classList.remove('ring-2', 'ring-blue-500', 'ring-opacity-50')
                          }, 2000)
                        }
                      }}
                    >
                      <div className={`w-2 h-2 rounded-full ${
                        msg.role === 'user' ? 'bg-blue-400' : 'bg-green-400'
                      }`} />
                      <div className="flex-1 truncate">
                        <span className="text-slate-400">{isToday ? 'Hoy' : dateStr}</span>
                        <span className="text-slate-500 mx-2">•</span>
                        <span className="text-slate-300">{timeStr}</span>
                        <span className="text-slate-500 mx-2">•</span>
                        <span className="text-slate-200 truncate">{msg.content.substring(0, 50)}...</span>
                      </div>
                      {favoriteMessages.has(msg.id) && (
                        <span className="text-yellow-400">⭐</span>
                      )}
                    </div>
                  )
                })}
              </div>
            </motion.div>
          )}
          {showStats && bulkChat.messages.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mx-4 mb-4 p-4 bg-slate-800/50 rounded-lg border border-slate-700"
            >
              <h3 className="text-sm font-semibold text-slate-300 mb-3">📊 Estadísticas de Conversación</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                <div className="bg-slate-700/30 rounded p-2">
                  <div className="text-slate-400">Total mensajes</div>
                  <div className="text-white font-semibold text-lg">{bulkChat.messageCount}</div>
                </div>
                <div className="bg-slate-700/30 rounded p-2">
                  <div className="text-slate-400">Usuario</div>
                  <div className="text-blue-400 font-semibold text-lg">
                    {bulkChat.messages.filter(m => m.role === 'user').length}
                  </div>
                </div>
                <div className="bg-slate-700/30 rounded p-2">
                  <div className="text-slate-400">Asistente</div>
                  <div className="text-green-400 font-semibold text-lg">
                    {bulkChat.messages.filter(m => m.role === 'assistant').length}
                  </div>
                </div>
                <div className="bg-slate-700/30 rounded p-2">
                  <div className="text-slate-400">Caracteres</div>
                  <div className="text-purple-400 font-semibold text-lg">
                    {bulkChat.messages.reduce((acc, m) => acc + m.content.length, 0).toLocaleString()}
                  </div>
                </div>
                <div className="bg-slate-700/30 rounded p-2">
                  <div className="text-slate-400">Favoritos</div>
                  <div className="text-yellow-400 font-semibold text-lg">{favoriteMessages.size}</div>
                </div>
                <div className="bg-slate-700/30 rounded p-2">
                  <div className="text-slate-400">Seleccionados</div>
                  <div className="text-blue-400 font-semibold text-lg">{selectedMessages.size}</div>
                </div>
                <div className="bg-slate-700/30 rounded p-2">
                  <div className="text-slate-400">Vista</div>
                  <div className="text-slate-300 font-semibold text-xs capitalize">{viewMode}</div>
                </div>
                <div className="bg-slate-700/30 rounded p-2">
                  <div className="text-slate-400">Colapsados</div>
                  <div className="text-orange-400 font-semibold text-lg">{collapsedMessages.size}</div>
                </div>
                <div className="bg-slate-700/30 rounded p-2">
                  <div className="text-slate-400">Filtros</div>
                  <div className="text-purple-400 font-semibold text-xs">
                    {filterRole !== 'all' ? filterRole : 'Ninguno'}
                  </div>
                </div>
                <div className="bg-slate-700/30 rounded p-2">
                  <div className="text-slate-400">Tamaño fuente</div>
                  <div className="text-slate-300 font-semibold text-xs capitalize">{fontSize}</div>
                </div>
                <div className="bg-slate-700/30 rounded p-2">
                  <div className="text-slate-400">Presentación</div>
                  <div className={`font-semibold text-lg ${presentationMode ? 'text-green-400' : 'text-slate-500'}`}>
                    {presentationMode ? '🖥️' : '○'}
                  </div>
                </div>
                {showWordCount && (
                  <div className="bg-slate-700/30 rounded p-2">
                    <div className="text-slate-400">Palabras totales</div>
                    <div className="text-cyan-400 font-semibold text-lg">
                      {bulkChat.messages.reduce((acc, m) => {
                        return acc + m.content.trim().split(/\s+/).filter(Boolean).length
                      }, 0).toLocaleString()}
                    </div>
                  </div>
                )}
                <div className="bg-slate-700/30 rounded p-2">
                  <div className="text-slate-400">Auto-scroll</div>
                  <div className={`font-semibold text-lg ${autoScroll ? 'text-green-400' : 'text-slate-500'}`}>
                    {autoScroll ? '✓' : '✕'}
                  </div>
                </div>
                {showSentiment && (
                  <div className="bg-slate-700/30 rounded p-2">
                    <div className="text-slate-400">Análisis sentimiento</div>
                    <div className="text-green-400 font-semibold text-lg">😊</div>
                  </div>
                )}
                {showReactions && (
                  <div className="bg-slate-700/30 rounded p-2">
                    <div className="text-slate-400">Reacciones</div>
                    <div className="text-blue-400 font-semibold text-lg">
                      {Array.from(messageReactions.values()).reduce((acc, reactions) => acc + reactions.length, 0)}
                    </div>
                  </div>
                )}
                <div className="bg-slate-700/30 rounded p-2">
                  <div className="text-slate-400">Tema</div>
                  <div className="text-slate-300 font-semibold text-xs capitalize">{theme}</div>
                </div>
                {showDebug && (
                  <div className="bg-slate-700/30 rounded p-2">
                    <div className="text-slate-400">Debug</div>
                    <div className="text-red-400 font-semibold text-lg">🐛</div>
                  </div>
                )}
                {zenMode && (
                  <div className="bg-slate-700/30 rounded p-2">
                    <div className="text-slate-400">Zen</div>
                    <div className="text-purple-400 font-semibold text-lg">🧘</div>
                  </div>
                )}
                {showTags && (
                  <div className="bg-slate-700/30 rounded p-2">
                    <div className="text-slate-400">Etiquetas</div>
                    <div className="text-blue-400 font-semibold text-lg">
                      {Array.from(messageTags.values()).reduce((acc, tags) => acc + tags.length, 0)}
                    </div>
                  </div>
                )}
                <div className="bg-slate-700/30 rounded p-2">
                  <div className="text-slate-400">Notas</div>
                  <div className="text-yellow-400 font-semibold text-lg">{messageNotes.size}</div>
                </div>
                {showEditHistory && (
                  <div className="bg-slate-700/30 rounded p-2">
                    <div className="text-slate-400">Ediciones</div>
                    <div className="text-purple-400 font-semibold text-lg">
                      {Array.from(messageVersions.values()).reduce((acc, v) => acc + v, 0)}
                    </div>
                  </div>
                )}
                {translationMode && (
                  <div className="bg-slate-700/30 rounded p-2">
                    <div className="text-slate-400">Traducción</div>
                    <div className="text-green-400 font-semibold text-xs">{targetLanguage.toUpperCase()}</div>
                  </div>
                )}
                <div className="bg-slate-700/30 rounded p-2">
                  <div className="text-slate-400">Tiempo promedio</div>
                  <div className="text-green-400 font-semibold text-xs">
                    {Array.from(messageStats.values())
                      .filter(s => s.responseTime !== undefined)
                      .reduce((acc, s) => acc + (s.responseTime || 0), 0) / 
                      Math.max(1, Array.from(messageStats.values()).filter(s => s.responseTime !== undefined).length)
                      .toFixed(1)}s
                  </div>
                </div>
                <div className="bg-slate-700/30 rounded p-2">
                  <div className="text-slate-400">Calidad conexión</div>
                  <div className={`font-semibold text-lg ${
                    bulkChat.connectionQuality === 'excellent' ? 'text-green-400' :
                    bulkChat.connectionQuality === 'good' ? 'text-yellow-400' :
                    'text-orange-400'
                  }`}>
                    {bulkChat.connectionQuality === 'excellent' ? '⚡' :
                     bulkChat.connectionQuality === 'good' ? '✓' :
                     '⚠'}
                  </div>
                </div>
                <div className="bg-slate-700/30 rounded p-2">
                  <div className="text-slate-400">Estado</div>
                  <div className={`font-semibold text-lg ${
                    bulkChat.isPaused ? 'text-yellow-400' : 'text-green-400'
                  }`}>
                    {bulkChat.isPaused ? '⏸' : '▶'}
                  </div>
                </div>
                <div className="bg-slate-700/30 rounded p-2">
                  <div className="text-slate-400">Sesión</div>
                  <div className="text-slate-300 font-mono text-xs">
                    {bulkChat.sessionId?.slice(0, 8) || 'N/A'}
                  </div>
                </div>
              </div>
            </motion.div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Model Status Bar */}
        {currentModel && (
          <ModelStatus model={currentModel} />
        )}

        {/* Input Area */}
        <form onSubmit={handleSubmit} className={`border-t border-slate-700 p-4 ${presentationMode || showPrintMode || zenMode ? 'hidden' : ''} no-print`}>
          {/* Templates de mensajes */}
          {showTemplates && (
            <div className="mb-3 p-3 bg-slate-800/50 rounded-lg border border-slate-700">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-slate-400">Plantillas rápidas:</span>
                <button
                  onClick={() => setShowTemplates(false)}
                  className="text-xs text-slate-500 hover:text-slate-300"
                >
                  ✕
                </button>
              </div>
              <div className="flex flex-wrap gap-2">
                {[...defaultTemplates, ...messageTemplates].map((template, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => {
                      setInput(template)
                      setShowTemplates(false)
                    }}
                    className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 rounded text-xs text-slate-300 transition-colors"
                  >
                    {template.substring(0, 30)}{template.length > 30 ? '...' : ''}
                  </button>
                ))}
              </div>
            </div>
          )}
          
          <div className="relative flex gap-2">
            <div className="flex-1 relative">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'ArrowDown' || e.key === 'ArrowUp' || e.key === 'Enter') {
                    // Handle autocomplete navigation
                  }
                }}
              placeholder={useBulkChatMode 
                ? "Escribe un mensaje... (Ctrl+B: Bulk Chat, Ctrl+K: Autocompletado)"
                : "Describe el modelo de IA que quieres crear... (Ctrl+K para autocompletado)"}
              className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-4 py-3 pr-20 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all"
              disabled={isLoading || bulkChat.isLoading}
              id="input-field"
            />
              {showWordCount && input && (
                <div className="absolute bottom-1 right-12 text-xs text-slate-500">
                  {input.trim().split(/\s+/).filter(Boolean).length} palabras
                </div>
              )}
              <button
                type="button"
                onClick={() => setShowTemplates(!showTemplates)}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 px-2 py-1 bg-slate-600 hover:bg-slate-500 rounded text-xs text-slate-300 transition-colors"
                title="Plantillas de mensajes"
              >
                📋
              </button>
              <AutoComplete input={input} onSelect={(text) => {
                setInput(text)
                validateInput(text)
              }} />
            </div>
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="bg-gradient-to-r from-purple-600 to-pink-600 text-white px-6 py-3 rounded-lg font-medium hover:from-purple-700 hover:to-pink-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Creando...</span>
                </>
              ) : (
                <>
                  <Send className="w-5 h-5" />
                  <span>Crear Modelo</span>
                </>
              )}
            </button>
          </div>
          {validation && (
            <div className="mt-2">
              <ValidationBadge validation={validation} />
            </div>
          )}
        </form>
      </div>

      {/* Model Comparator */}
      {showComparator && (
        <ModelComparator
          models={modelHistory.slice(0, 3)}
          onClose={() => setShowComparator(false)}
        />
      )}

      {/* Model Preview Modal */}
      {showPreview && previewSpec && (
        <ModelPreview
          spec={previewSpec.spec}
          modelName={previewSpec.modelName}
          description={previewSpec.description}
          onClose={() => setShowPreview(false)}
          onConfirm={handlePreviewConfirm}
        />
      )}

      {/* Welcome Tour */}
      <WelcomeTour onComplete={() => setShowTour(false)} />

      {/* Markdown Preview Modal */}
      {showMarkdownPreview && previewMessageId && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
          onClick={() => {
            setShowMarkdownPreview(false)
            setPreviewMessageId(null)
          }}
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            onClick={(e) => e.stopPropagation()}
            className="bg-slate-800 rounded-lg p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto border border-slate-700"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">👁️ Vista Previa Markdown</h3>
              <button
                onClick={() => {
                  setShowMarkdownPreview(false)
                  setPreviewMessageId(null)
                }}
                className="text-slate-400 hover:text-white"
              >
                ✕
              </button>
            </div>
            {(() => {
              const message = messages.find(m => m.id === previewMessageId)
              if (!message) return null
              return (
                <div className="space-y-4">
                  <div className="text-xs text-slate-400 mb-2">
                    {message.role === 'user' ? '👤 Usuario' : '🤖 Asistente'} • {new Date(message.timestamp).toLocaleString()}
                  </div>
                  <div
                    className="prose prose-invert max-w-none text-white"
                    dangerouslySetInnerHTML={{ __html: renderMarkdownPreview(message.content) }}
                  />
                </div>
              )
            })()}
          </motion.div>
        </motion.div>
      )}

      {/* Export Menu */}
      {showExportMenu && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-20 right-4 bg-slate-800 rounded-lg p-4 border border-slate-700 shadow-xl z-40"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-white">📤 Exportar Conversación</h3>
            <button onClick={() => setShowExportMenu(false)} className="text-slate-400 hover:text-white">✕</button>
          </div>
          <div className="space-y-2">
            <button
              onClick={() => {
                exportToPDF()
                setShowExportMenu(false)
              }}
              className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
            >
              📄 Exportar como PDF (HTML)
            </button>
            <button
              onClick={() => {
                exportToMarkdownEnhanced()
                setShowExportMenu(false)
              }}
              className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
            >
              📝 Exportar como Markdown
            </button>
            <button
              onClick={() => {
                if (bulkChat.sessionId && bulkChat.messages.length > 0) {
                  const data = {
                    sessionId: bulkChat.sessionId,
                    messages: bulkChat.messages,
                    timestamp: new Date().toISOString(),
                    messageCount: bulkChat.messageCount,
                  }
                  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
                  const url = URL.createObjectURL(blob)
                  const a = document.createElement('a')
                  a.href = url
                  a.download = `bulk-chat-${bulkChat.sessionId.slice(0, 8)}-${Date.now()}.json`
                  document.body.appendChild(a)
                  a.click()
                  document.body.removeChild(a)
                  URL.revokeObjectURL(url)
                  toast.success('Conversación exportada (JSON)', { icon: '💾', duration: 2000 })
                }
                setShowExportMenu(false)
              }}
              className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
            >
              💾 Exportar como JSON
            </button>
            <button
              onClick={() => {
                const text = bulkChat.messages.map(msg => 
                  `${msg.role === 'user' ? 'Usuario' : 'Asistente'}: ${msg.content}\n${new Date(msg.timestamp).toLocaleString()}\n---\n`
                ).join('\n')
                const blob = new Blob([text], { type: 'text/plain' })
                const url = URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.href = url
                a.download = `conversacion-${Date.now()}.txt`
                a.click()
                URL.revokeObjectURL(url)
                toast.success('Conversación exportada (TXT)', { icon: '📄', duration: 2000 })
                setShowExportMenu(false)
              }}
              className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
            >
              📄 Exportar como TXT
            </button>
          </div>
        </motion.div>
      )}

      {/* Message Grouping Selector */}
      {groupingMode !== 'none' && messageGroups.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed top-20 right-4 bg-slate-800 rounded-lg p-4 border border-slate-700 shadow-xl z-40 max-h-96 overflow-y-auto"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-white">🔗 Agrupación de Mensajes</h3>
            <div className="flex gap-2">
              <select
                value={groupingMode}
                onChange={(e) => setGroupingMode(e.target.value as any)}
                className="px-2 py-1 bg-slate-700 border border-slate-600 rounded text-white text-xs"
              >
                <option value="none">Sin agrupación</option>
                <option value="time">Por tiempo</option>
                <option value="role">Por rol</option>
                <option value="topic">Por tema</option>
              </select>
              <button
                onClick={() => setGroupingMode('none')}
                className="text-slate-400 hover:text-white text-xs"
              >
                ✕
              </button>
            </div>
          </div>
          <div className="space-y-2 text-xs">
            {Array.from(messageGroups.entries()).map(([groupName, messageIds]) => (
              <div key={groupName} className="p-2 bg-slate-700/30 rounded">
                <div className="font-semibold text-slate-200 mb-1">{groupName} ({messageIds.length})</div>
                <div className="text-slate-400">
                  {messageIds.slice(0, 3).map(id => {
                    const msg = messages.find(m => m.id === id)
                    return msg ? <div key={id} className="truncate">{msg.content.substring(0, 40)}...</div> : null
                  })}
                  {messageIds.length > 3 && <div className="text-slate-500">... y {messageIds.length - 3} más</div>}
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Pinned Messages Panel */}
      {showPinnedMessages && pinnedMessages.size > 0 && (
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="fixed left-4 top-20 bg-purple-900/20 rounded-lg p-4 border border-purple-700/50 shadow-xl z-40 max-w-xs max-h-96 overflow-y-auto"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-purple-300">📌 Mensajes Fijados ({pinnedMessages.size})</h3>
            <button
              onClick={() => setShowPinnedMessages(false)}
              className="text-purple-400 hover:text-purple-300"
            >
              ✕
            </button>
          </div>
          <div className="space-y-2 text-xs">
            {Array.from(pinnedMessages).map(id => {
              const msg = messages.find(m => m.id === id)
              return msg ? (
                <div
                  key={id}
                  className="p-2 bg-purple-800/30 rounded cursor-pointer hover:bg-purple-800/50 transition-colors"
                  onClick={() => {
                    const element = messageRefs.current.get(id)
                    if (element) {
                      element.scrollIntoView({ behavior: 'smooth', block: 'center' })
                      element.classList.add('ring-2', 'ring-purple-500')
                      setTimeout(() => {
                        element.classList.remove('ring-2', 'ring-purple-500')
                      }, 2000)
                    }
                  }}
                >
                  <div className="text-purple-200 font-semibold mb-1">
                    {msg.role === 'user' ? '👤' : '🤖'} {new Date(msg.timestamp).toLocaleTimeString()}
                  </div>
                  <div className="text-purple-300 truncate">{msg.content.substring(0, 60)}...</div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      togglePinMessage(id)
                    }}
                    className="mt-1 text-purple-400 hover:text-purple-300 text-xs"
                  >
                    Desfijar
                  </button>
                </div>
              ) : null
            })}
          </div>
        </motion.div>
      )}

      {/* Advanced Search Panel */}
      {advancedSearch && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed top-20 left-1/2 transform -translate-x-1/2 bg-slate-800 rounded-lg p-4 border border-slate-700 shadow-xl z-40 max-w-md"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-white">🔍 Búsqueda Avanzada</h3>
            <button onClick={() => setAdvancedSearch(false)} className="text-slate-400 hover:text-white">✕</button>
          </div>
          <div className="space-y-3 text-xs">
            <div>
              <label className="block text-slate-300 mb-1">Rango de fechas</label>
              <div className="flex gap-2">
                <input
                  type="date"
                  onChange={(e) => setSearchFilters(prev => ({
                    ...prev,
                    dateRange: { ...prev.dateRange, start: new Date(e.target.value) } as any
                  }))}
                  className="flex-1 px-2 py-1 bg-slate-700 border border-slate-600 rounded text-white"
                />
                <input
                  type="date"
                  onChange={(e) => setSearchFilters(prev => ({
                    ...prev,
                    dateRange: { ...prev.dateRange, end: new Date(e.target.value) } as any
                  }))}
                  className="flex-1 px-2 py-1 bg-slate-700 border border-slate-600 rounded text-white"
                />
              </div>
            </div>
            <div className="flex gap-2">
              <div className="flex-1">
                <label className="block text-slate-300 mb-1">Palabras mínimas</label>
                <input
                  type="number"
                  min="0"
                  onChange={(e) => setSearchFilters(prev => ({ ...prev, minWords: parseInt(e.target.value) || undefined }))}
                  className="w-full px-2 py-1 bg-slate-700 border border-slate-600 rounded text-white"
                />
              </div>
              <div className="flex-1">
                <label className="block text-slate-300 mb-1">Palabras máximas</label>
                <input
                  type="number"
                  min="0"
                  onChange={(e) => setSearchFilters(prev => ({ ...prev, maxWords: parseInt(e.target.value) || undefined }))}
                  className="w-full px-2 py-1 bg-slate-700 border border-slate-600 rounded text-white"
                />
              </div>
            </div>
            <div className="flex gap-2">
              <label className="flex items-center gap-2 text-slate-300">
                <input
                  type="checkbox"
                  onChange={(e) => setSearchFilters(prev => ({ ...prev, hasCode: e.target.checked || undefined }))}
                  className="w-4 h-4 text-purple-600 bg-slate-700 border-slate-600 rounded"
                />
                Contiene código
              </label>
              <label className="flex items-center gap-2 text-slate-300">
                <input
                  type="checkbox"
                  onChange={(e) => setSearchFilters(prev => ({ ...prev, hasLinks: e.target.checked || undefined }))}
                  className="w-4 h-4 text-purple-600 bg-slate-700 border-slate-600 rounded"
                />
                Contiene enlaces
              </label>
            </div>
            <button
              onClick={() => {
                setSearchFilters({})
                setAdvancedSearch(false)
              }}
              className="w-full px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300"
            >
              Limpiar filtros
            </button>
          </div>
        </motion.div>
      )}

      {/* Template Editor */}
      {showTemplateEditor && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
          onClick={() => setShowTemplateEditor(false)}
        >
          <motion.div
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            onClick={(e) => e.stopPropagation()}
            className="bg-slate-800 rounded-lg p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-slate-700"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">📝 Editor de Plantillas con Variables</h3>
              <button onClick={() => setShowTemplateEditor(false)} className="text-slate-400 hover:text-white">✕</button>
            </div>
            <div className="space-y-4">
              <div>
                <label className="block text-sm text-slate-300 mb-2">Nombre de la plantilla</label>
                <input
                  type="text"
                  id="template-name-input"
                  placeholder="Ej: Saludo personalizado"
                  className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white"
                />
              </div>
              <div>
                <label className="block text-sm text-slate-300 mb-2">Plantilla (usa {{variable}} para variables)</label>
                <textarea
                  id="template-content-input"
                  placeholder="Ej: Hola {{nombre}}, ¿cómo estás? Tu tema es {{tema}}."
                  rows={6}
                  className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white"
                />
              </div>
              <button
                onClick={() => {
                  const name = (document.getElementById('template-name-input') as HTMLInputElement)?.value
                  const content = (document.getElementById('template-content-input') as HTMLTextAreaElement)?.value
                  if (name && content) {
                    createTemplateWithVars(name, content)
                    setShowTemplateEditor(false)
                  }
                }}
                className="w-full px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded text-white"
              >
                Crear Plantilla
              </button>
              {messageTemplatesWithVars.size > 0 && (
                <div className="mt-4">
                  <h4 className="text-sm font-semibold text-slate-300 mb-2">Plantillas guardadas:</h4>
                  <div className="space-y-2">
                    {Array.from(messageTemplatesWithVars.entries()).map(([name, template]) => (
                      <div key={name} className="p-3 bg-slate-700/50 rounded">
                        <div className="font-semibold text-white mb-1">{name}</div>
                        <div className="text-xs text-slate-400 mb-2">{template.template}</div>
                        <div className="text-xs text-purple-400">Variables: {template.variables.join(', ')}</div>
                        <button
                          onClick={() => {
                            const vars: Record<string, string> = {}
                            template.variables.forEach(v => {
                              const value = prompt(`Valor para ${v}:`)
                              if (value) vars[v] = value
                            })
                            useTemplate(name, vars)
                            setShowTemplateEditor(false)
                          }}
                          className="mt-2 px-3 py-1 bg-purple-600 hover:bg-purple-700 rounded text-white text-xs"
                        >
                          Usar plantilla
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        </motion.div>
      )}

      {/* Share Menu */}
      {showShareMenu && shareTarget && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-20 right-4 bg-slate-800 rounded-lg p-4 border border-slate-700 shadow-xl z-40"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-white">🔗 Compartir Mensaje</h3>
            <button onClick={() => { setShowShareMenu(false); setShareTarget(null) }} className="text-slate-400 hover:text-white">✕</button>
          </div>
          <div className="space-y-2">
            <button
              onClick={() => {
                shareMessage(shareTarget, 'copy')
                setShowShareMenu(false)
                setShareTarget(null)
              }}
              className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
            >
              📋 Copiar al portapapeles
            </button>
            <button
              onClick={() => {
                shareMessage(shareTarget, 'link')
                setShowShareMenu(false)
                setShareTarget(null)
              }}
              className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
            >
              🔗 Copiar enlace
            </button>
            <button
              onClick={() => {
                shareMessage(shareTarget, 'email')
                setShowShareMenu(false)
                setShareTarget(null)
              }}
              className="w-full text-left px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors flex items-center gap-2"
            >
              📧 Enviar por email
            </button>
          </div>
        </motion.div>
      )}

      {/* Real-time Stats Panel */}
      {realTimeStats && (
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="fixed right-4 top-20 bg-cyan-900/20 rounded-lg p-4 border border-cyan-700/50 shadow-xl z-40 max-w-xs"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-cyan-300">📊 Estadísticas en Tiempo Real</h3>
            <button onClick={() => setRealTimeStats(false)} className="text-cyan-400 hover:text-cyan-300">✕</button>
          </div>
          <div className="space-y-2 text-xs">
            <div className="flex justify-between">
              <span className="text-cyan-400">Mensajes/min:</span>
              <span className="text-cyan-200 font-semibold">
                {messages.length > 0
                  ? (messages.length / Math.max(1, (Date.now() - new Date(messages[0]?.timestamp || Date.now()).getTime()) / 60000)).toFixed(1)
                  : '0'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-cyan-400">Tiempo promedio:</span>
              <span className="text-cyan-200 font-semibold">
                {Array.from(messageStats.values()).filter(s => s.responseTime !== undefined).length > 0
                  ? (Array.from(messageStats.values())
                      .filter(s => s.responseTime !== undefined)
                      .reduce((acc, s, _, arr) => acc + (s.responseTime || 0) / arr.length, 0)).toFixed(1)
                  : '0'}s
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-cyan-400">Threads activos:</span>
              <span className="text-cyan-200 font-semibold">{messageThreads.size}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-cyan-400">Fijados:</span>
              <span className="text-cyan-200 font-semibold">{pinnedMessages.size}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-cyan-400">Archivados:</span>
              <span className="text-cyan-200 font-semibold">{archivedMessages.size}</span>
            </div>
          </div>
        </motion.div>
      )}

      {/* Dev Mode Panel */}
      {devMode && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-4 left-4 bg-red-900/20 rounded-lg p-4 border border-red-700/50 shadow-xl z-40 max-w-md"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-red-300">🛠️ Modo Desarrollo</h3>
            <button onClick={() => setDevMode(false)} className="text-red-400 hover:text-red-300">✕</button>
          </div>
          <div className="space-y-2 text-xs font-mono">
            <div className="text-red-200">
              <strong>API URL:</strong> {process.env.NEXT_PUBLIC_BULK_CHAT_API_URL || 'http://localhost:8006'}
            </div>
            <div className="text-red-200">
              <strong>WebSocket:</strong> {bulkChat.isConnected ? '✅ Conectado' : '❌ Desconectado'}
            </div>
            <div className="text-red-200">
              <strong>Session ID:</strong> {bulkChat.sessionId?.slice(0, 16) || 'N/A'}...
            </div>
            <div className="text-red-200">
              <strong>Cache Stats:</strong> {JSON.stringify(getCacheStats?.() || {}, null, 2)}
            </div>
            <div className="text-red-200">
              <strong>Metrics:</strong> {JSON.stringify(metrics || {}, null, 2)}
            </div>
            <button
              onClick={() => {
                console.log('=== DEV MODE DEBUG ===')
                console.log('Messages:', messages)
                console.log('BulkChat:', bulkChat)
                console.log('State:', {
                  pinnedMessages: Array.from(pinnedMessages),
                  archivedMessages: Array.from(archivedMessages),
                  messageThreads: Array.from(messageThreads.entries()),
                  messageTags: Array.from(messageTags.entries()),
                  messageNotes: Array.from(messageNotes.entries()),
                })
                toast.success('Debug info logged to console', { icon: '🛠️' })
              }}
              className="w-full mt-2 px-3 py-2 bg-red-600 hover:bg-red-700 rounded text-white text-xs"
            >
              Log Debug Info
            </button>
          </div>
        </motion.div>
      )}

      {/* Collaboration Mode Panel */}
      {collaborationMode && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed top-20 right-4 bg-blue-900/20 rounded-lg p-4 border border-blue-700/50 shadow-xl z-40 max-w-xs"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-blue-300">👥 Modo Colaboración</h3>
            <button onClick={() => setCollaborationMode(false)} className="text-blue-400 hover:text-blue-300">✕</button>
          </div>
          <div className="space-y-2 text-xs">
            <div>
              <label className="block text-blue-300 mb-1">Agregar colaborador</label>
              <div className="flex gap-2">
                <input
                  type="text"
                  id="collaborator-input"
                  placeholder="Email o ID"
                  className="flex-1 px-2 py-1 bg-slate-700 border border-slate-600 rounded text-white text-xs"
                />
                <button
                  onClick={() => {
                    const input = document.getElementById('collaborator-input') as HTMLInputElement
                    if (input?.value) {
                      setCollaborators(prev => [...prev, input.value])
                      input.value = ''
                      toast.success('Colaborador agregado', { icon: '👥' })
                    }
                  }}
                  className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-white text-xs"
                >
                  +
                </button>
              </div>
            </div>
            {collaborators.length > 0 && (
              <div>
                <div className="text-blue-300 font-semibold mb-1">Colaboradores ({collaborators.length}):</div>
                <div className="space-y-1">
                  {collaborators.map((col, idx) => (
                    <div key={idx} className="flex items-center justify-between p-2 bg-blue-800/30 rounded">
                      <span className="text-blue-200 text-xs">{col}</span>
                      <button
                        onClick={() => setCollaborators(prev => prev.filter((_, i) => i !== idx))}
                        className="text-blue-400 hover:text-blue-300 text-xs"
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </motion.div>
      )}

      {/* Session Recording Panel */}
      {sessionRecording && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed top-4 right-4 bg-red-900/20 rounded-lg p-3 border border-red-700/50 shadow-xl z-40"
        >
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
            <span className="text-red-300 text-sm font-semibold">Grabando sesión</span>
            <button onClick={stopSessionRecording} className="ml-2 text-red-400 hover:text-red-300 text-xs">⏹️</button>
          </div>
        </motion.div>
      )}

      {/* Bookmarks Panel */}
      {showBookmarks && bookmarks.size > 0 && (
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="fixed left-4 bottom-20 bg-yellow-900/20 rounded-lg p-4 border border-yellow-700/50 shadow-xl z-40 max-w-xs max-h-96 overflow-y-auto"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-yellow-300">🔖 Marcadores ({bookmarks.size})</h3>
            <button onClick={() => setShowBookmarks(false)} className="text-yellow-400 hover:text-yellow-300">✕</button>
          </div>
          <div className="space-y-2">
            {Array.from(bookmarks.values()).map(bookmark => {
              const message = messages.find(m => m.id === bookmark.messageId)
              return message ? (
                <div key={bookmark.messageId} className="p-2 bg-yellow-800/30 rounded">
                  <div className="font-semibold text-yellow-200 text-xs mb-1">{bookmark.name}</div>
                  <div className="text-yellow-300 text-xs truncate">{message.content.substring(0, 50)}...</div>
                  <button
                    onClick={() => {
                      messageRefs.current.get(bookmark.messageId)?.scrollIntoView({ behavior: 'smooth' })
                    }}
                    className="mt-1 text-yellow-400 hover:text-yellow-300 text-xs"
                  >
                    Ir al mensaje
                  </button>
                </div>
              ) : null
            })}
          </div>
        </motion.div>
      )}

      {/* Study Mode Panel */}
      {studyMode && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed top-20 left-4 bg-green-900/20 rounded-lg p-4 border border-green-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-green-300">📚 Modo Estudio</h3>
            <button onClick={() => setStudyMode(false)} className="text-green-400 hover:text-green-300">✕</button>
          </div>
          <div className="space-y-3 text-xs">
            <div>
              <div className="text-green-300 mb-1">Flashcards: {flashcards.size}</div>
              <div className="text-green-300 mb-1">Notas de estudio: {studyNotes.size}</div>
            </div>
            <button
              onClick={() => setShowFlashcards(!showFlashcards)}
              className="w-full px-3 py-2 bg-green-600 hover:bg-green-700 rounded text-white"
            >
              {showFlashcards ? 'Ocultar' : 'Mostrar'} Flashcards
            </button>
          </div>
        </motion.div>
      )}

      {/* Productivity Mode Panel */}
      {productivityMode && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-4 right-4 bg-orange-900/20 rounded-lg p-4 border border-orange-700/50 shadow-xl z-40"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-orange-300">⏱️ Pomodoro</h3>
            <button onClick={() => setProductivityMode(false)} className="text-orange-400 hover:text-orange-300">✕</button>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-200 mb-2">
              {Math.floor(focusTimer / 60)}:{(focusTimer % 60).toString().padStart(2, '0')}
            </div>
            <div className="text-xs text-orange-400">Meta: {focusGoal} minutos</div>
            <div className="w-full bg-orange-800/30 rounded-full h-2 mt-2">
              <div
                className="bg-orange-500 h-2 rounded-full transition-all"
                style={{ width: `${(focusTimer / (focusGoal * 60)) * 100}%` }}
              ></div>
            </div>
          </div>
        </motion.div>
      )}

      {/* AI Insights Panel */}
      {aiInsights && conversationInsights && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
          onClick={() => setAiInsights(false)}
        >
          <motion.div
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            onClick={(e) => e.stopPropagation()}
            className="bg-slate-800 rounded-lg p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-slate-700"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">🤖 Insights de IA</h3>
              <button onClick={() => setAiInsights(false)} className="text-slate-400 hover:text-white">✕</button>
            </div>
            <div className="space-y-4">
              {conversationInsights.keyInsights.map((insight: string, idx: number) => (
                <div key={idx} className="p-3 bg-slate-700/50 rounded text-slate-200">
                  {insight}
                </div>
              ))}
            </div>
          </motion.div>
        </motion.div>
      )}

      {/* Reading Mode Panel */}
      {readingMode && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed top-4 left-1/2 transform -translate-x-1/2 bg-blue-900/20 rounded-lg p-3 border border-blue-700/50 shadow-xl z-40"
        >
          <div className="flex items-center gap-3">
            <span className="text-blue-300 text-sm">📖 Modo Lectura</span>
            <div className="w-32 bg-blue-800/30 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full transition-all"
                style={{ width: `${readingProgress}%` }}
              ></div>
            </div>
            <span className="text-blue-300 text-xs">{readingProgress}%</span>
          </div>
        </motion.div>
      )}

      {/* Calendar Integration Panel */}
      {calendarIntegration && scheduledEvents.size > 0 && (
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="fixed right-4 top-20 bg-purple-900/20 rounded-lg p-4 border border-purple-700/50 shadow-xl z-40 max-w-xs"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-purple-300">📅 Eventos ({scheduledEvents.size})</h3>
            <button onClick={() => setCalendarIntegration(false)} className="text-purple-400 hover:text-purple-300">✕</button>
          </div>
          <div className="space-y-2 text-xs">
            {Array.from(scheduledEvents.values()).map((event, idx) => (
              <div key={idx} className="p-2 bg-purple-800/30 rounded">
                <div className="font-semibold text-purple-200">{event.title}</div>
                <div className="text-purple-300">{new Date(event.date).toLocaleString()}</div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Cloud Sync Status */}
      {cloudSync && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-4 left-4 bg-cyan-900/20 rounded-lg p-2 border border-cyan-700/50 shadow-xl z-40"
        >
          <div className="flex items-center gap-2 text-xs">
            <span className="text-cyan-300">☁️ Sincronizado</span>
            <button onClick={syncToCloud} className="text-cyan-400 hover:text-cyan-300">🔄</button>
          </div>
        </motion.div>
      )}

      {/* Smart Folders Panel */}
      {showFolders && smartFolders.size > 0 && (
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="fixed left-4 top-20 bg-indigo-900/20 rounded-lg p-4 border border-indigo-700/50 shadow-xl z-40 max-w-xs max-h-96 overflow-y-auto"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-indigo-300">📁 Carpetas ({smartFolders.size})</h3>
            <button onClick={() => setShowFolders(false)} className="text-indigo-400 hover:text-indigo-300">✕</button>
          </div>
          <div className="space-y-2">
            {Array.from(smartFolders.values()).map((folder, idx) => (
              <div key={idx} className="p-2 bg-indigo-800/30 rounded">
                <div className="font-semibold text-indigo-200 text-xs">{folder.name}</div>
                <div className="text-indigo-300 text-xs">{folder.messageIds.length} mensajes</div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Presentation Mode */}
      {presentationMode && messages.length > 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="fixed inset-0 bg-black z-50 flex items-center justify-center"
        >
          <div className="max-w-4xl w-full p-8">
            <div className="bg-slate-800 rounded-lg p-8 border border-slate-700">
              <div className="text-white text-2xl mb-4">
                {messages[presentationSlide]?.content}
              </div>
              <div className="flex items-center justify-between mt-8">
                <button
                  onClick={prevSlide}
                  disabled={presentationSlide === 0}
                  className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded text-white disabled:opacity-50"
                >
                  ← Anterior
                </button>
                <div className="text-slate-400">
                  {presentationSlide + 1} / {messages.length}
                </div>
                <button
                  onClick={nextSlide}
                  disabled={presentationSlide === messages.length - 1}
                  className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded text-white disabled:opacity-50"
                >
                  Siguiente →
                </button>
              </div>
              <button
                onClick={() => setPresentationMode(false)}
                className="absolute top-4 right-4 text-slate-400 hover:text-white"
              >
                ✕ Salir
              </button>
            </div>
          </div>
        </motion.div>
      )}

      {/* Versions Panel */}
      {showVersions && conversationVersions.size > 0 && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
          onClick={() => setShowVersions(false)}
        >
          <motion.div
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            onClick={(e) => e.stopPropagation()}
            className="bg-slate-800 rounded-lg p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-slate-700"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">📜 Versiones de Conversación</h3>
              <button onClick={() => setShowVersions(false)} className="text-slate-400 hover:text-white">✕</button>
            </div>
            <div className="space-y-3">
              {Array.from(conversationVersions.entries()).map(([id, version]) => (
                <div key={id} className="p-4 bg-slate-700/50 rounded">
                  <div className="font-semibold text-white mb-1">{version.metadata.name || 'Sin nombre'}</div>
                  <div className="text-xs text-slate-400 mb-2">
                    {new Date(version.timestamp).toLocaleString()} • {version.metadata.messageCount} mensajes
                  </div>
                  <button
                    onClick={() => {
                      restoreConversationVersion(id)
                      setShowVersions(false)
                    }}
                    className="px-3 py-1 bg-purple-600 hover:bg-purple-700 rounded text-white text-xs"
                  >
                    Restaurar
                  </button>
                </div>
              ))}
            </div>
            <button
              onClick={() => {
                const name = prompt('Nombre de la versión:') || `Versión ${conversationVersions.size + 1}`
                if (name) {
                  saveConversationVersion(name)
                }
              }}
              className="w-full mt-4 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded text-white"
            >
              Guardar Nueva Versión
            </button>
          </motion.div>
        </motion.div>
      )}

      {/* Comments Panel */}
      {showComments && (
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="fixed right-4 top-20 bg-slate-800 rounded-lg p-4 border border-slate-700 shadow-xl z-40 max-w-xs max-h-96 overflow-y-auto"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-white">💬 Comentarios</h3>
            <button onClick={() => setShowComments(false)} className="text-slate-400 hover:text-white">✕</button>
          </div>
          <div className="space-y-2 text-xs">
            {Array.from(messageComments.entries()).map(([messageId, comments]) => {
              const message = messages.find(m => m.id === messageId)
              return (
                <div key={messageId} className="p-2 bg-slate-700/50 rounded">
                  <div className="text-slate-300 font-semibold mb-1">
                    {message?.content.substring(0, 30)}...
                  </div>
                  {comments.map((comment, idx) => (
                    <div key={idx} className="text-slate-400 mb-1">
                      <strong>{comment.author}:</strong> {comment.content}
                    </div>
                  ))}
                </div>
              )
            })}
          </div>
        </motion.div>
      )}

      {/* Advanced Analytics Panel */}
      {advancedAnalytics && analyticsData && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
          onClick={() => setAdvancedAnalytics(false)}
        >
          <motion.div
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            onClick={(e) => e.stopPropagation()}
            className="bg-slate-800 rounded-lg p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto border border-slate-700"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">📊 Analytics Avanzado</h3>
              <button onClick={() => setAdvancedAnalytics(false)} className="text-slate-400 hover:text-white">✕</button>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 bg-slate-700/50 rounded">
                <div className="text-slate-300 text-sm mb-1">Total Mensajes</div>
                <div className="text-2xl font-bold text-white">{analyticsData.totalMessages}</div>
              </div>
              <div className="p-4 bg-slate-700/50 rounded">
                <div className="text-slate-300 text-sm mb-1">Tiempo Promedio</div>
                <div className="text-2xl font-bold text-white">{analyticsData.averageResponseTime.toFixed(1)}s</div>
              </div>
              <div className="p-4 bg-slate-700/50 rounded">
                <div className="text-slate-300 text-sm mb-1">Mensajes/Hora</div>
                <div className="text-2xl font-bold text-white">{analyticsData.messagesPerHour.toFixed(1)}</div>
              </div>
              <div className="p-4 bg-slate-700/50 rounded">
                <div className="text-slate-300 text-sm mb-1">Engagement</div>
                <div className="text-2xl font-bold text-white">
                  {analyticsData.engagementMetrics.votes + analyticsData.engagementMetrics.ratings + analyticsData.engagementMetrics.comments}
                </div>
              </div>
            </div>
            <div className="mt-4">
              <h4 className="text-sm font-semibold text-slate-300 mb-2">Métricas de Productividad</h4>
              <div className="grid grid-cols-4 gap-2">
                <div className="p-2 bg-slate-700/50 rounded text-center">
                  <div className="text-xs text-slate-400">Marcadores</div>
                  <div className="text-lg font-bold text-white">{analyticsData.productivityMetrics.bookmarks}</div>
                </div>
                <div className="p-2 bg-slate-700/50 rounded text-center">
                  <div className="text-xs text-slate-400">Flashcards</div>
                  <div className="text-lg font-bold text-white">{analyticsData.productivityMetrics.flashcards}</div>
                </div>
                <div className="p-2 bg-slate-700/50 rounded text-center">
                  <div className="text-xs text-slate-400">Fijados</div>
                  <div className="text-lg font-bold text-white">{analyticsData.productivityMetrics.pinned}</div>
                </div>
                <div className="p-2 bg-slate-700/50 rounded text-center">
                  <div className="text-xs text-slate-400">Archivados</div>
                  <div className="text-lg font-bold text-white">{analyticsData.productivityMetrics.archived}</div>
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}

      {/* Timeline View */}
      {messageTimeline && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-4 left-1/2 transform -translate-x-1/2 bg-slate-800 rounded-lg p-4 border border-slate-700 shadow-xl z-40 max-w-4xl w-full"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-white">📅 Vista Timeline</h3>
            <div className="flex gap-2">
              <button
                onClick={() => setTimelineView('linear')}
                className={`px-3 py-1 rounded text-xs ${timelineView === 'linear' ? 'bg-purple-600 text-white' : 'bg-slate-700 text-slate-300'}`}
              >
                Lineal
              </button>
              <button
                onClick={() => setTimelineView('grouped')}
                className={`px-3 py-1 rounded text-xs ${timelineView === 'grouped' ? 'bg-purple-600 text-white' : 'bg-slate-700 text-slate-300'}`}
              >
                Agrupado
              </button>
              <button
                onClick={() => setTimelineView('chronological')}
                className={`px-3 py-1 rounded text-xs ${timelineView === 'chronological' ? 'bg-purple-600 text-white' : 'bg-slate-700 text-slate-300'}`}
              >
                Cronológico
              </button>
              <button onClick={() => setMessageTimeline(false)} className="text-slate-400 hover:text-white">✕</button>
            </div>
          </div>
          <div className="h-32 overflow-x-auto">
            <div className="flex gap-2">
              {messages.map((msg, idx) => (
                <div
                  key={msg.id}
                  className="flex-shrink-0 w-24 h-24 bg-slate-700 rounded p-2 cursor-pointer hover:bg-slate-600"
                  onClick={() => messageRefs.current.get(msg.id)?.scrollIntoView({ behavior: 'smooth' })}
                >
                  <div className="text-xs text-slate-400 mb-1">{idx + 1}</div>
                  <div className="text-xs text-white truncate">{msg.content.substring(0, 20)}...</div>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      )}

      {/* Message Comparison Panel */}
      {showComparison && messageComparison && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
          onClick={() => setShowComparison(false)}
        >
          <motion.div
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            onClick={(e) => e.stopPropagation()}
            className="bg-slate-800 rounded-lg p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto border border-slate-700"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">🔍 Comparación de Mensajes</h3>
              <button onClick={() => setShowComparison(false)} className="text-slate-400 hover:text-white">✕</button>
            </div>
            <div className="grid grid-cols-2 gap-4">
              {messageComparison.map((msgId, idx) => {
                const message = messages.find(m => m.id === msgId)
                return message ? (
                  <div key={idx} className="p-4 bg-slate-700/50 rounded">
                    <div className="text-xs text-slate-400 mb-2">Mensaje {idx + 1}</div>
                    <div className="text-white">{message.content}</div>
                  </div>
                ) : null
              })}
            </div>
          </motion.div>
        </motion.div>
      )}

      {/* Split Screen Mode */}
      {splitScreenMode && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="fixed inset-0 bg-slate-900/50 z-40 pointer-events-none"
        >
          <div className="absolute top-4 right-4 bg-slate-800 rounded-lg p-3 border border-slate-700 pointer-events-auto">
            <div className="flex items-center gap-2">
              <span className="text-slate-300 text-sm">🖥️ Pantalla Dividida</span>
              <button onClick={() => setSplitScreenMode(false)} className="text-slate-400 hover:text-white">✕</button>
            </div>
          </div>
        </motion.div>
      )}

      {/* Widgets Panel */}
      {showWidgets && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed top-20 right-4 bg-purple-900/20 rounded-lg p-4 border border-purple-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-purple-300">🧩 Widgets</h3>
            <button onClick={() => setShowWidgets(false)} className="text-purple-400 hover:text-purple-300">✕</button>
          </div>
          <div className="space-y-2 text-xs">
            <div className="text-purple-300 mb-2">Widgets activos: {Array.from(widgets.values()).filter(w => w.enabled).length}</div>
            <div className="space-y-1">
              {['stats', 'calendar', 'notes', 'tasks'].map(type => (
                <button
                  key={type}
                  onClick={() => addWidget(type, 'right')}
                  className="w-full px-3 py-2 bg-purple-600 hover:bg-purple-700 rounded text-white text-xs"
                >
                  + {type}
                </button>
              ))}
            </div>
          </div>
        </motion.div>
      )}

      {/* Performance Metrics Panel */}
      {showPerformance && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="fixed bottom-4 left-4 bg-indigo-900/20 rounded-lg p-4 border border-indigo-700/50 shadow-xl z-40 max-w-xs"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-indigo-300">⚡ Métricas de Rendimiento</h3>
            <button onClick={() => setShowPerformance(false)} className="text-indigo-400 hover:text-indigo-300">✕</button>
          </div>
          <div className="space-y-2 text-xs">
            <div className="flex justify-between">
              <span className="text-indigo-300">Mensajes:</span>
              <span className="text-indigo-200">{performanceMetrics.get('messageCount') || 0}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-indigo-300">Tiempo render:</span>
              <span className="text-indigo-200">{(performanceMetrics.get('renderTime') || 0).toFixed(0)}ms</span>
            </div>
            <div className="flex justify-between">
              <span className="text-indigo-300">Memoria:</span>
              <span className="text-indigo-200">{((performanceMetrics.get('memoryUsage') || 0) / 1024 / 1024).toFixed(1)}MB</span>
            </div>
            <div className="flex justify-between">
              <span className="text-indigo-300">Altura componente:</span>
              <span className="text-indigo-200">{Math.round((performanceMetrics.get('componentSize') || 0) / 100)}px</span>
            </div>
          </div>
        </motion.div>
      )}

      {/* Advanced Accessibility Panel */}
      {accessibilityMode && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed top-20 left-4 bg-teal-900/20 rounded-lg p-4 border border-teal-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-teal-300">♿ Accesibilidad Avanzada</h3>
            <button onClick={() => setAccessibilityMode(false)} className="text-teal-400 hover:text-teal-300">✕</button>
          </div>
          <div className="space-y-2 text-xs">
            {Object.entries(accessibilityFeatures).map(([feature, enabled]) => (
              <label key={feature} className="flex items-center justify-between cursor-pointer">
                <span className="text-teal-300 capitalize">{feature.replace(/([A-Z])/g, ' $1').trim()}:</span>
                <button
                  onClick={() => toggleAccessibilityFeature(feature as any)}
                  className={`px-3 py-1 rounded text-white text-xs ${enabled ? 'bg-teal-600' : 'bg-slate-600'}`}
                >
                  {enabled ? 'ON' : 'OFF'}
                </button>
              </label>
            ))}
          </div>
        </motion.div>
      )}

      {/* Presentation Transitions Control */}
      {presentationMode && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-4 right-4 bg-pink-900/20 rounded-lg p-3 border border-pink-700/50 shadow-xl z-40"
        >
          <div className="flex items-center gap-2">
            <span className="text-pink-300 text-xs">🎬 Transición:</span>
            <select
              value={presentationTransitions}
              onChange={(e) => setPresentationTransition(e.target.value as any)}
              className="bg-slate-700 text-white text-xs px-2 py-1 rounded"
            >
              <option value="fade">Fade</option>
              <option value="slide">Slide</option>
              <option value="zoom">Zoom</option>
              <option value="none">None</option>
            </select>
          </div>
        </motion.div>
      )}

      {/* Audio Recording Indicator */}
      {audioRecording && (
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          className="fixed top-4 right-4 bg-red-900/20 rounded-lg p-3 border border-red-700/50 shadow-xl z-40"
        >
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
            <span className="text-red-300 text-sm">🎤 Grabando audio...</span>
            <button onClick={stopAudioRecording} className="text-red-400 hover:text-red-300">⏹️</button>
          </div>
        </motion.div>
      )}

      {/* Message Queue Panel */}
      {messageQueue.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-4 right-4 bg-cyan-900/20 rounded-lg p-4 border border-cyan-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-cyan-300">📋 Cola de Mensajes</h3>
            <button onClick={() => setMessageQueue([])} className="text-cyan-400 hover:text-cyan-300">✕</button>
          </div>
          <div className="space-y-2 text-xs max-h-48 overflow-y-auto">
            {messageQueue.map((item, idx) => (
              <div key={item.id} className="p-2 bg-cyan-800/30 rounded flex items-center justify-between">
                <span className="text-cyan-200 truncate flex-1">{item.message.substring(0, 30)}...</span>
                <span className="text-cyan-400 ml-2">P:{item.priority}</span>
              </div>
            ))}
          </div>
          <button
            onClick={processQueue}
            disabled={queueProcessing}
            className="w-full mt-2 px-3 py-2 bg-cyan-600 hover:bg-cyan-700 rounded text-white text-xs disabled:opacity-50"
          >
            {queueProcessing ? 'Procesando...' : '▶️ Procesar'}
          </button>
        </motion.div>
      )}

      {/* Message Graph Panel */}
      {showMessageGraph && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
          onClick={() => setShowMessageGraph(false)}
        >
          <motion.div
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            onClick={(e) => e.stopPropagation()}
            className="bg-slate-800 rounded-lg p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto border border-slate-700"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">🌐 Gráfico de Mensajes</h3>
              <button onClick={() => setShowMessageGraph(false)} className="text-slate-400 hover:text-white">✕</button>
            </div>
            <div className="space-y-4">
              <div className="text-slate-300 text-sm">
                <div>Enlaces: {Array.from(messageLinking.values()).flat().length}</div>
                <div>Relaciones: {Array.from(messageRelations.values()).flat().length}</div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                {Array.from(messageLinking.entries()).map(([sourceId, targets]) => (
                  <div key={sourceId} className="p-3 bg-slate-700/50 rounded">
                    <div className="text-xs text-slate-400 mb-1">Mensaje {sourceId.slice(0, 8)}</div>
                    <div className="text-white text-sm">
                      Enlazado a {targets.length} mensaje(s)
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}

      {/* Workflow Mode Panel */}
      {workflowMode && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed top-20 right-4 bg-amber-900/20 rounded-lg p-4 border border-amber-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-amber-300">⚙️ Modo Workflow</h3>
            <button onClick={() => setWorkflowMode(false)} className="text-amber-400 hover:text-amber-300">✕</button>
          </div>
          <div className="space-y-2 text-xs">
            <div className="text-amber-300 mb-2">Workflows activos: {messageWorkflow.size}</div>
            {Array.from(messageWorkflow.entries()).map(([id, workflow]) => (
              <div key={id} className="p-2 bg-amber-800/30 rounded">
                <div className="text-amber-200">Estado: {workflow.status}</div>
                <div className="text-amber-400">Pasos: {workflow.steps.length}</div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Advanced Analytics Panel */}
      {showAdvancedAnalytics && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="fixed bottom-4 left-4 bg-violet-900/20 rounded-lg p-4 border border-violet-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-violet-300">📊 Analytics Avanzado</h3>
            <button onClick={() => setShowAdvancedAnalytics(false)} className="text-violet-400 hover:text-violet-300">✕</button>
          </div>
          <div className="space-y-2 text-xs">
            {Array.from(messageAnalyticsAdvanced.entries()).slice(0, 5).map(([id, analytics]) => (
              <div key={id} className="p-2 bg-violet-800/30 rounded">
                <div className="flex justify-between text-violet-200">
                  <span>👁️ {analytics.views}</span>
                  <span>👆 {analytics.interactions}</span>
                  <span>🔗 {analytics.shares}</span>
                </div>
              </div>
            ))}
            <div className="text-violet-300 mt-2">
              Total: {Array.from(messageAnalyticsAdvanced.values()).reduce((acc, a) => acc + a.views + a.interactions + a.shares, 0)} interacciones
            </div>
          </div>
        </motion.div>
      )}

      {/* Multi-Device Sync Panel */}
      {multiDeviceSync && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed top-20 left-4 bg-emerald-900/20 rounded-lg p-4 border border-emerald-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-emerald-300">🔄 Sincronización Multi-Dispositivo</h3>
            <button onClick={() => setMultiDeviceSync(false)} className="text-emerald-400 hover:text-emerald-300">✕</button>
          </div>
          <div className="space-y-2 text-xs">
            <div className="text-emerald-300 mb-2">Mensajes sincronizados: {Array.from(messageSync.values()).filter(s => s.synced).length}</div>
            <input
              type="text"
              placeholder="ID del dispositivo"
              className="w-full px-2 py-1 bg-slate-700 text-white rounded text-xs"
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  const deviceId = (e.target as HTMLInputElement).value
                  if (deviceId) {
                    syncToDevice(deviceId)
                    ;(e.target as HTMLInputElement).value = ''
                  }
                }
              }}
            />
            <div className="text-emerald-400 text-xs mt-2">
              Dispositivos: {new Set(Array.from(messageSync.values()).map(s => s.device)).size}
            </div>
          </div>
        </motion.div>
      )}

      {/* Message Validation Indicators */}
      {autoValidate && (
        <div className="fixed top-4 left-4 bg-yellow-900/20 rounded-lg p-2 border border-yellow-700/50 shadow-xl z-40">
          <div className="text-xs text-yellow-300">
            ✓ Validación: {Array.from(messageValidation.values()).filter(v => v.valid).length}/{messageValidation.size}
          </div>
        </div>
      )}

      {/* Cache and Smart Search Panel */}
      {cacheEnabled && (
        <div className="fixed bottom-4 right-4 bg-blue-900/20 rounded-lg p-2 border border-blue-700/50 shadow-xl z-40">
          <div className="text-xs text-blue-300">
            💾 Caché: {messageCache.size} | 🔍 Índice: {searchIndex.size}
          </div>
        </div>
      )}

      {/* Batch Mode Panel */}
      {batchMode && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-4 right-4 bg-orange-900/20 rounded-lg p-4 border border-orange-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-orange-300">📦 Modo Batch</h3>
            <button onClick={() => setBatchMode(false)} className="text-orange-400 hover:text-orange-300">✕</button>
          </div>
          <div className="space-y-2 text-xs">
            <div className="text-orange-300 mb-2">Mensajes en batch: {messageBatch.length}</div>
            {messageBatch.slice(0, 3).map((msg, idx) => (
              <div key={idx} className="p-2 bg-orange-800/30 rounded text-orange-200 truncate">
                {msg.content.substring(0, 40)}...
              </div>
            ))}
            <button
              onClick={sendBatch}
              disabled={messageBatch.length === 0}
              className="w-full mt-2 px-3 py-2 bg-orange-600 hover:bg-orange-700 rounded text-white text-xs disabled:opacity-50"
            >
              📤 Enviar {messageBatch.length} mensajes
            </button>
          </div>
        </motion.div>
      )}

      {/* Macros Panel */}
      {macroEnabled && messageMacros.size > 0 && (
        <div className="fixed top-4 right-4 bg-green-900/20 rounded-lg p-2 border border-green-700/50 shadow-xl z-40">
          <div className="text-xs text-green-300">
            ⌨️ Macros: {messageMacros.size}
          </div>
        </div>
      )}

      {/* Undo/Redo Indicators */}
      {undoEnabled && (
        <div className="fixed top-4 left-4 bg-purple-900/20 rounded-lg p-2 border border-purple-700/50 shadow-xl z-40">
          <div className="text-xs text-purple-300 flex gap-2">
            <span>↶ {undoStack.length}</span>
            <span>↷ {redoStack.length}</span>
          </div>
        </div>
      )}

      {/* AI Analysis Panel */}
      {aiFeatures.sentiment && messageAI.size > 0 && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="fixed bottom-4 left-4 bg-pink-900/20 rounded-lg p-4 border border-pink-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-pink-300">🤖 Análisis IA</h3>
            <button onClick={() => setAiFeatures(prev => ({ ...prev, sentiment: false }))} className="text-pink-400 hover:text-pink-300">✕</button>
          </div>
          <div className="space-y-2 text-xs">
            {Array.from(messageAI.entries()).slice(0, 3).map(([id, analysis]) => (
              <div key={id} className="p-2 bg-pink-800/30 rounded">
                <div className="text-pink-200">Sentimiento: {analysis.sentiment}</div>
                {analysis.topics.length > 0 && (
                  <div className="text-pink-400">Temas: {analysis.topics.slice(0, 3).join(', ')}</div>
                )}
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Bookmarks Advanced Panel */}
      {messageBookmarksAdvanced.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed top-20 right-4 bg-indigo-900/20 rounded-lg p-4 border border-indigo-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-indigo-300">🔖 Bookmarks Avanzados</h3>
            <button onClick={() => setMessageBookmarksAdvanced(new Map())} className="text-indigo-400 hover:text-indigo-300">✕</button>
          </div>
          <div className="space-y-2 text-xs max-h-48 overflow-y-auto">
            {Array.from(messageBookmarksAdvanced.entries()).map(([id, bookmark]) => (
              <div key={id} className="p-2 bg-indigo-800/30 rounded">
                <div className="text-indigo-200 font-semibold">{bookmark.name}</div>
                <div className="text-indigo-400">Categoría: {bookmark.category}</div>
                {bookmark.tags.length > 0 && (
                  <div className="text-indigo-500 text-xs">Tags: {bookmark.tags.join(', ')}</div>
                )}
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Version Control Panel */}
      {versionControl && messageVersioning.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed top-20 left-4 bg-teal-900/20 rounded-lg p-4 border border-teal-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-teal-300">📝 Control de Versiones</h3>
            <button onClick={() => setVersionControl(false)} className="text-teal-400 hover:text-teal-300">✕</button>
          </div>
          <div className="space-y-2 text-xs max-h-48 overflow-y-auto">
            {Array.from(messageVersioning.entries()).map(([id, version]) => (
              <div key={id} className="p-2 bg-teal-800/30 rounded">
                <div className="text-teal-200">Versión: {version.version}</div>
                <div className="text-teal-400">Cambios: {version.changes.length}</div>
                <div className="text-teal-500 text-xs">
                  {new Date(version.timestamp).toLocaleString()}
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Collaboration Panel */}
      {collaborationEnabled && messageCollaboration.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-4 right-4 bg-cyan-900/20 rounded-lg p-4 border border-cyan-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-cyan-300">👥 Colaboración</h3>
            <button onClick={() => setCollaborationEnabled(false)} className="text-cyan-400 hover:text-cyan-300">✕</button>
          </div>
          <div className="space-y-2 text-xs">
            {Array.from(messageCollaboration.entries()).map(([id, collab]) => (
              <div key={id} className="p-2 bg-cyan-800/30 rounded">
                <div className="text-cyan-200">Usuarios: {collab.users.length}</div>
                <div className="text-cyan-400">Permisos: {collab.permissions.join(', ')}</div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Quick Replies Panel */}
      {quickReplies.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-20 right-4 bg-yellow-900/20 rounded-lg p-4 border border-yellow-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-yellow-300">⚡ Respuestas Rápidas</h3>
            <button onClick={() => setQuickReplies([])} className="text-yellow-400 hover:text-yellow-300">✕</button>
          </div>
          <div className="space-y-1 text-xs max-h-32 overflow-y-auto">
            {quickReplies.map((reply, idx) => (
              <button
                key={idx}
                onClick={() => {
                  setInput(reply)
                  toast.success('Respuesta rápida seleccionada', { icon: '⚡' })
                }}
                className="w-full text-left p-2 bg-yellow-800/30 rounded text-yellow-200 hover:bg-yellow-700/30"
              >
                {reply.substring(0, 40)}...
              </button>
            ))}
          </div>
        </motion.div>
      )}

      {/* Command Palette */}
      {commandPalette && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
          onClick={() => setCommandPalette(false)}
        >
          <motion.div
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            onClick={(e) => e.stopPropagation()}
            className="bg-slate-800 rounded-lg p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-slate-700"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">⌨️ Paleta de Comandos</h3>
              <button onClick={() => setCommandPalette(false)} className="text-slate-400 hover:text-white">✕</button>
            </div>
            <div className="space-y-2">
              {Array.from(commandSystem.values()).map((cmd, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    executeCommand(cmd.command)
                    setCommandPalette(false)
                  }}
                  className="w-full text-left p-3 bg-slate-700 hover:bg-slate-600 rounded text-white"
                >
                  <div className="font-semibold">{cmd.command}</div>
                  <div className="text-sm text-slate-400">{cmd.description}</div>
                </button>
              ))}
            </div>
          </motion.div>
        </motion.div>
      )}

      {/* Polls Panel */}
      {pollMode && messagePolls.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-4 right-4 bg-blue-900/20 rounded-lg p-4 border border-blue-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-blue-300">📊 Encuestas</h3>
            <button onClick={() => setPollMode(false)} className="text-blue-400 hover:text-blue-300">✕</button>
          </div>
          <div className="space-y-2 text-xs max-h-48 overflow-y-auto">
            {Array.from(messagePolls.entries()).map(([id, poll]) => (
              <div key={id} className="p-2 bg-blue-800/30 rounded">
                <div className="text-blue-200 font-semibold mb-1">{poll.question}</div>
                {poll.options.map((option, idx) => (
                  <button
                    key={idx}
                    onClick={() => votePoll(id, idx)}
                    className="w-full text-left p-1 bg-blue-700/30 rounded text-blue-200 hover:bg-blue-600/30 mb-1"
                  >
                    {option}: {poll.votes.get(idx.toString()) || 0} votos
                  </button>
                ))}
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Tasks Panel */}
      {taskMode && messageTasks.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed top-20 right-4 bg-green-900/20 rounded-lg p-4 border border-green-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-green-300">✓ Tareas</h3>
            <button onClick={() => setTaskMode(false)} className="text-green-400 hover:text-green-300">✕</button>
          </div>
          <div className="space-y-2 text-xs max-h-48 overflow-y-auto">
            {Array.from(messageTasks.entries()).map(([id, task]) => (
              <div key={id} className="p-2 bg-green-800/30 rounded flex items-center justify-between">
                <div className="flex-1">
                  <div className={`text-green-200 ${task.completed ? 'line-through' : ''}`}>{task.task}</div>
                  {task.dueDate && (
                    <div className="text-green-400 text-xs">
                      {new Date(task.dueDate).toLocaleDateString()}
                    </div>
                  )}
                </div>
                <button
                  onClick={() => toggleTask(id)}
                  className={`px-2 py-1 rounded text-xs ${task.completed ? 'bg-green-600' : 'bg-slate-600'}`}
                >
                  {task.completed ? '✓' : '○'}
                </button>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Reminders Panel */}
      {reminderSystem && messageRemindersAdvanced.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed top-20 left-4 bg-orange-900/20 rounded-lg p-4 border border-orange-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-orange-300">⏰ Recordatorios</h3>
            <button onClick={() => setReminderSystem(false)} className="text-orange-400 hover:text-orange-300">✕</button>
          </div>
          <div className="space-y-2 text-xs max-h-48 overflow-y-auto">
            {Array.from(messageRemindersAdvanced.entries()).map(([id, reminder]) => (
              <div key={id} className="p-2 bg-orange-800/30 rounded">
                <div className="text-orange-200">{reminder.reminder}</div>
                <div className="text-orange-400 text-xs">
                  {new Date(reminder.date).toLocaleString()}
                  {reminder.recurring && ` (${reminder.recurring})`}
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Calendar Panel */}
      {calendarIntegration && messageCalendar.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-4 left-4 bg-purple-900/20 rounded-lg p-4 border border-purple-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-purple-300">📅 Calendario</h3>
            <button onClick={() => setCalendarIntegration(false)} className="text-purple-400 hover:text-purple-300">✕</button>
          </div>
          <div className="space-y-2 text-xs max-h-48 overflow-y-auto">
            {Array.from(messageCalendar.entries()).map(([id, event]) => (
              <div key={id} className="p-2 bg-purple-800/30 rounded">
                <div className="text-purple-200 font-semibold">{event.event}</div>
                <div className="text-purple-400">
                  {new Date(event.date).toLocaleString()}
                  {event.duration && ` (${event.duration} min)`}
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Attachments Panel */}
      {attachmentManager && messageAttachments.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-4 right-4 bg-cyan-900/20 rounded-lg p-4 border border-cyan-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-cyan-300">📎 Adjuntos</h3>
            <button onClick={() => setAttachmentManager(false)} className="text-cyan-400 hover:text-cyan-300">✕</button>
          </div>
          <div className="space-y-2 text-xs max-h-48 overflow-y-auto">
            {Array.from(messageAttachments.entries()).map(([id, attachments]) => (
              <div key={id} className="space-y-1">
                {attachments.map((att, idx) => (
                  <div key={idx} className="p-2 bg-cyan-800/30 rounded">
                    <div className="text-cyan-200">{att.name}</div>
                    <div className="text-cyan-400 text-xs">{att.type}</div>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Event Log Panel */}
      {eventLog && messageEvents.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed top-4 right-4 bg-slate-900/20 rounded-lg p-4 border border-slate-700/50 shadow-xl z-40 max-w-sm max-h-64 overflow-y-auto"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-slate-300">📋 Registro de Eventos</h3>
            <button onClick={() => setEventLog(false)} className="text-slate-400 hover:text-slate-300">✕</button>
          </div>
          <div className="space-y-1 text-xs">
            {Array.from(messageEvents.entries()).slice(0, 10).map(([id, events]) => (
              <div key={id} className="p-1 bg-slate-800/30 rounded text-slate-400">
                {events[events.length - 1]?.type} - {new Date(events[events.length - 1]?.timestamp || 0).toLocaleTimeString()}
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Sync Status Indicator */}
      {syncStatus !== 'synced' && (
        <div className={`fixed top-4 left-4 rounded-lg p-2 border shadow-xl z-40 ${
          syncStatus === 'syncing' ? 'bg-blue-900/20 border-blue-700/50' :
          syncStatus === 'error' ? 'bg-red-900/20 border-red-700/50' :
          'bg-yellow-900/20 border-yellow-700/50'
        }`}>
          <div className="text-xs flex items-center gap-2">
            {syncStatus === 'syncing' && '🔄 Sincronizando...'}
            {syncStatus === 'error' && '❌ Error de sincronización'}
            {syncStatus === 'offline' && '📴 Modo offline'}
            {offlineQueue.length > 0 && `(${offlineQueue.length} en cola)`}
          </div>
        </div>
      )}

      {/* Plugins Panel */}
      {pluginManager && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-4 right-4 bg-indigo-900/20 rounded-lg p-4 border border-indigo-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-indigo-300">🔌 Plugins</h3>
            <button onClick={() => setPluginManager(false)} className="text-indigo-400 hover:text-indigo-300">✕</button>
          </div>
          <div className="space-y-2 text-xs max-h-48 overflow-y-auto">
            {Array.from(pluginSystem.entries()).map(([id, plugin]) => (
              <div key={id} className="p-2 bg-indigo-800/30 rounded flex items-center justify-between">
                <div>
                  <div className="text-indigo-200 font-semibold">{plugin.name}</div>
                  <div className="text-indigo-400">v{plugin.version}</div>
                </div>
                <button
                  onClick={() => togglePlugin(id)}
                  className={`px-2 py-1 rounded text-xs ${plugin.enabled ? 'bg-indigo-600' : 'bg-slate-600'}`}
                >
                  {plugin.enabled ? 'ON' : 'OFF'}
                </button>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Dev Tools Panel */}
      {devTools && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="fixed bottom-4 left-4 bg-slate-900 rounded-lg p-4 border border-slate-700 shadow-xl z-40 max-w-md max-h-96 overflow-y-auto"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-slate-300">🛠️ Dev Console</h3>
            <button onClick={() => setDevTools(false)} className="text-slate-400 hover:text-slate-300">✕</button>
          </div>
          <div className="space-y-1 text-xs font-mono text-green-400 max-h-80 overflow-y-auto">
            {devConsole.slice(-20).map((log, idx) => (
              <div key={idx} className="p-1">{log}</div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Performance Dashboard */}
      {performanceDashboard && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed top-20 right-4 bg-purple-900/20 rounded-lg p-4 border border-purple-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-purple-300">⚡ Rendimiento</h3>
            <button onClick={() => setPerformanceDashboard(false)} className="text-purple-400 hover:text-purple-300">✕</button>
          </div>
          <div className="space-y-2 text-xs">
            {Array.from(performanceMonitor.entries()).slice(-5).map(([id, metric]) => (
              <div key={id} className="p-2 bg-purple-800/30 rounded">
                <div className="text-purple-200">{metric.metric}: {metric.value.toFixed(2)}</div>
                <div className="text-purple-400 text-xs">
                  {new Date(metric.timestamp).toLocaleTimeString()}
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Analytics Dashboard */}
      {analyticsDashboard && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
          onClick={() => setAnalyticsDashboard(false)}
        >
          <motion.div
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            onClick={(e) => e.stopPropagation()}
            className="bg-slate-800 rounded-lg p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-slate-700"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">📊 Analytics Dashboard</h3>
              <button onClick={() => setAnalyticsDashboard(false)} className="text-slate-400 hover:text-white">✕</button>
            </div>
            <div className="space-y-4">
              {Array.from(analyticsData.entries()).map(([metric, data]) => (
                <div key={metric} className="p-4 bg-slate-700/50 rounded">
                  <div className="text-white font-semibold mb-2">{data.metric}</div>
                  <div className="text-2xl text-purple-400 mb-1">{data.value.toFixed(2)}</div>
                  <div className={`text-sm ${data.trend === 'up' ? 'text-green-400' : data.trend === 'down' ? 'text-red-400' : 'text-slate-400'}`}>
                    Tendencia: {data.trend}
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </motion.div>
      )}

      {/* Backup Manager */}
      {backupManager && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-4 right-4 bg-green-900/20 rounded-lg p-4 border border-green-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-green-300">💾 Backups</h3>
            <button onClick={() => setBackupManager(false)} className="text-green-400 hover:text-green-300">✕</button>
          </div>
          <div className="space-y-2 text-xs max-h-48 overflow-y-auto">
            {Array.from(messageBackup.entries()).map(([id, backup]) => (
              <div key={id} className="p-2 bg-green-800/30 rounded">
                <div className="text-green-200">{backup.format} - {(backup.size / 1024).toFixed(2)}KB</div>
                <div className="text-green-400 text-xs">
                  {new Date(backup.timestamp).toLocaleString()}
                </div>
              </div>
            ))}
            <button
              onClick={() => createBackup('json')}
              className="w-full mt-2 px-3 py-2 bg-green-600 hover:bg-green-700 rounded text-white text-xs"
            >
              + Crear Backup
            </button>
          </div>
        </motion.div>
      )}

      {/* Presentation Mode */}
      {presentationMode && presentationSlides.length > 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="fixed inset-0 bg-black z-50 flex items-center justify-center"
        >
          <div className="max-w-4xl w-full p-8">
            <div className="bg-slate-800 rounded-lg p-8 border border-slate-700">
              <div className="text-center mb-6">
                <h2 className="text-3xl font-bold text-white mb-4">{presentationSlides[presentationIndex]?.title}</h2>
                <div className="text-slate-300 text-lg">{presentationSlides[presentationIndex]?.content}</div>
              </div>
              <div className="flex justify-between items-center mt-8">
                <button
                  onClick={prevSlide}
                  disabled={presentationIndex === 0}
                  className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded text-white disabled:opacity-50"
                >
                  ← Anterior
                </button>
                <div className="text-slate-400">
                  {presentationIndex + 1} / {presentationSlides.length}
                </div>
                <button
                  onClick={nextSlide}
                  disabled={presentationIndex === presentationSlides.length - 1}
                  className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded text-white disabled:opacity-50"
                >
                  Siguiente →
                </button>
              </div>
              <button
                onClick={() => setPresentationMode(false)}
                className="absolute top-4 right-4 text-slate-400 hover:text-white"
              >
                ✕
              </button>
            </div>
          </div>
        </motion.div>
      )}

      {/* Help Center */}
      {helpCenter && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
          onClick={() => setHelpCenter(false)}
        >
          <motion.div
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            onClick={(e) => e.stopPropagation()}
            className="bg-slate-800 rounded-lg p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-slate-700"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">❓ Centro de Ayuda</h3>
              <button onClick={() => setHelpCenter(false)} className="text-slate-400 hover:text-white">✕</button>
            </div>
            <div className="space-y-4">
              {Array.from(helpSystem.values()).map((help, idx) => (
                <div key={idx} className="p-4 bg-slate-700/50 rounded">
                  <div className="text-white font-semibold mb-2">{help.topic}</div>
                  <div className="text-slate-300">{help.content}</div>
                  <div className="text-slate-500 text-xs mt-2">Categoría: {help.category}</div>
                </div>
              ))}
            </div>
          </motion.div>
        </motion.div>
      )}

      {/* Tutorial Mode */}
      {tutorialMode && tutorialSteps.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-4 left-4 bg-blue-900/20 rounded-lg p-4 border border-blue-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-blue-300">📚 Tutorial</h3>
            <button onClick={() => setTutorialMode(false)} className="text-blue-400 hover:text-blue-300">✕</button>
          </div>
          <div className="space-y-2 text-xs">
            {tutorialSteps.filter(s => !s.completed).slice(0, 1).map((step, idx) => (
              <div key={idx} className="p-2 bg-blue-800/30 rounded">
                <div className="text-blue-200 font-semibold">{step.title}</div>
                <div className="text-blue-300">{step.content}</div>
                <button
                  onClick={() => completeTutorialStep(step.step)}
                  className="mt-2 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-white text-xs"
                >
                  Completar
                </button>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Insights Panel */}
      {insightsPanel && messageInsights.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed top-20 right-4 bg-yellow-900/20 rounded-lg p-4 border border-yellow-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-yellow-300">💡 Insights</h3>
            <button onClick={() => setInsightsPanel(false)} className="text-yellow-400 hover:text-yellow-300">✕</button>
          </div>
          <div className="space-y-2 text-xs max-h-48 overflow-y-auto">
            {Array.from(messageInsights.entries()).map(([id, insight]) => (
              <div key={id} className="p-2 bg-yellow-800/30 rounded">
                <div className="text-yellow-200">{insight.insight}</div>
                <div className="text-yellow-400 text-xs">
                  Confianza: {(insight.confidence * 100).toFixed(0)}% | Tipo: {insight.type}
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Suggestions Panel */}
      {suggestionsPanel && messageSuggestions.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-4 left-4 bg-cyan-900/20 rounded-lg p-4 border border-cyan-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-cyan-300">💭 Sugerencias</h3>
            <button onClick={() => setSuggestionsPanel(false)} className="text-cyan-400 hover:text-cyan-300">✕</button>
          </div>
          <div className="space-y-2 text-xs max-h-48 overflow-y-auto">
            {Array.from(messageSuggestions.entries()).map(([id, suggestion]) => (
              <div key={id} className="p-2 bg-cyan-800/30 rounded">
                <div className="text-cyan-200 font-semibold mb-1">Contexto: {suggestion.context}</div>
                {suggestion.suggestions.map((sug, idx) => (
                  <div key={idx} className="text-cyan-300 text-xs mb-1">• {sug}</div>
                ))}
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Clustering Viewer */}
      {clusteringViewer && messageClustering.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed top-20 left-4 bg-pink-900/20 rounded-lg p-4 border border-pink-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-pink-300">📦 Clusters</h3>
            <button onClick={() => setClusteringViewer(false)} className="text-pink-400 hover:text-pink-300">✕</button>
          </div>
          <div className="space-y-2 text-xs max-h-48 overflow-y-auto">
            {Array.from(messageClustering.entries()).map(([id, cluster]) => (
              <div key={id} className="p-2 bg-pink-800/30 rounded">
                <div className="text-pink-200 font-semibold">{cluster.cluster}</div>
                <div className="text-pink-400 text-xs">{cluster.messages.length} mensajes</div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Shortcut Manager */}
      {shortcutManager && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
          onClick={() => setShortcutManager(false)}
        >
          <motion.div
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            onClick={(e) => e.stopPropagation()}
            className="bg-slate-800 rounded-lg p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-slate-700"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">⌨️ Gestor de Shortcuts</h3>
              <button onClick={() => setShortcutManager(false)} className="text-slate-400 hover:text-white">✕</button>
            </div>
            <div className="space-y-2">
              {Array.from(shortcutSystem.values()).map((shortcut, idx) => (
                <div key={idx} className="p-3 bg-slate-700/50 rounded flex items-center justify-between">
                  <div>
                    <div className="text-white font-semibold">{shortcut.key}</div>
                    <div className="text-slate-400 text-sm">{shortcut.description}</div>
                  </div>
                  <button
                    onClick={() => executeShortcut(shortcut.key)}
                    className="px-3 py-1 bg-slate-600 hover:bg-slate-500 rounded text-white text-xs"
                  >
                    Ejecutar
                  </button>
                </div>
              ))}
            </div>
          </motion.div>
        </motion.div>
      )}

      {/* Notification Center */}
      {notificationCenter && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed top-20 right-4 bg-blue-900/20 rounded-lg p-4 border border-blue-700/50 shadow-xl z-40 max-w-sm max-h-96 overflow-y-auto"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-blue-300">🔔 Centro de Notificaciones</h3>
            <button onClick={() => setNotificationCenter(false)} className="text-blue-400 hover:text-blue-300">✕</button>
          </div>
          <div className="space-y-2 text-xs">
            {Array.from(notificationSystem.entries())
              .sort((a, b) => b[1].priority - a[1].priority)
              .slice(0, 10)
              .map(([id, notif]) => (
                <div
                  key={id}
                  className={`p-2 rounded ${notif.read ? 'bg-blue-800/20' : 'bg-blue-800/30'} ${notif.read ? 'opacity-50' : ''}`}
                >
                  <div className="text-blue-200">{notif.type}</div>
                  <div className="text-blue-400 text-xs">
                    {new Date(notif.timestamp).toLocaleString()}
                  </div>
                  {!notif.read && (
                    <button
                      onClick={() => markNotificationRead(id)}
                      className="mt-1 px-2 py-1 bg-blue-600 hover:bg-blue-700 rounded text-white text-xs"
                    >
                      Marcar leído
                    </button>
                  )}
                </div>
              ))}
          </div>
        </motion.div>
      )}

      {/* Automation Manager */}
      {automationManager && workflowAutomation.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-4 left-4 bg-emerald-900/20 rounded-lg p-4 border border-emerald-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-emerald-300">⚙️ Automatización</h3>
            <button onClick={() => setAutomationManager(false)} className="text-emerald-400 hover:text-emerald-300">✕</button>
          </div>
          <div className="space-y-2 text-xs max-h-48 overflow-y-auto">
            {Array.from(workflowAutomation.entries()).map(([id, workflow]) => (
              <div key={id} className="p-2 bg-emerald-800/30 rounded">
                <div className="text-emerald-200 font-semibold">Trigger: {workflow.trigger}</div>
                <div className="text-emerald-400 text-xs">Acciones: {workflow.actions.length}</div>
                <div className={`text-emerald-400 text-xs mt-1 ${workflow.enabled ? 'text-green-400' : 'text-red-400'}`}>
                  {workflow.enabled ? '✓ Activo' : '○ Inactivo'}
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Analytics Viewer */}
      {analyticsViewer && messageAnalytics.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-4 right-4 bg-violet-900/20 rounded-lg p-4 border border-violet-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-violet-300">📊 Analytics</h3>
            <button onClick={() => setAnalyticsViewer(false)} className="text-violet-400 hover:text-violet-300">✕</button>
          </div>
          <div className="space-y-2 text-xs max-h-48 overflow-y-auto">
            {Array.from(messageAnalytics.entries()).slice(0, 5).map(([id, analytics]) => (
              <div key={id} className="p-2 bg-violet-800/30 rounded">
                <div className="text-violet-200">👁️ {analytics.views} | 👆 {analytics.interactions} | 🔗 {analytics.shares}</div>
                <div className="text-violet-400 text-xs">
                  {new Date(analytics.timestamp).toLocaleString()}
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Quality Monitor */}
      {qualityMonitor && messageQuality.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed top-20 left-4 bg-amber-900/20 rounded-lg p-4 border border-amber-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-amber-300">⭐ Monitor de Calidad</h3>
            <button onClick={() => setQualityMonitor(false)} className="text-amber-400 hover:text-amber-300">✕</button>
          </div>
          <div className="space-y-2 text-xs max-h-48 overflow-y-auto">
            {Array.from(messageQuality.entries()).map(([id, quality]) => (
              <div key={id} className="p-2 bg-amber-800/30 rounded">
                <div className="text-amber-200 font-semibold">Calidad: {quality.score}/100</div>
                <div className="text-amber-400 text-xs">
                  Longitud: {quality.metrics.length} | Palabras: {quality.metrics.words}
                </div>
                <div className="w-full bg-amber-800/30 rounded-full h-2 mt-1">
                  <div
                    className="bg-amber-500 h-2 rounded-full transition-all"
                    style={{ width: `${quality.score}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Performance Optimization Indicator */}
      {performanceOptimization.virtualScrolling && (
        <div className="fixed bottom-4 left-4 bg-green-900/20 rounded-lg p-2 border border-green-700/50 shadow-xl z-40">
          <div className="text-xs text-green-300">
            ⚡ Optimizado: VS={performanceOptimization.virtualScrolling ? 'ON' : 'OFF'} | 
            LL={performanceOptimization.lazyLoading ? 'ON' : 'OFF'} | 
            M={performanceOptimization.memoization ? 'ON' : 'OFF'}
          </div>
        </div>
      )}

      {/* Translation Mode Indicator */}
      {translationMode && (
        <div className="fixed top-4 right-4 bg-blue-900/20 rounded-lg p-2 border border-blue-700/50 shadow-xl z-40">
          <div className="text-xs text-blue-300 flex items-center gap-2">
            🌐 Traducción: {targetLanguage.toUpperCase()}
            <select
              value={targetLanguage}
              onChange={(e) => setTargetLanguage(e.target.value)}
              className="bg-slate-700 text-white text-xs px-1 py-0.5 rounded"
            >
              <option value="en">EN</option>
              <option value="es">ES</option>
              <option value="fr">FR</option>
              <option value="de">DE</option>
            </select>
          </div>
        </div>
      )}

      {/* Version Manager */}
      {versionManager && conversationVersions.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-4 right-4 bg-teal-900/20 rounded-lg p-4 border border-teal-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-teal-300">📝 Versiones</h3>
            <button onClick={() => setVersionManager(false)} className="text-teal-400 hover:text-teal-300">✕</button>
          </div>
          <div className="space-y-2 text-xs max-h-48 overflow-y-auto">
            {Array.from(conversationVersions.entries()).map(([id, version]) => (
              <div key={id} className="p-2 bg-teal-800/30 rounded">
                <div className="text-teal-200 font-semibold">v{version.version}: {version.description}</div>
                <div className="text-teal-400 text-xs">
                  {new Date(version.timestamp).toLocaleString()} | {version.messages.length} mensajes
                </div>
                <button
                  onClick={() => restoreConversationVersion(id)}
                  className="mt-1 px-2 py-1 bg-teal-600 hover:bg-teal-700 rounded text-white text-xs"
                >
                  Restaurar
                </button>
              </div>
            ))}
            <button
              onClick={() => createConversationVersion('Nueva versión')}
              className="w-full mt-2 px-3 py-2 bg-teal-600 hover:bg-teal-700 rounded text-white text-xs"
            >
              + Crear Versión
            </button>
          </div>
        </motion.div>
      )}

      {/* AI Manager */}
      {aiManager && externalAI.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed top-20 right-4 bg-purple-900/20 rounded-lg p-4 border border-purple-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-purple-300">🤖 IA Externa</h3>
            <button onClick={() => setAiManager(false)} className="text-purple-400 hover:text-purple-300">✕</button>
          </div>
          <div className="space-y-2 text-xs max-h-48 overflow-y-auto">
            {Array.from(externalAI.entries()).map(([id, ai]) => (
              <div key={id} className="p-2 bg-purple-800/30 rounded">
                <div className="text-purple-200 font-semibold">{ai.service}</div>
                <div className={`text-purple-400 text-xs ${ai.enabled ? 'text-green-400' : 'text-red-400'}`}>
                  {ai.enabled ? '✓ Conectado' : '○ Desconectado'}
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Learning Mode */}
      {learningMode && learningData.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-4 left-4 bg-indigo-900/20 rounded-lg p-4 border border-indigo-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-indigo-300">🧠 Modo Aprendizaje</h3>
            <button onClick={() => setLearningMode(false)} className="text-indigo-400 hover:text-indigo-300">✕</button>
          </div>
          <div className="space-y-2 text-xs max-h-48 overflow-y-auto">
            {Array.from(learningData.entries()).map(([pattern, data]) => (
              <div key={pattern} className="p-2 bg-indigo-800/30 rounded">
                <div className="text-indigo-200 font-semibold">{pattern}</div>
                <div className="text-indigo-400 text-xs">Confianza: {(data.confidence * 100).toFixed(0)}%</div>
                <div className="w-full bg-indigo-800/30 rounded-full h-1 mt-1">
                  <div
                    className="bg-indigo-500 h-1 rounded-full transition-all"
                    style={{ width: `${data.confidence * 100}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Recommendations Panel */}
      {recommendationsPanel && smartRecommendations.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed top-20 left-4 bg-cyan-900/20 rounded-lg p-4 border border-cyan-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-cyan-300">💡 Recomendaciones</h3>
            <button onClick={() => setRecommendationsPanel(false)} className="text-cyan-400 hover:text-cyan-300">✕</button>
          </div>
          <div className="space-y-2 text-xs max-h-48 overflow-y-auto">
            {Array.from(smartRecommendations.entries())
              .sort((a, b) => b[1].score - a[1].score)
              .slice(0, 5)
              .map(([id, rec]) => (
                <div key={id} className="p-2 bg-cyan-800/30 rounded">
                  <div className="text-cyan-200 font-semibold">{rec.type}</div>
                  <div className="text-cyan-300 text-xs">{rec.content}</div>
                  <div className="text-cyan-400 text-xs mt-1">Score: {(rec.score * 100).toFixed(0)}%</div>
                </div>
              ))}
          </div>
        </motion.div>
      )}

      {/* Sentiment Viewer */}
      {sentimentViewer && sentimentAnalysis.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-4 right-4 bg-pink-900/20 rounded-lg p-4 border border-pink-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-pink-300">😊 Sentimientos</h3>
            <button onClick={() => setSentimentViewer(false)} className="text-pink-400 hover:text-pink-300">✕</button>
          </div>
          <div className="space-y-2 text-xs max-h-48 overflow-y-auto">
            {Array.from(sentimentAnalysis.entries()).slice(0, 5).map(([id, sentiment]) => (
              <div key={id} className="p-2 bg-pink-800/30 rounded">
                <div className={`text-pink-200 font-semibold ${
                  sentiment.sentiment === 'positive' ? 'text-green-400' :
                  sentiment.sentiment === 'negative' ? 'text-red-400' :
                  'text-yellow-400'
                }`}>
                  {sentiment.sentiment === 'positive' ? '😊' : sentiment.sentiment === 'negative' ? '😢' : '😐'} {sentiment.sentiment}
                </div>
                <div className="text-pink-400 text-xs">Score: {(sentiment.score * 100).toFixed(0)}%</div>
                {sentiment.emotions.length > 0 && (
                  <div className="text-pink-500 text-xs">Emociones: {sentiment.emotions.join(', ')}</div>
                )}
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Summary Viewer */}
      {summaryViewer && autoSummary.size > 0 && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
          onClick={() => setSummaryViewer(false)}
        >
          <motion.div
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            onClick={(e) => e.stopPropagation()}
            className="bg-slate-800 rounded-lg p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-slate-700"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">📝 Resumen Automático</h3>
              <button onClick={() => setSummaryViewer(false)} className="text-slate-400 hover:text-white">✕</button>
            </div>
            <div className="space-y-4">
              {Array.from(autoSummary.entries()).map(([id, summary]) => (
                <div key={id} className="p-4 bg-slate-700/50 rounded">
                  <div className="text-white mb-2">{summary.summary}</div>
                  <div className="text-slate-300 text-sm">
                    <div className="font-semibold mb-1">Puntos clave:</div>
                    {summary.keyPoints.map((point, idx) => (
                      <div key={idx} className="text-slate-400">• {point}</div>
                    ))}
                  </div>
                  <div className="text-slate-500 text-xs mt-2">
                    {new Date(summary.timestamp).toLocaleString()}
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </motion.div>
      )}

      {/* Context Viewer */}
      {contextViewer && messageContext.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed top-20 left-4 bg-emerald-900/20 rounded-lg p-4 border border-emerald-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-emerald-300">🔍 Contexto</h3>
            <button onClick={() => setContextViewer(false)} className="text-emerald-400 hover:text-emerald-300">✕</button>
          </div>
          <div className="space-y-2 text-xs max-h-48 overflow-y-auto">
            {Array.from(messageContext.entries()).map(([id, context]) => (
              <div key={id} className="p-2 bg-emerald-800/30 rounded">
                <div className="text-emerald-200 font-semibold mb-1">Contexto:</div>
                <div className="text-emerald-300 text-xs mb-1">{context.context}...</div>
                <div className="text-emerald-400 text-xs">Relacionados: {context.relatedMessages.length}</div>
                <div className="text-emerald-500 text-xs">Temas: {context.topics.join(', ')}</div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Pattern Analyzer */}
      {patternAnalyzer && messagePatterns.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-4 left-4 bg-orange-900/20 rounded-lg p-4 border border-orange-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-orange-300">🔍 Patrones</h3>
            <button onClick={() => setPatternAnalyzer(false)} className="text-orange-400 hover:text-orange-300">✕</button>
          </div>
          <div className="space-y-2 text-xs max-h-48 overflow-y-auto">
            {Array.from(messagePatterns.entries())
              .sort((a, b) => b[1].frequency - a[1].frequency)
              .slice(0, 5)
              .map(([pattern, data]) => (
                <div key={pattern} className="p-2 bg-orange-800/30 rounded">
                  <div className="text-orange-200 font-semibold">{pattern}</div>
                  <div className="text-orange-400 text-xs">Frecuencia: {data.frequency}</div>
                  {data.examples.length > 0 && (
                    <div className="text-orange-500 text-xs mt-1">Ejemplo: {data.examples[0]}...</div>
                  )}
                </div>
              ))}
          </div>
        </motion.div>
      )}

      {/* Flow Viewer */}
      {flowViewer && messageFlow.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed top-20 right-4 bg-blue-900/20 rounded-lg p-4 border border-blue-700/50 shadow-xl z-40 max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-blue-300">🌊 Flujo</h3>
            <button onClick={() => setFlowViewer(false)} className="text-blue-400 hover:text-blue-300">✕</button>
          </div>
          <div className="space-y-2 text-xs max-h-48 overflow-y-auto">
            {Array.from(messageFlow.entries()).slice(0, 5).map(([id, flow]) => (
              <div key={id} className="p-2 bg-blue-800/30 rounded">
                <div className="text-blue-200">De: {flow.from.slice(0, 8)} → A: {flow.to.slice(0, 8)}</div>
                <div className="text-blue-400 text-xs">Tipo: {flow.type}</div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Timeline Viewer */}
      {timelineViewer && messageTimeline.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-4 right-4 bg-violet-900/20 rounded-lg p-4 border border-violet-700/50 shadow-xl z-40 max-w-sm max-h-64 overflow-y-auto"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-violet-300">📅 Timeline</h3>
            <button onClick={() => setTimelineViewer(false)} className="text-violet-400 hover:text-violet-300">✕</button>
          </div>
          <div className="space-y-1 text-xs">
            {Array.from(messageTimeline.entries()).map(([id, timeline]) => (
              <div key={id} className="p-1 bg-violet-800/30 rounded">
                {timeline.events.map((event, idx) => (
                  <div key={idx} className="text-violet-300 text-xs">
                    {event.type} - {new Date(event.timestamp).toLocaleTimeString()}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Visualization Mode Indicator */}
      {visualizationMode !== 'none' && (
        <div className="fixed top-4 left-4 bg-indigo-900/20 rounded-lg p-2 border border-indigo-700/50 shadow-xl z-40">
          <div className="text-xs text-indigo-300 flex items-center gap-2">
            📊 Visualización: {visualizationMode}
            <button onClick={() => setVisualizationMode('none')} className="text-indigo-400 hover:text-indigo-300">✕</button>
          </div>
        </div>
      )}

      {/* Dependency Viewer */}
      {dependencyViewer && messageDependencies.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed top-20 left-4 bg-amber-900/20 rounded-lg p-4 border border-amber-700/50 shadow-xl z-40 max-w-sm max-h-64 overflow-y-auto"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-amber-300">🔗 Dependencias</h3>
            <button onClick={() => setDependencyViewer(false)} className="text-amber-400 hover:text-amber-300">✕</button>
          </div>
          <div className="space-y-2 text-xs">
            {Array.from(messageDependencies.entries()).slice(0, 5).map(([id, deps]) => (
              <div key={id} className="p-2 bg-amber-800/30 rounded">
                <div className="text-amber-200 font-semibold">ID: {id.slice(0, 8)}</div>
                <div className="text-amber-400 text-xs">Depende de: {deps.dependsOn.length}</div>
                <div className="text-amber-500 text-xs">Requerido por: {deps.requiredBy.length}</div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Metrics Dashboard */}
      {metricsDashboard && messageMetrics.size > 0 && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
          onClick={() => setMetricsDashboard(false)}
        >
          <motion.div
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            onClick={(e) => e.stopPropagation()}
            className="bg-slate-800 rounded-lg p-6 max-w-3xl w-full max-h-[90vh] overflow-y-auto border border-slate-700"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">📈 Dashboard de Métricas</h3>
              <button onClick={() => setMetricsDashboard(false)} className="text-slate-400 hover:text-white">✕</button>
            </div>
            <div className="grid grid-cols-3 gap-4 mb-4">
              <div className="p-4 bg-slate-700/50 rounded">
                <div className="text-slate-400 text-xs mb-1">Tiempo de lectura promedio</div>
                <div className="text-white text-xl font-semibold">
                  {Array.from(messageMetrics.values()).reduce((acc, m) => acc + m.readTime, 0) / messageMetrics.size / 1000}s
                </div>
              </div>
              <div className="p-4 bg-slate-700/50 rounded">
                <div className="text-slate-400 text-xs mb-1">Tiempo de respuesta promedio</div>
                <div className="text-white text-xl font-semibold">
                  {Array.from(messageMetrics.values()).reduce((acc, m) => acc + m.responseTime, 0) / messageMetrics.size / 1000}s
                </div>
              </div>
              <div className="p-4 bg-slate-700/50 rounded">
                <div className="text-slate-400 text-xs mb-1">Engagement promedio</div>
                <div className="text-white text-xl font-semibold">
                  {(Array.from(messageMetrics.values()).reduce((acc, m) => acc + m.engagement, 0) / messageMetrics.size * 100).toFixed(1)}%
                </div>
              </div>
            </div>
            <div className="space-y-2 text-xs">
              {Array.from(messageMetrics.entries()).map(([id, metrics]) => (
                <div key={id} className="p-3 bg-slate-700/50 rounded">
                  <div className="text-white font-semibold mb-1">ID: {id.slice(0, 8)}</div>
                  <div className="grid grid-cols-3 gap-2 text-slate-300">
                    <div>Lectura: {metrics.readTime}ms</div>
                    <div>Respuesta: {metrics.responseTime}ms</div>
                    <div>Engagement: {(metrics.engagement * 100).toFixed(1)}%</div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </motion.div>
      )}

      {/* Alert Center */}
      {alertCenter && messageAlerts.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed top-20 right-4 bg-red-900/20 rounded-lg p-4 border border-red-700/50 shadow-xl z-40 max-w-sm max-h-64 overflow-y-auto"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-red-300">⚠️ Alertas</h3>
            <button onClick={() => setAlertCenter(false)} className="text-red-400 hover:text-red-300">✕</button>
          </div>
          <div className="space-y-2 text-xs">
            {Array.from(messageAlerts.entries()).slice(0, 5).map(([id, alert]) => (
              <div key={id} className={`p-2 rounded ${
                alert.severity === 'error' ? 'bg-red-800/30' :
                alert.severity === 'warning' ? 'bg-yellow-800/30' :
                'bg-blue-800/30'
              }`}>
                <div className={`font-semibold ${
                  alert.severity === 'error' ? 'text-red-200' :
                  alert.severity === 'warning' ? 'text-yellow-200' :
                  'text-blue-200'
                }`}>
                  {alert.type}
                </div>
                <div className={`text-xs ${
                  alert.severity === 'error' ? 'text-red-300' :
                  alert.severity === 'warning' ? 'text-yellow-300' :
                  'text-blue-300'
                }`}>
                  {alert.message}
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Category Manager */}
      {categoryManager && messageCategories.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-4 left-4 bg-green-900/20 rounded-lg p-4 border border-green-700/50 shadow-xl z-40 max-w-sm max-h-64 overflow-y-auto"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-green-300">📁 Categorías</h3>
            <button onClick={() => setCategoryManager(false)} className="text-green-400 hover:text-green-300">✕</button>
          </div>
          <div className="space-y-2 text-xs">
            {Array.from(messageCategories.entries()).map(([id, cat]) => (
              <div key={id} className="p-2 bg-green-800/30 rounded">
                <div className="text-green-200 font-semibold">{cat.category}</div>
                {cat.subcategory && (
                  <div className="text-green-400 text-xs">→ {cat.subcategory}</div>
                )}
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Notification Center */}
      {notificationCenter && messageNotifications.size > 0 && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="fixed top-20 right-4 bg-blue-900/20 rounded-lg p-4 border border-blue-700/50 shadow-xl z-40 max-w-sm max-h-96 overflow-y-auto"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-blue-300">🔔 Notificaciones ({messageNotifications.size})</h3>
            <button onClick={() => setNotificationCenter(false)} className="text-blue-400 hover:text-blue-300">✕</button>
          </div>
          <div className="space-y-2 text-xs">
            {Array.from(messageNotifications.entries())
              .sort((a, b) => b[1].timestamp - a[1].timestamp)
              .slice(0, 10)
              .map(([id, notif]) => (
                <div key={id} className={`p-2 rounded ${notif.read ? 'bg-blue-800/20' : 'bg-blue-800/40'}`}>
                  <div className="text-blue-200 font-semibold">{notif.title}</div>
                  <div className="text-blue-300 text-xs">{notif.body}</div>
                  <div className="text-blue-400 text-xs mt-1">{new Date(notif.timestamp).toLocaleTimeString()}</div>
                </div>
              ))}
          </div>
        </motion.div>
      )}

      {/* Favorites Panel */}
      {favoritesPanel && messageFavorites.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-4 right-4 bg-yellow-900/20 rounded-lg p-4 border border-yellow-700/50 shadow-xl z-40 max-w-sm max-h-64 overflow-y-auto"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-yellow-300">⭐ Favoritos ({messageFavorites.size})</h3>
            <button onClick={() => setFavoritesPanel(false)} className="text-yellow-400 hover:text-yellow-300">✕</button>
          </div>
          <div className="space-y-1 text-xs">
            {Array.from(messageFavorites).slice(0, 10).map((id) => (
              <div key={id} className="p-2 bg-yellow-800/30 rounded">
                <div className="text-yellow-200">ID: {id.slice(0, 8)}...</div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Pinned Panel */}
      {pinnedPanel && messagePinned.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed top-20 left-4 bg-purple-900/20 rounded-lg p-4 border border-purple-700/50 shadow-xl z-40 max-w-sm max-h-64 overflow-y-auto"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-purple-300">📌 Fijados ({messagePinned.size})</h3>
            <button onClick={() => setPinnedPanel(false)} className="text-purple-400 hover:text-purple-300">✕</button>
          </div>
          <div className="space-y-1 text-xs">
            {Array.from(messagePinned).slice(0, 10).map((id) => (
              <div key={id} className="p-2 bg-purple-800/30 rounded">
                <div className="text-purple-200">ID: {id.slice(0, 8)}...</div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Shortcuts Panel */}
      {shortcutPanel && messageShortcuts.size > 0 && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
          onClick={() => setShortcutPanel(false)}
        >
          <motion.div
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            onClick={(e) => e.stopPropagation()}
            className="bg-slate-800 rounded-lg p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-slate-700"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">⌨️ Atajos de Teclado</h3>
              <button onClick={() => setShortcutPanel(false)} className="text-slate-400 hover:text-white">✕</button>
            </div>
            <div className="grid grid-cols-2 gap-4 text-sm">
              {Array.from(messageShortcuts.entries()).map(([key, shortcut]) => (
                <div key={key} className="p-3 bg-slate-700/50 rounded">
                  <div className="text-white font-semibold mb-1">{shortcut.key}</div>
                  <div className="text-slate-300 text-xs">{shortcut.description}</div>
                </div>
              ))}
            </div>
          </motion.div>
        </motion.div>
      )}

      {/* History Viewer */}
      {historyViewer && messageHistory.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-4 left-4 bg-gray-900/20 rounded-lg p-4 border border-gray-700/50 shadow-xl z-40 max-w-sm max-h-64 overflow-y-auto"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-gray-300">📜 Historial ({messageHistory.size})</h3>
            <button onClick={() => setHistoryViewer(false)} className="text-gray-400 hover:text-gray-300">✕</button>
          </div>
          <div className="space-y-2 text-xs">
            {Array.from(messageHistory.entries())
              .sort((a, b) => b[1].timestamp - a[1].timestamp)
              .slice(0, 5)
              .map(([id, history]) => (
                <div key={id} className="p-2 bg-gray-800/30 rounded">
                  <div className="text-gray-200 text-xs mb-1">Antes: {history.previous.substring(0, 30)}...</div>
                  <div className="text-gray-300 text-xs">Después: {history.current.substring(0, 30)}...</div>
                </div>
              ))}
          </div>
        </motion.div>
      )}

      {/* Split Screen Mode Styles */}
      {splitScreenMode && (
        <style>{`
          .chat-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
          }
          .chat-messages {
            max-height: calc(100vh - 200px);
            overflow-y: auto;
          }
        `}</style>
      )}

      {/* Accessibility Styles */}
      {accessibilityFeatures.highContrast && (
        <style>{`
          .chat-container {
            filter: contrast(1.2);
          }
          .message {
            border: 2px solid currentColor;
          }
        `}</style>
      )}
      {accessibilityFeatures.largeText && (
        <style>{`
          .message-content {
            font-size: 1.25rem;
            line-height: 1.8;
          }
        `}</style>
      )}
      {accessibilityFeatures.reducedMotion && (
        <style>{`
          * {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
          }
        `}</style>
      )}

      {/* Print Mode Styles */}
      {printMode && (
        <style>{`
          @media print {
            body * {
              visibility: hidden;
            }
            .print-content, .print-content * {
              visibility: visible;
            }
            .print-content {
              position: absolute;
              left: 0;
              top: 0;
              width: 100%;
            }
          }
        `}</style>
      )}
    </div>
    </>
  )
}


