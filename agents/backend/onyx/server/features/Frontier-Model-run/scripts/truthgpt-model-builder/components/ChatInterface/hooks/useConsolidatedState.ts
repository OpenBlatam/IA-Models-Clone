import { useState, useCallback } from 'react'
import { Model } from '@/store/modelStore'

/**
 * Consolidated state hook that groups related state variables
 * Reduces the number of individual useState calls
 */
export interface ConsolidatedChatState {
  // Core chat state
  input: string
  isLoading: boolean
  messages: Array<{ id: string; role: 'user' | 'assistant'; content: string; timestamp: number }>
  
  // UI visibility states
  ui: {
    showPreview: boolean
    showHistory: boolean
    showComparator: boolean
    showTour: boolean
    showMetrics: boolean
    showProactive: boolean
    showStats: boolean
    showFilters: boolean
    showTemplates: boolean
    showCommandPalette: boolean
    showAccessibility: boolean
    showDebug: boolean
    showWordCount: boolean
    showSentiment: boolean
    showPrintMode: boolean
    showReactions: boolean
    showTags: boolean
    showEditHistory: boolean
    showComments: boolean
    showVersions: boolean
    showComparison: boolean
    showWidgets: boolean
    showPerformance: boolean
    showAdvancedAnalytics: boolean
  }
  
  // Model state
  models: {
    selectedModels: Model[]
    validation: any
    previewSpec: any
    modelHistory: any[]
  }
  
  // View and display settings
  display: {
    viewMode: 'normal' | 'compact' | 'comfortable'
    theme: 'dark' | 'light' | 'auto'
    fontSize: 'small' | 'medium' | 'large'
    autoScroll: boolean
    showCodeSyntax: boolean
    showMessageTimestamps: boolean
    showTypingDots: boolean
    compactMode: boolean
    zenMode: boolean
    readMode: boolean
    presentationMode: boolean
    fullscreenMode: boolean
    splitScreenMode: boolean
  }
  
  // Search and filter state
  search: {
    searchQuery: string
    currentSearchIndex: number
    highlightSearch: boolean
    showSmartSuggestions: boolean
    filterRole: 'all' | 'user' | 'assistant'
    advancedSearch: boolean
    searchFilters: {
      dateRange?: { start: Date; end: Date }
      minWords?: number
      maxWords?: number
      hasCode?: boolean
      hasLinks?: boolean
    }
  }
  
  // Message collections
  messageCollections: {
    favoriteMessages: Set<string>
    selectedMessages: Set<string>
    collapsedMessages: Set<string>
    archivedMessages: Set<string>
    pinnedMessages: Set<string>
    messageBookmarks: Set<string>
    encryptedMessages: Set<string>
  }
  
  // Feature flags
  features: {
    useBulkChatMode: boolean
    voiceInputEnabled: boolean
    voiceOutputEnabled: boolean
    isRecording: boolean
    autoSave: boolean
    autoFormat: boolean
    autoTranslate: boolean
    autoSummarize: boolean
    autoValidate: boolean
    autoFilter: boolean
    autoComplete: boolean
    typingPrediction: boolean
    smartSearch: boolean
    smartSuggestionsEnabled: boolean
    collaborationMode: boolean
    devMode: boolean
    accessibilityMode: boolean
    messageDeduplication: boolean
    messageCompression: boolean
    cacheEnabled: boolean
    versionControl: boolean
    collaborationEnabled: boolean
    undoEnabled: boolean
    macroEnabled: boolean
    highlightEnabled: boolean
    importEnabled: boolean
    reminderSystem: boolean
    eventLog: boolean
    realTimeStats: boolean
    smartNotifications: boolean
    customShortcutsEnabled: boolean
    encryptionEnabled: boolean
    multiDeviceSync: boolean
    realTimeSync: boolean
    cloudSync: boolean
    offlineMode: boolean
    batchMode: boolean
    pollMode: boolean
    taskMode: boolean
    workflowMode: boolean
    quizMode: boolean
    studyMode: boolean
    productivityMode: boolean
    tutorialMode: boolean
    learningMode: boolean
    dictationMode: boolean
    translationMode: boolean
    readingMode: boolean
    locationSharing: boolean
    calendarIntegration: boolean
    pluginSystem: boolean
    sessionRecording: boolean
  }
}

const defaultState: ConsolidatedChatState = {
  input: '',
  isLoading: false,
  messages: [],
  ui: {
    showPreview: false,
    showHistory: false,
    showComparator: false,
    showTour: false,
    showMetrics: false,
    showProactive: false,
    showStats: false,
    showFilters: false,
    showTemplates: false,
    showCommandPalette: false,
    showAccessibility: false,
    showDebug: false,
    showWordCount: false,
    showSentiment: false,
    showPrintMode: false,
    showReactions: true,
    showTags: true,
    showEditHistory: false,
    showComments: false,
    showVersions: false,
    showComparison: false,
    showWidgets: false,
    showPerformance: false,
    showAdvancedAnalytics: false,
  },
  models: {
    selectedModels: [],
    validation: null,
    previewSpec: null,
    modelHistory: [],
  },
  display: {
    viewMode: 'normal',
    theme: 'dark',
    fontSize: 'medium',
    autoScroll: true,
    showCodeSyntax: true,
    showMessageTimestamps: true,
    showTypingDots: true,
    compactMode: false,
    zenMode: false,
    readMode: false,
    presentationMode: false,
    fullscreenMode: false,
    splitScreenMode: false,
  },
  search: {
    searchQuery: '',
    currentSearchIndex: -1,
    highlightSearch: true,
    showSmartSuggestions: false,
    filterRole: 'all',
    advancedSearch: false,
    searchFilters: {},
  },
  messageCollections: {
    favoriteMessages: new Set(),
    selectedMessages: new Set(),
    collapsedMessages: new Set(),
    archivedMessages: new Set(),
    pinnedMessages: new Set(),
    messageBookmarks: new Set(),
    encryptedMessages: new Set(),
  },
  features: {
    useBulkChatMode: true,
    voiceInputEnabled: false,
    voiceOutputEnabled: false,
    isRecording: false,
    autoSave: true,
    autoFormat: true,
    autoTranslate: false,
    autoSummarize: false,
    autoValidate: false,
    autoFilter: false,
    autoComplete: true,
    typingPrediction: true,
    smartSearch: true,
    smartSuggestionsEnabled: true,
    collaborationMode: false,
    devMode: false,
    accessibilityMode: false,
    messageDeduplication: true,
    messageCompression: false,
    cacheEnabled: true,
    versionControl: true,
    collaborationEnabled: false,
    undoEnabled: true,
    macroEnabled: true,
    highlightEnabled: true,
    importEnabled: true,
    reminderSystem: true,
    eventLog: true,
    realTimeStats: false,
    smartNotifications: true,
    customShortcutsEnabled: true,
    encryptionEnabled: false,
    multiDeviceSync: false,
    realTimeSync: false,
    cloudSync: false,
    offlineMode: false,
    batchMode: false,
    pollMode: false,
    taskMode: false,
    workflowMode: false,
    quizMode: false,
    studyMode: false,
    productivityMode: false,
    tutorialMode: false,
    learningMode: false,
    dictationMode: false,
    translationMode: false,
    readingMode: false,
    locationSharing: false,
    calendarIntegration: false,
    pluginSystem: false,
    sessionRecording: false,
  },
}

export function useConsolidatedState() {
  const [state, setState] = useState<ConsolidatedChatState>(defaultState)

  const updateState = useCallback((updates: Partial<ConsolidatedChatState>) => {
    setState((prev) => ({ ...prev, ...updates }))
  }, [])

  const updateUI = useCallback((updates: Partial<ConsolidatedChatState['ui']>) => {
    setState((prev) => ({
      ...prev,
      ui: { ...prev.ui, ...updates },
    }))
  }, [])

  const updateDisplay = useCallback((updates: Partial<ConsolidatedChatState['display']>) => {
    setState((prev) => ({
      ...prev,
      display: { ...prev.display, ...updates },
    }))
  }, [])

  const updateSearch = useCallback((updates: Partial<ConsolidatedChatState['search']>) => {
    setState((prev) => ({
      ...prev,
      search: { ...prev.search, ...updates },
    }))
  }, [])

  const updateFeatures = useCallback((updates: Partial<ConsolidatedChatState['features']>) => {
    setState((prev) => ({
      ...prev,
      features: { ...prev.features, ...updates },
    }))
  }, [])

  const toggleUI = useCallback((key: keyof ConsolidatedChatState['ui']) => {
    setState((prev) => ({
      ...prev,
      ui: { ...prev.ui, [key]: !prev.ui[key] },
    }))
  }, [])

  const toggleFeature = useCallback((key: keyof ConsolidatedChatState['features']) => {
    setState((prev) => ({
      ...prev,
      features: { ...prev.features, [key]: !prev.features[key] },
    }))
  }, [])

  const addMessage = useCallback((message: { role: 'user' | 'assistant'; content: string }) => {
    setState((prev) => ({
      ...prev,
      messages: [
        ...prev.messages,
        {
          id: Date.now().toString(),
          ...message,
          timestamp: Date.now(),
        },
      ],
    }))
  }, [])

  const clearMessages = useCallback(() => {
    setState((prev) => ({ ...prev, messages: [] }))
  }, [])

  return {
    state,
    updateState,
    updateUI,
    updateDisplay,
    updateSearch,
    updateFeatures,
    toggleUI,
    toggleFeature,
    addMessage,
    clearMessages,
  }
}




