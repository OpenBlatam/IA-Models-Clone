import { useState, useCallback } from 'react'
import { Model } from '@/store/modelStore'

export interface ChatState {
  input: string
  isLoading: boolean
  messages: Array<{ id: string; role: 'user' | 'assistant'; content: string; timestamp: number }>
  showPreview: boolean
  showHistory: boolean
  showComparator: boolean
  selectedModels: Model[]
  validation: any
  previewSpec: any
  modelHistory: any[]
}

export interface ChatUIState {
  showTour: boolean
  showMetrics: boolean
  showProactive: boolean
  showStats: boolean
  viewMode: 'normal' | 'compact' | 'comfortable'
  theme: 'dark' | 'light' | 'auto'
  fontSize: 'small' | 'medium' | 'large'
  autoScroll: boolean
}

export interface ChatFeatureFlags {
  useBulkChatMode: boolean
  voiceInputEnabled: boolean
  voiceOutputEnabled: boolean
  readMode: boolean
  presentationMode: boolean
  collaborationMode: boolean
  devMode: boolean
  accessibilityMode: boolean
}

/**
 * Custom hook for chat state management
 * Centralizes all chat-related state
 */
export function useChatState() {
  const [state, setState] = useState<ChatState>({
    input: '',
    isLoading: false,
    messages: [],
    showPreview: false,
    showHistory: false,
    showComparator: false,
    selectedModels: [],
    validation: null,
    previewSpec: null,
    modelHistory: [],
  })

  const [uiState, setUIState] = useState<ChatUIState>({
    showTour: false,
    showMetrics: false,
    showProactive: false,
    showStats: false,
    viewMode: 'normal',
    theme: 'dark',
    fontSize: 'medium',
    autoScroll: true,
  })

  const [featureFlags, setFeatureFlags] = useState<ChatFeatureFlags>({
    useBulkChatMode: true,
    voiceInputEnabled: false,
    voiceOutputEnabled: false,
    readMode: false,
    presentationMode: false,
    collaborationMode: false,
    devMode: false,
    accessibilityMode: false,
  })

  const updateState = useCallback((updates: Partial<ChatState>) => {
    setState((prev) => ({ ...prev, ...updates }))
  }, [])

  const updateUIState = useCallback((updates: Partial<ChatUIState>) => {
    setUIState((prev) => ({ ...prev, ...updates }))
  }, [])

  const updateFeatureFlags = useCallback((updates: Partial<ChatFeatureFlags>) => {
    setFeatureFlags((prev) => ({ ...prev, ...updates }))
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
    uiState,
    featureFlags,
    updateState,
    updateUIState,
    updateFeatureFlags,
    addMessage,
    clearMessages,
  }
}

