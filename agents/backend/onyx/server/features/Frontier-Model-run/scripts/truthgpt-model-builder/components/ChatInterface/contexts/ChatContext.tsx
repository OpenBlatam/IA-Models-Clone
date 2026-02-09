/**
 * Chat Context Provider
 * Provides global chat state and actions
 */

'use client'

import React, { createContext, useContext, ReactNode } from 'react'
import { useChatState } from '../hooks/useChatState'
import { useChatActions } from '../hooks/useChatActions'

interface ChatContextType {
  // State from useChatState
  state: ReturnType<typeof useChatState>['state']
  uiState: ReturnType<typeof useChatState>['uiState']
  featureFlags: ReturnType<typeof useChatState>['featureFlags']
  
  // Actions
  updateState: ReturnType<typeof useChatState>['updateState']
  updateUIState: ReturnType<typeof useChatState>['updateUIState']
  addMessage: ReturnType<typeof useChatState>['addMessage']
  handleSendMessage: ReturnType<typeof useChatActions>['handleSendMessage']
  handlePreviewModel: ReturnType<typeof useChatActions>['handlePreviewModel']
}

const ChatContext = createContext<ChatContextType | null>(null)

interface ChatProviderProps {
  children: ReactNode
}

export function ChatProvider({ children }: ChatProviderProps) {
  const chatState = useChatState()
  const chatActions = useChatActions()

  const value: ChatContextType = {
    state: chatState.state,
    uiState: chatState.uiState,
    featureFlags: chatState.featureFlags,
    updateState: chatState.updateState,
    updateUIState: chatState.updateUIState,
    addMessage: chatState.addMessage,
    handleSendMessage: chatActions.handleSendMessage,
    handlePreviewModel: chatActions.handlePreviewModel,
  }

  return (
    <ChatContext.Provider value={value}>
      {children}
    </ChatContext.Provider>
  )
}

export function useChat(): ChatContextType {
  const context = useContext(ChatContext)
  if (!context) {
    throw new Error('useChat must be used within ChatProvider')
  }
  return context
}




