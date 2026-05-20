/**
 * Basic Example - ChatInterface
 * Ejemplo básico de uso del ChatInterface refactorizado
 */

'use client'

import React from 'react'
import { ChatProvider, SettingsProvider, ThemeProvider } from '../contexts'
import { useChatState, useChatActions } from '../hooks'
import { MessageList, InputArea, Toolbar } from '../components'
import { useSearchAndFilters } from '../hooks/useSearchAndFilters'
import { useSettings } from '../hooks/useSettings'

function ChatInterfaceBasic() {
  const chatState = useChatState()
  const chatActions = useChatActions()
  const search = useSearchAndFilters(chatState.state.messages)
  const settings = useSettings()

  const handleSend = async () => {
    if (!chatState.state.input.trim() || chatState.state.isLoading) return

    const userMessage = chatState.state.input.trim()
    chatState.updateState({ input: '', isLoading: true })
    chatState.addMessage({ role: 'user', content: userMessage })

    try {
      const result = await chatActions.handleSendMessage(userMessage)
      if (result.success) {
        chatState.addMessage({
          role: 'assistant',
          content: `Respuesta: ${result.model?.name || 'Modelo creado'}`,
        })
      }
    } catch (error) {
      chatState.addMessage({
        role: 'assistant',
        content: 'Error al procesar el mensaje',
      })
    } finally {
      chatState.updateState({ isLoading: false })
    }
  }

  return (
    <div className="chat-interface-basic">
      <Toolbar
        searchQuery={search.searchQuery}
        onSearchChange={search.setSearchQuery}
        showFilters={false}
        onToggleFilters={() => {}}
        viewMode={settings.viewMode}
        onViewModeChange={(mode) => settings.updateSetting('viewMode', mode)}
      />

      <div className="chat-interface-basic__messages">
        <MessageList
          messages={search.filteredMessages}
          viewMode={settings.viewMode}
          searchQuery={search.searchQuery}
          highlightSearch={search.highlightSearch}
        />
      </div>

      <div className="chat-interface-basic__input">
        <InputArea
          value={chatState.state.input}
          onChange={(value) => chatState.updateState({ input: value })}
          onSend={handleSend}
          isLoading={chatState.state.isLoading}
        />
      </div>
    </div>
  )
}

export default function BasicExample() {
  return (
    <ThemeProvider>
      <SettingsProvider>
        <ChatProvider>
          <ChatInterfaceBasic />
        </ChatProvider>
      </SettingsProvider>
    </ThemeProvider>
  )
}




