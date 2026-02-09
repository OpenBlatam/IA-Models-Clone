/**
 * Advanced Example - ChatInterface
 * Ejemplo avanzado con todas las características
 */

'use client'

import React, { useEffect } from 'react'
import { ChatProvider, SettingsProvider, ThemeProvider } from '../contexts'
import {
  useChatState,
  useChatActions,
  useMessageManagement,
  useSearchAndFilters,
  useSettings,
  useVoiceFeatures,
  useExportImport,
  useAccessibility,
  usePerformance,
  useNotifications,
  useTheming,
  useCollaboration,
  useKeyboardShortcuts,
  useVirtualization,
} from '../hooks'
import {
  MessageList,
  InputArea,
  Toolbar,
  Sidebar,
  FilterBar,
  LoadingSpinner,
  EmptyState,
  ErrorBoundary,
  ExportModal,
} from '../components'
import { trackMessageSent, setAnalyticsTracker } from '../utils/analyticsUtils'
import { safeAsync, handleError } from '../utils/errorHandling'

function ChatInterfaceAdvanced() {
  // Hooks principales
  const chatState = useChatState()
  const chatActions = useChatActions()
  const messageManagement = useMessageManagement(
    chatState.state.messages.map(m => m.id)
  )
  const search = useSearchAndFilters(chatState.state.messages)
  const settings = useSettings()

  // Hooks avanzados
  const voiceFeatures = useVoiceFeatures()
  const exportImport = useExportImport()
  const accessibility = useAccessibility()
  const performance = usePerformance()
  const notifications = useNotifications()
  const theming = useTheming()
  const collaboration = useCollaboration()
  const shortcuts = useKeyboardShortcuts(true)
  const virtualization = useVirtualization(chatState.state.messages, {
    itemHeight: 80,
    containerHeight: 600,
    overscan: 5,
  })

  // UI State
  const [showSidebar, setShowSidebar] = React.useState(false)
  const [showExportModal, setShowExportModal] = React.useState(false)
  const [showFilters, setShowFilters] = React.useState(false)

  // Setup analytics
  useEffect(() => {
    setAnalyticsTracker({
      track: (event) => console.log('[Analytics]', event),
      page: (name, properties) => console.log('[Page]', name, properties),
      identify: (userId, traits) => console.log('[Identify]', userId, traits),
      reset: () => console.log('[Reset]'),
    })
  }, [])

  // Setup keyboard shortcuts
  useEffect(() => {
    shortcuts.registerShortcut('send', {
      key: 'Enter',
      ctrl: true,
      handler: handleSend,
      description: 'Enviar mensaje',
    })

    shortcuts.registerShortcut('export', {
      key: 'e',
      ctrl: true,
      handler: () => setShowExportModal(true),
      description: 'Exportar mensajes',
    })

    return () => {
      shortcuts.unregisterShortcut('send')
      shortcuts.unregisterShortcut('export')
    }
  }, [shortcuts])

  // Handle send
  const handleSend = async () => {
    if (!chatState.state.input.trim() || chatState.state.isLoading) return

    const userMessage = chatState.state.input.trim()
    chatState.updateState({ input: '', isLoading: true })
    chatState.addMessage({ role: 'user', content: userMessage })

    await safeAsync(
      async () => {
        const result = await chatActions.handleSendMessage(userMessage)

        if (result.success) {
          chatState.addMessage({
            role: 'assistant',
            content: `Modelo "${result.model?.name}" creado exitosamente`,
          })

          // Track analytics
          trackMessageSent(userMessage.length, false, false)

          // Notify if enabled
          if (notifications.smartNotifications) {
            notifications.checkAndNotify({
              id: `msg-${Date.now()}`,
              content: `Modelo "${result.model?.name}" creado`,
              role: 'assistant',
            })
          }
        }
      },
      undefined,
      (error) => {
        handleError(error, 'ChatInterface')
        chatState.addMessage({
          role: 'assistant',
          content: 'Error al procesar el mensaje',
        })
      }
    )

    chatState.updateState({ isLoading: false })
  }

  // Performance tracking
  useEffect(() => {
    const stopMeasure = performance.measureRender('ChatInterfaceAdvanced')
    return stopMeasure
  }, [chatState.state.messages.length, performance])

  return (
    <ErrorBoundary>
      <div className={`chat-interface-advanced theme-${theming.theme}`}>
        <Toolbar
          searchQuery={search.searchQuery}
          onSearchChange={search.setSearchQuery}
          showFilters={showFilters}
          onToggleFilters={() => setShowFilters(!showFilters)}
          viewMode={settings.viewMode}
          onViewModeChange={(mode) => settings.updateSetting('viewMode', mode)}
          onExport={() => setShowExportModal(true)}
          onSettings={() => setShowSidebar(true)}
        />

        {showFilters && (
          <FilterBar
            filters={search.searchFilters}
            onFiltersChange={search.setSearchFilters}
            onClear={search.clearSearch}
            isOpen={showFilters}
            onToggle={() => setShowFilters(false)}
          />
        )}

        <div className="chat-interface-advanced__main">
          {chatState.state.messages.length === 0 ? (
            <EmptyState
              icon="messages"
              title="No hay mensajes"
              description="Comienza una conversación escribiendo un mensaje"
            />
          ) : (
            <div
              ref={virtualization.containerRef}
              className="chat-interface-advanced__messages"
              style={{ height: '600px', overflow: 'auto' }}
            >
              <div style={{ height: virtualization.totalHeight, position: 'relative' }}>
                <div style={{ transform: `translateY(${virtualization.offsetY}px)` }}>
                  {virtualization.visibleItems.map(index => {
                    const message = chatState.state.messages[index]
                    return (
                      <div key={message.id} style={{ height: '80px' }}>
                        {/* Render message */}
                      </div>
                    )
                  })}
                </div>
              </div>
            </div>
          )}

          {chatState.state.isLoading && (
            <LoadingSpinner text="Procesando..." />
          )}
        </div>

        <InputArea
          value={chatState.state.input}
          onChange={(value) => chatState.updateState({ input: value })}
          onSend={handleSend}
          isLoading={chatState.state.isLoading}
          voiceInputEnabled={voiceFeatures.voiceInputEnabled}
        />

        <Sidebar
          isOpen={showSidebar}
          onClose={() => setShowSidebar(false)}
        />

        <ExportModal
          isOpen={showExportModal}
          onClose={() => setShowExportModal(false)}
          onExport={async (format) => {
            const content = await exportImport.exportMessages(
              chatState.state.messages,
              format
            )
            exportImport.exportToFile(
              content,
              `chat-${Date.now()}.${format}`,
              `text/${format}`
            )
          }}
        />
      </div>
    </ErrorBoundary>
  )
}

export default function AdvancedExample() {
  return (
    <ThemeProvider>
      <SettingsProvider>
        <ChatProvider>
          <ChatInterfaceAdvanced />
        </ChatProvider>
      </SettingsProvider>
    </ThemeProvider>
  )
}




