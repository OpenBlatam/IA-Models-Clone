'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useChatState } from './hooks/useChatState'
import { useChatActions } from './hooks/useChatActions'
import ChatInput from './ChatInput'
import ChatMessages from './ChatMessages'
import ModelStatus from '../ModelStatus'
import Suggestions from '../Suggestions'
import ModelPreview from '../ModelPreview'
import ModelHistory from '../ModelHistory'
import { useModelStore } from '@/store/modelStore'
import { toast } from 'react-hot-toast'

interface ChatInterfaceProps {
  devMode?: boolean
}

export default function ChatInterface({ devMode = false }: ChatInterfaceProps) {
  const { state, uiState, featureFlags, updateState, updateUIState, addMessage } = useChatState()
  const { handleSendMessage, handlePreviewModel } = useChatActions()
  const { models } = useModelStore()

  const handleSend = async () => {
    if (!state.input.trim() || state.isLoading) return

    const userMessage = state.input.trim()
    updateState({ input: '', isLoading: true })
    addMessage({ role: 'user', content: userMessage })

    const result = await handleSendMessage(userMessage)

    if (result.success) {
      addMessage({
        role: 'assistant',
        content: `Modelo "${result.model?.name}" creado exitosamente. Estado: ${result.model?.status}`,
      })
      toast.success('Modelo creado exitosamente')
    } else {
      addMessage({
        role: 'assistant',
        content: `Error: ${result.error}`,
      })
      toast.error(result.error || 'Error al crear el modelo')
    }

    updateState({ isLoading: false })
  }

  const handlePreview = async () => {
    if (!state.input.trim()) return

    const result = await handlePreviewModel(state.input)
    if (result.success) {
      updateState({ previewSpec: result.spec, showPreview: true })
    }
  }

  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto">
      <div className="flex-1 overflow-y-auto mb-4">
        <ChatMessages
          messages={state.messages}
          isLoading={state.isLoading}
          viewMode={uiState.viewMode}
        />
      </div>

      <div className="space-y-4">
        {state.isLoading && (
          <ModelStatus
            modelId={models[models.length - 1]?.id}
            status={models[models.length - 1]?.status || 'creating'}
          />
        )}

        <Suggestions
          onSelect={(suggestion) => updateState({ input: suggestion })}
        />

        <ChatInput
          value={state.input}
          onChange={(value) => updateState({ input: value })}
          onSend={handleSend}
          isLoading={state.isLoading}
        />
      </div>

      {state.showPreview && state.previewSpec && (
        <ModelPreview
          spec={state.previewSpec}
          onClose={() => updateState({ showPreview: false })}
        />
      )}

      {state.showHistory && (
        <ModelHistory
          onClose={() => updateState({ showHistory: false })}
          onSelectModel={(model) => {
            updateState({ showHistory: false, selectedModels: [model] })
          }}
        />
      )}
    </div>
  )
}

