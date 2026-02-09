/**
 * InputArea Component
 * Complete input area with text input, voice input, and quick actions
 */

'use client'

import React, { useState, useCallback, useRef } from 'react'
import { Send, Mic, Volume2, FileText, Zap } from 'lucide-react'
import { motion } from 'framer-motion'
import { useVoiceFeatures } from '../hooks/useVoiceFeatures'

interface InputAreaProps {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  isLoading?: boolean
  placeholder?: string
  voiceInputEnabled?: boolean
  voiceOutputEnabled?: boolean
  showQuickActions?: boolean
  maxLength?: number
  autoFocus?: boolean
}

export function InputArea({
  value,
  onChange,
  onSend,
  isLoading = false,
  placeholder = 'Escribe tu mensaje...',
  voiceInputEnabled: externalVoiceInputEnabled,
  voiceOutputEnabled: externalVoiceOutputEnabled,
  showQuickActions = true,
  maxLength = 10000,
  autoFocus = false,
}: InputAreaProps) {
  const [isFocused, setIsFocused] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  
  const voiceFeatures = useVoiceFeatures()
  const voiceInputEnabled = externalVoiceInputEnabled ?? voiceFeatures.voiceInputEnabled
  const voiceOutputEnabled = externalVoiceOutputEnabled ?? voiceFeatures.voiceOutputEnabled

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (!isLoading && value.trim()) {
        onSend()
      }
    }
  }, [value, isLoading, onSend])

  const handleVoiceToggle = useCallback(async () => {
    if (voiceFeatures.isRecording) {
      voiceFeatures.stopRecording()
    } else {
      try {
        await voiceFeatures.startRecording()
      } catch (error) {
        console.error('Error starting voice recording:', error)
      }
    }
  }, [voiceFeatures])

  const handleSendClick = useCallback(() => {
    if (!isLoading && value.trim()) {
      onSend()
    }
  }, [value, isLoading, onSend])

  // Auto-resize textarea
  React.useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`
    }
  }, [value])

  return (
    <div className={`input-area ${isFocused ? 'input-area--focused' : ''}`}>
      <div className="input-area__container">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => {
            if (e.target.value.length <= maxLength) {
              onChange(e.target.value)
            }
          }}
          onKeyDown={handleKeyDown}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          placeholder={placeholder}
          disabled={isLoading}
          rows={1}
          className="input-area__textarea"
          autoFocus={autoFocus}
          maxLength={maxLength}
        />
        
        <div className="input-area__actions">
          {voiceInputEnabled && (
            <button
              type="button"
              onClick={handleVoiceToggle}
              className={`input-area__button input-area__button--voice ${
                voiceFeatures.isRecording ? 'input-area__button--recording' : ''
              }`}
              title={voiceFeatures.isRecording ? 'Detener grabación' : 'Grabar voz'}
            >
              <Mic size={20} />
              {voiceFeatures.isRecording && (
                <span className="input-area__recording-indicator" />
              )}
            </button>
          )}

          {voiceOutputEnabled && (
            <button
              type="button"
              className="input-area__button"
              title="Reproducir respuesta"
            >
              <Volume2 size={20} />
            </button>
          )}

          <button
            type="button"
            onClick={handleSendClick}
            disabled={isLoading || !value.trim()}
            className="input-area__button input-area__button--send"
            title="Enviar mensaje"
          >
            {isLoading ? (
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
              >
                <Zap size={20} />
              </motion.div>
            ) : (
              <Send size={20} />
            )}
          </button>
        </div>
      </div>

      {showQuickActions && (
        <div className="input-area__quick-actions">
          <button type="button" className="input-area__quick-action">
            <FileText size={16} />
            <span>Plantilla</span>
          </button>
        </div>
      )}

      {maxLength && (
        <div className="input-area__counter">
          {value.length} / {maxLength}
        </div>
      )}

      {voiceFeatures.transcription && (
        <div className="input-area__transcription">
          Transcripción: {voiceFeatures.transcription}
        </div>
      )}
    </div>
  )
}

export default InputArea




