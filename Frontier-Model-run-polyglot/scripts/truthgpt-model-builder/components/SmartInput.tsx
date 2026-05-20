'use client'

import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Send, Loader2, Sparkles, X } from 'lucide-react'
import { useDebounce } from '@/lib/hooks'
import AutoComplete from './AutoComplete'
import ValidationBadge from './ValidationBadge'

interface SmartInputProps {
  value: string
  onChange: (value: string) => void
  onSubmit: (value: string) => void
  isLoading?: boolean
  placeholder?: string
  onValidationChange?: (validation: any) => void
  showSuggestions?: boolean
}

export default function SmartInput({
  value,
  onChange,
  onSubmit,
  isLoading = false,
  placeholder = 'Describe tu modelo de IA... (Ctrl+Enter para enviar)',
  onValidationChange,
  showSuggestions = true,
}: SmartInputProps) {
  const [isFocused, setIsFocused] = useState(false)
  const [showAutoComplete, setShowAutoComplete] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const debouncedValue = useDebounce(value, 500)

  useEffect(() => {
    // Auto-resize textarea
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`
    }
  }, [value])

  const handleSubmit = () => {
    if (value.trim() && !isLoading) {
      onSubmit(value.trim())
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      handleSubmit()
    } else if (e.key === 'Escape') {
      setIsFocused(false)
      setShowAutoComplete(false)
    }
  }

  return (
    <div className="relative">
      <div
        className={`relative transition-all duration-200 ${
          isFocused
            ? 'ring-2 ring-blue-500 ring-opacity-50'
            : 'ring-1 ring-slate-700'
        } rounded-lg bg-slate-800/50 border border-slate-700`}
      >
        {/* Input area */}
        <div className="flex items-start gap-2 p-4">
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={value}
              onChange={(e) => onChange(e.target.value)}
              onFocus={() => {
                setIsFocused(true)
                setShowAutoComplete(showSuggestions && value.length > 0)
              }}
              onBlur={() => {
                // Delay to allow click on suggestions
                setTimeout(() => setIsFocused(false), 200)
              }}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              className="w-full bg-transparent text-white placeholder-slate-400 resize-none focus:outline-none min-h-[60px] max-h-[200px] text-sm"
              rows={1}
            />

            {/* Character counter */}
            <div className="absolute bottom-2 right-2 text-xs text-slate-500">
              {value.length}/1000
            </div>
          </div>

          {/* Submit button */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleSubmit}
            disabled={!value.trim() || isLoading}
            className="flex items-center justify-center w-10 h-10 rounded-lg bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 disabled:cursor-not-allowed transition-colors"
            title="Enviar (Ctrl+Enter)"
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 text-white animate-spin" />
            ) : (
              <Send className="w-5 h-5 text-white" />
            )}
          </motion.button>
        </div>

        {/* Auto-complete overlay */}
        {showAutoComplete && isFocused && showSuggestions && (
          <div className="absolute bottom-full left-0 right-0 mb-2">
            <AutoComplete
              input={value}
              onSelect={(suggestion) => {
                onChange(suggestion)
                setShowAutoComplete(false)
              }}
              onClose={() => setShowAutoComplete(false)}
            />
          </div>
        )}
      </div>

      {/* Validation badge */}
      {debouncedValue && (
        <div className="mt-2">
          <ValidationBadge
            description={debouncedValue}
            onValidationChange={onValidationChange}
          />
        </div>
      )}

      {/* Quick tips */}
      {isFocused && value.length === 0 && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-2 p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg"
        >
          <div className="flex items-start gap-2">
            <Sparkles className="w-4 h-4 text-blue-400 mt-0.5" />
            <div className="text-xs text-slate-300">
              <p className="font-semibold mb-1">Tips:</p>
              <ul className="space-y-1 list-disc list-inside text-slate-400">
                <li>Sé específico sobre el tipo de modelo que necesitas</li>
                <li>Menciona el dominio (NLP, visión, series temporales, etc.)</li>
                <li>Indica el nivel de complejidad deseado</li>
                <li>Usa Ctrl+Enter para enviar rápidamente</li>
              </ul>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  )
}


