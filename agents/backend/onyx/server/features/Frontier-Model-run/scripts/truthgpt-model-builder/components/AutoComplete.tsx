'use client'

import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Sparkles, ArrowUp } from 'lucide-react'

interface Suggestion {
  text: string
  category: string
  icon: string
}

const autocompleteSuggestions: Record<string, Suggestion[]> = {
  'análisis': [
    { text: 'análisis de sentimientos', category: 'NLP', icon: '📊' },
    { text: 'análisis de datos', category: 'Análisis', icon: '📈' },
    { text: 'análisis de imágenes', category: 'Visión', icon: '🖼️' },
  ],
  'modelo': [
    { text: 'modelo de clasificación', category: 'Clasificación', icon: '🏷️' },
    { text: 'modelo de predicción', category: 'Regresión', icon: '🔮' },
    { text: 'modelo generativo', category: 'Generativo', icon: '✨' },
  ],
  'para': [
    { text: 'para detectar objetos', category: 'Visión', icon: '👁️' },
    { text: 'para predecir precios', category: 'Regresión', icon: '💰' },
    { text: 'para generar texto', category: 'Generativo', icon: '📝' },
  ],
  'clasificar': [
    { text: 'clasificar imágenes', category: 'Visión', icon: '🖼️' },
    { text: 'clasificar texto', category: 'NLP', icon: '📄' },
    { text: 'clasificar spam', category: 'Clasificación', icon: '📧' },
  ],
}

interface AutoCompleteProps {
  input: string
  onSelect: (text: string) => void
  className?: string
}

export default function AutoComplete({ input, onSelect, className }: AutoCompleteProps) {
  const [suggestions, setSuggestions] = useState<Suggestion[]>([])
  const [selectedIndex, setSelectedIndex] = useState(0)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!input.trim()) {
      setSuggestions([])
      return
    }

    const words = input.toLowerCase().split(' ')
    const lastWord = words[words.length - 1]

    // Find matching suggestions
    const matches: Suggestion[] = []
    
    Object.entries(autocompleteSuggestions).forEach(([key, values]) => {
      if (lastWord.includes(key) || key.includes(lastWord)) {
        matches.push(...values)
      }
    })

    // Also check full phrase matches
    const fullPhrase = input.toLowerCase()
    Object.values(autocompleteSuggestions).flat().forEach(suggestion => {
      if (suggestion.text.includes(fullPhrase) && !matches.includes(suggestion)) {
        matches.push(suggestion)
      }
    })

    setSuggestions(matches.slice(0, 5))
    setSelectedIndex(0)
  }, [input])

  const handleSelect = (suggestion: Suggestion) => {
    const words = input.split(' ')
    words[words.length - 1] = suggestion.text
    onSelect(words.join(' '))
    setSuggestions([])
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (suggestions.length === 0) return

    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setSelectedIndex(prev => (prev + 1) % suggestions.length)
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setSelectedIndex(prev => (prev - 1 + suggestions.length) % suggestions.length)
    } else if (e.key === 'Enter' && selectedIndex >= 0) {
      e.preventDefault()
      handleSelect(suggestions[selectedIndex])
    } else if (e.key === 'Escape') {
      setSuggestions([])
    }
  }

  if (suggestions.length === 0) return null

  return (
    <div className={`relative ${className}`} ref={containerRef}>
      <AnimatePresence>
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          className="absolute bottom-full left-0 right-0 mb-2 bg-slate-800 border border-slate-700 rounded-lg shadow-xl overflow-hidden z-50"
        >
          {suggestions.map((suggestion, index) => (
            <button
              key={index}
              onClick={() => handleSelect(suggestion)}
              onMouseEnter={() => setSelectedIndex(index)}
              className={`w-full text-left px-4 py-3 flex items-center gap-3 transition-colors ${
                index === selectedIndex
                  ? 'bg-slate-700 text-white'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              <span className="text-xl">{suggestion.icon}</span>
              <div className="flex-1">
                <p className="text-sm font-medium">{suggestion.text}</p>
                <p className="text-xs text-slate-400">{suggestion.category}</p>
              </div>
              {index === selectedIndex && (
                <ArrowUp className="w-4 h-4 text-purple-400" />
              )}
            </button>
          ))}
        </motion.div>
      </AnimatePresence>
    </div>
  )
}


