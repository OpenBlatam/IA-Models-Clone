'use client'

import { motion } from 'framer-motion'
import { Sparkles, Image, MessageSquare, TrendingUp, Brain, Zap } from 'lucide-react'
import { cn } from '@/lib/cn'

const suggestions = [
  {
    icon: MessageSquare,
    text: 'Un modelo para análisis de sentimientos en español',
    category: 'NLP',
    color: 'from-blue-500 to-cyan-500',
  },
  {
    icon: Image,
    text: 'Clasificador de imágenes para detectar objetos',
    category: 'Visión',
    color: 'from-purple-500 to-pink-500',
  },
  {
    icon: TrendingUp,
    text: 'Modelo de predicción de series temporales',
    category: 'Time Series',
    color: 'from-green-500 to-emerald-500',
  },
  {
    icon: Brain,
    text: 'Generador de texto con GPT',
    category: 'Generativo',
    color: 'from-orange-500 to-red-500',
  },
  {
    icon: Zap,
    text: 'Clasificador de spam en emails',
    category: 'Clasificación',
    color: 'from-yellow-500 to-amber-500',
  },
  {
    icon: Sparkles,
    text: 'Traductor automático español-inglés',
    category: 'NLP',
    color: 'from-indigo-500 to-purple-500',
  },
]

interface SuggestionsProps {
  onSelect: (text: string) => void
  className?: string
}

export default function Suggestions({ onSelect, className }: SuggestionsProps) {
  return (
    <div className={cn('space-y-3', className)}>
      <p className="text-sm font-medium text-slate-400 mb-4">Sugerencias rápidas:</p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {suggestions.map((suggestion, index) => {
          const Icon = suggestion.icon
          return (
            <motion.button
              key={index}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              onClick={() => onSelect(suggestion.text)}
              className="group relative p-4 bg-slate-800/50 hover:bg-slate-700/50 border border-slate-700 rounded-lg transition-all text-left overflow-hidden"
            >
              <div className={`absolute inset-0 bg-gradient-to-r ${suggestion.color} opacity-0 group-hover:opacity-10 transition-opacity`} />
              <div className="relative flex items-start gap-3">
                <div className={`p-2 rounded-lg bg-gradient-to-r ${suggestion.color} flex-shrink-0`}>
                  <Icon className="w-4 h-4 text-white" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium text-slate-400 mb-1">{suggestion.category}</p>
                  <p className="text-sm text-slate-200 group-hover:text-white transition-colors">
                    {suggestion.text}
                  </p>
                </div>
              </div>
            </motion.button>
          )
        })}
      </div>
    </div>
  )
}


