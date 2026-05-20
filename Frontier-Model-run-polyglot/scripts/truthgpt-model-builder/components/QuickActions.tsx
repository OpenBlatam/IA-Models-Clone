'use client'

import { motion } from 'framer-motion'
import { Zap, Sparkles, TrendingUp, Code } from 'lucide-react'

const quickActions = [
  {
    icon: Zap,
    label: 'Rápido',
    description: 'Modelo simple y rápido',
    prompt: 'Un modelo simple y rápido para clasificación básica',
  },
  {
    icon: TrendingUp,
    label: 'Precisión',
    description: 'Alta precisión',
    prompt: 'Un modelo muy preciso para clasificación con alta exactitud',
  },
  {
    icon: Sparkles,
    label: 'Avanzado',
    description: 'Modelo complejo',
    prompt: 'Un modelo complejo y avanzado con arquitectura profunda',
  },
  {
    icon: Code,
    label: 'Personalizado',
    description: 'Editar manualmente',
    prompt: '',
  },
]

interface QuickActionsProps {
  onSelect: (prompt: string) => void
}

export default function QuickActions({ onSelect }: QuickActionsProps) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {quickActions.map((action, index) => {
        const Icon = action.icon
        return (
          <motion.button
            key={index}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.1 }}
            onClick={() => action.prompt && onSelect(action.prompt)}
            className="p-4 bg-slate-800/50 hover:bg-slate-700/50 border border-slate-700 rounded-lg transition-all text-center group"
          >
            <div className="w-10 h-10 mx-auto mb-2 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 flex items-center justify-center group-hover:scale-110 transition-transform">
              <Icon className="w-5 h-5 text-white" />
            </div>
            <p className="text-sm font-medium text-white mb-1">{action.label}</p>
            <p className="text-xs text-slate-400">{action.description}</p>
          </motion.button>
        )
      })}
    </div>
  )
}


