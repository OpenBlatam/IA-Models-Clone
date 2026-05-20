'use client'

import { motion } from 'framer-motion'
import { CheckCircle, AlertCircle, Info } from 'lucide-react'

interface ValidationResult {
  isValid: boolean
  message: string
  type: 'success' | 'warning' | 'error' | 'info'
}

interface ValidationBadgeProps {
  validation: ValidationResult | null
}

export default function ValidationBadge({ validation }: ValidationBadgeProps) {
  if (!validation) return null

  const icons = {
    success: CheckCircle,
    warning: AlertCircle,
    error: AlertCircle,
    info: Info,
  }

  const colors = {
    success: 'text-green-400 bg-green-400/10 border-green-400/20',
    warning: 'text-yellow-400 bg-yellow-400/10 border-yellow-400/20',
    error: 'text-red-400 bg-red-400/10 border-red-400/20',
    info: 'text-blue-400 bg-blue-400/10 border-blue-400/20',
  }

  const Icon = icons[validation.type]

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -10 }}
      className={`flex items-center gap-2 px-3 py-2 rounded-lg border ${colors[validation.type]}`}
    >
      <Icon className="w-4 h-4" />
      <p className="text-sm">{validation.message}</p>
    </motion.div>
  )
}


