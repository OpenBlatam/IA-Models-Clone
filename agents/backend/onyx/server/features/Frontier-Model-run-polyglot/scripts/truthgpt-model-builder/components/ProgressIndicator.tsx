'use client'

import { motion } from 'framer-motion'
import { CheckCircle2, Loader2 } from 'lucide-react'

interface ProgressIndicatorProps {
  progress: number
  currentStep: string
  status: 'creating' | 'completed' | 'failed'
}

export default function ProgressIndicator({ progress, currentStep, status }: ProgressIndicatorProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-sm text-slate-400">{currentStep}</span>
        <span className="text-sm font-medium text-slate-300">{progress}%</span>
      </div>
      <div className="w-full bg-slate-700 rounded-full h-2 overflow-hidden">
        <motion.div
          className={`h-full ${
            status === 'completed' 
              ? 'bg-gradient-to-r from-green-500 to-emerald-500'
              : status === 'failed'
              ? 'bg-gradient-to-r from-red-500 to-rose-500'
              : 'bg-gradient-to-r from-purple-500 to-pink-500'
          }`}
          initial={{ width: 0 }}
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
        />
      </div>
      {status === 'completed' && (
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          className="flex items-center gap-2 text-green-400 text-sm"
        >
          <CheckCircle2 className="w-4 h-4" />
          <span>Completado</span>
        </motion.div>
      )}
    </div>
  )
}


