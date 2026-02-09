'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { FileText, X, RotateCcw, Clock } from 'lucide-react'
import { getDraft, clearDraft, hasRecentDraft, type Draft } from '@/lib/auto-save'
import { format } from 'date-fns'
import { toast } from 'react-hot-toast'

interface DraftRecoveryProps {
  onRestore: (draft: Draft) => void
}

export default function DraftRecovery({ onRestore }: DraftRecoveryProps) {
  const [draft, setDraft] = useState<Draft | null>(null)
  const [show, setShow] = useState(false)

  useEffect(() => {
    const checkDraft = () => {
      if (hasRecentDraft(30)) {
        const savedDraft = getDraft()
        if (savedDraft && savedDraft.input.trim().length > 0) {
          setDraft(savedDraft)
          setShow(true)
        }
      }
    }

    checkDraft()
  }, [])

  const handleRestore = () => {
    if (draft) {
      onRestore(draft)
      clearDraft()
      setShow(false)
      toast.success('Borrador restaurado')
    }
  }

  const handleDismiss = () => {
    setShow(false)
    toast('Borrador descartado', { icon: 'ℹ️' })
  }

  const handleClear = () => {
    clearDraft()
    setShow(false)
    toast.success('Borrador eliminado')
  }

  if (!show || !draft) return null

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        className="mb-4 bg-blue-500/10 border border-blue-500/30 rounded-lg p-4"
      >
        <div className="flex items-start gap-3">
          <FileText className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <h3 className="text-sm font-semibold text-white">Borrador encontrado</h3>
              <span className="text-xs text-slate-400 flex items-center gap-1">
                <Clock className="w-3 h-3" />
                {format(draft.timestamp, 'HH:mm')}
              </span>
            </div>
            <p className="text-sm text-slate-300 mb-3 line-clamp-2">
              {draft.input.substring(0, 100)}...
            </p>
            <div className="flex items-center gap-2">
              <button
                onClick={handleRestore}
                className="flex items-center gap-2 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm transition-colors"
              >
                <RotateCcw className="w-4 h-4" />
                Restaurar
              </button>
              <button
                onClick={handleClear}
                className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg text-sm transition-colors"
              >
                Eliminar
              </button>
              <button
                onClick={handleDismiss}
                className="p-1.5 hover:bg-slate-700 rounded-lg transition-colors"
                title="Descartar"
              >
                <X className="w-4 h-4 text-slate-400" />
              </button>
            </div>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  )
}


