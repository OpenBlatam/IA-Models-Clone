/**
 * Componente ErrorDisplay
 * ========================
 * 
 * Componente para mostrar errores de forma amigable
 */

'use client'

import React from 'react'
import { AlertCircle, RefreshCw, X } from 'lucide-react'
import { motion } from 'framer-motion'
import Button from './Button'
import { getFriendlyErrorMessage, isRecoverableError } from '@/lib/utils/errorBoundaryUtils'

export interface ErrorDisplayProps {
  error: Error
  onRetry?: () => void
  onDismiss?: () => void
  showDetails?: boolean
  className?: string
}

export default function ErrorDisplay({
  error,
  onRetry,
  onDismiss,
  showDetails = false,
  className = ''
}: ErrorDisplayProps) {
  const friendlyMessage = getFriendlyErrorMessage(error)
  const recoverable = isRecoverableError(error)

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className={`
        rounded-lg border border-red-200 dark:border-red-800
        bg-red-50 dark:bg-red-900/20 p-4
        ${className}
      `}
    >
      <div className="flex items-start gap-3">
        <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
        
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-semibold text-red-900 dark:text-red-100 mb-1">
            Error
          </h3>
          <p className="text-sm text-red-800 dark:text-red-200 mb-2">
            {friendlyMessage}
          </p>

          {showDetails && (
            <details className="mt-2">
              <summary className="text-xs text-red-700 dark:text-red-300 cursor-pointer">
                Detalles técnicos
              </summary>
              <pre className="mt-2 text-xs text-red-600 dark:text-red-400 bg-red-100 dark:bg-red-900/30 p-2 rounded overflow-auto">
                {error.name}: {error.message}
                {error.stack && `\n\n${error.stack}`}
              </pre>
            </details>
          )}

          <div className="flex gap-2 mt-3">
            {recoverable && onRetry && (
              <Button
                variant="primary"
                size="sm"
                onClick={onRetry}
                className="flex items-center gap-1.5"
              >
                <RefreshCw className="w-4 h-4" />
                Reintentar
              </Button>
            )}
            {onDismiss && (
              <Button
                variant="secondary"
                size="sm"
                onClick={onDismiss}
              >
                Cerrar
              </Button>
            )}
          </div>
        </div>

        {onDismiss && (
          <button
            onClick={onDismiss}
            className="flex-shrink-0 text-red-400 hover:text-red-600 dark:hover:text-red-400 transition-colors"
            aria-label="Cerrar"
          >
            <X className="w-5 h-5" />
          </button>
        )}
      </div>
    </motion.div>
  )
}






