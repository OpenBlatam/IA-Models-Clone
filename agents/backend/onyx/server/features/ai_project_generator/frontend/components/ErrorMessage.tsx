'use client'

import { AlertCircle } from 'lucide-react'
import clsx from 'clsx'

interface ErrorMessageProps {
  message: string
  className?: string
  onDismiss?: () => void
}

const ErrorMessage = ({ message, className, onDismiss }: ErrorMessageProps) => {
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.key === 'Enter' || e.key === ' ') && onDismiss) {
      e.preventDefault()
      onDismiss()
    }
  }

  return (
    <div
      className={clsx(
        'flex items-start gap-3 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700',
        className
      )}
      role="alert"
      aria-live="polite"
    >
      <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" aria-hidden="true" />
      <div className="flex-1">
        <p className="font-medium">Error</p>
        <p className="text-sm">{message}</p>
      </div>
      {onDismiss && (
        <button
          onClick={onDismiss}
          onKeyDown={handleKeyDown}
          className="text-red-600 hover:text-red-800 focus:outline-none focus:ring-2 focus:ring-red-500 rounded"
          tabIndex={0}
          aria-label="Dismiss error message"
        >
          ×
        </button>
      )}
    </div>
  )
}

export default ErrorMessage

