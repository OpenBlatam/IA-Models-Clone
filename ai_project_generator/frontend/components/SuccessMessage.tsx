'use client'

import { CheckCircle } from 'lucide-react'
import clsx from 'clsx'

interface SuccessMessageProps {
  message: string
  className?: string
  onDismiss?: () => void
}

const SuccessMessage = ({ message, className, onDismiss }: SuccessMessageProps) => {
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.key === 'Enter' || e.key === ' ') && onDismiss) {
      e.preventDefault()
      onDismiss()
    }
  }

  return (
    <div
      className={clsx(
        'flex items-start gap-3 p-4 bg-green-50 border border-green-200 rounded-lg text-green-700',
        className
      )}
      role="alert"
      aria-live="polite"
    >
      <CheckCircle className="w-5 h-5 flex-shrink-0 mt-0.5" aria-hidden="true" />
      <div className="flex-1">
        <p className="font-medium">Success</p>
        <p className="text-sm">{message}</p>
      </div>
      {onDismiss && (
        <button
          onClick={onDismiss}
          onKeyDown={handleKeyDown}
          className="text-green-600 hover:text-green-800 focus:outline-none focus:ring-2 focus:ring-green-500 rounded"
          tabIndex={0}
          aria-label="Dismiss success message"
        >
          ×
        </button>
      )}
    </div>
  )
}

export default SuccessMessage

