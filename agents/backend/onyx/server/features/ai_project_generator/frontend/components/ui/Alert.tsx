'use client'

import { useCallback } from 'react'
import clsx from 'clsx'
import { AlertCircle, CheckCircle, Info, X, XCircle } from 'lucide-react'

type AlertVariant = 'success' | 'error' | 'warning' | 'info'

interface AlertProps {
  variant: AlertVariant
  title?: string
  message: string
  onDismiss?: () => void
  className?: string
}

const Alert = ({ variant, title, message, onDismiss, className }: AlertProps) => {
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if ((e.key === 'Enter' || e.key === ' ') && onDismiss) {
        e.preventDefault()
        onDismiss()
      }
    },
    [onDismiss]
  )

  const variantClasses = {
    success: 'bg-green-50 border-green-200 text-green-700',
    error: 'bg-red-50 border-red-200 text-red-700',
    warning: 'bg-yellow-50 border-yellow-200 text-yellow-700',
    info: 'bg-blue-50 border-blue-200 text-blue-700',
  }

  const iconClasses = {
    success: 'text-green-600',
    error: 'text-red-600',
    warning: 'text-yellow-600',
    info: 'text-blue-600',
  }

  const icons = {
    success: CheckCircle,
    error: XCircle,
    warning: AlertCircle,
    info: Info,
  }

  const Icon = icons[variant]

  return (
    <div
      className={clsx(
        'flex items-start gap-3 p-4 border rounded-lg',
        variantClasses[variant],
        className
      )}
      role="alert"
      aria-live="polite"
    >
      <Icon className={clsx('w-5 h-5 flex-shrink-0 mt-0.5', iconClasses[variant])} aria-hidden="true" />
      <div className="flex-1">
        {title && <p className="font-medium mb-1">{title}</p>}
        <p className="text-sm">{message}</p>
      </div>
      {onDismiss && (
        <button
          onClick={onDismiss}
          onKeyDown={handleKeyDown}
          className="text-current opacity-70 hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-current rounded"
          tabIndex={0}
          aria-label="Dismiss alert"
        >
          <X className="w-4 h-4" aria-hidden="true" />
        </button>
      )}
    </div>
  )
}

export default Alert

