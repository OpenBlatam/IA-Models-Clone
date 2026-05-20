'use client'

import { useEffect } from 'react'
import clsx from 'clsx'
import { CheckCircle, XCircle, AlertCircle, Info, X } from 'lucide-react'

type ToastVariant = 'success' | 'error' | 'warning' | 'info'

interface ToastProps {
  message: string
  variant?: ToastVariant
  duration?: number
  onClose: () => void
  className?: string
}

const Toast = ({ message, variant = 'info', duration = 5000, onClose, className }: ToastProps) => {
  useEffect(() => {
    if (duration > 0) {
      const timer = setTimeout(onClose, duration)
      return () => clearTimeout(timer)
    }
  }, [duration, onClose])

  const variantClasses = {
    success: 'bg-green-50 border-green-200 text-green-800',
    error: 'bg-red-50 border-red-200 text-red-800',
    warning: 'bg-yellow-50 border-yellow-200 text-yellow-800',
    info: 'bg-blue-50 border-blue-200 text-blue-800',
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
        'flex items-start gap-3 p-4 border rounded-lg shadow-lg min-w-[300px] max-w-md',
        variantClasses[variant],
        className
      )}
      role="alert"
      aria-live="polite"
    >
      <Icon className={clsx('w-5 h-5 flex-shrink-0 mt-0.5', iconClasses[variant])} aria-hidden="true" />
      <p className="flex-1 text-sm">{message}</p>
      <button
        onClick={onClose}
        className="text-current opacity-70 hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-current rounded"
        tabIndex={0}
        aria-label="Close toast"
      >
        <X className="w-4 h-4" aria-hidden="true" />
      </button>
    </div>
  )
}

export default Toast

