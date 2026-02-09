import React from 'react'
import { AlertCircle, CheckCircle, Info, AlertTriangle, X } from 'lucide-react'
import { clsx } from 'clsx'
import { Button } from './Button'

interface AlertProps {
  variant?: 'success' | 'error' | 'warning' | 'info'
  title?: string
  children: React.ReactNode
  onClose?: () => void
  className?: string
}

const Alert: React.FC<AlertProps> = ({
  variant = 'info',
  title,
  children,
  onClose,
  className,
}) => {
  const variants = {
    success: {
      bg: 'bg-green-50',
      border: 'border-green-200',
      text: 'text-green-800',
      icon: CheckCircle,
      iconColor: 'text-green-600',
    },
    error: {
      bg: 'bg-red-50',
      border: 'border-red-200',
      text: 'text-red-800',
      icon: AlertCircle,
      iconColor: 'text-red-600',
    },
    warning: {
      bg: 'bg-yellow-50',
      border: 'border-yellow-200',
      text: 'text-yellow-800',
      icon: AlertTriangle,
      iconColor: 'text-yellow-600',
    },
    info: {
      bg: 'bg-blue-50',
      border: 'border-blue-200',
      text: 'text-blue-800',
      icon: Info,
      iconColor: 'text-blue-600',
    },
  }

  const variantStyles = variants[variant]
  const Icon = variantStyles.icon

  return (
    <div
      className={clsx(
        'rounded-lg border p-4',
        variantStyles.bg,
        variantStyles.border,
        className
      )}
      role="alert"
    >
      <div className="flex items-start">
        <Icon className={clsx('w-5 h-5 mt-0.5', variantStyles.iconColor)} />
        <div className="ml-3 flex-1">
          {title && (
            <h3 className={clsx('text-sm font-medium mb-1', variantStyles.text)}>
              {title}
            </h3>
          )}
          <div className={clsx('text-sm', variantStyles.text)}>{children}</div>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className={clsx(
              'ml-4 flex-shrink-0 rounded-md p-1.5 transition-colors',
              variantStyles.text,
              'hover:bg-opacity-20'
            )}
            aria-label="Close alert"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  )
}

export default Alert




