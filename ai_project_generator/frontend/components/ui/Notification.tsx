'use client'

import { ReactNode } from 'react'
import { X, CheckCircle, XCircle, AlertCircle, Info } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { cn } from '@/lib/utils'

interface NotificationProps {
  id: string
  title?: string
  message: string
  type?: 'success' | 'error' | 'warning' | 'info'
  duration?: number
  onClose: (id: string) => void
  icon?: ReactNode
  action?: {
    label: string
    onClick: () => void
  }
}

const Notification = ({
  id,
  title,
  message,
  type = 'info',
  duration = 5000,
  onClose,
  icon,
  action,
}: NotificationProps) => {
  const typeConfig = {
    success: {
      icon: <CheckCircle className="w-5 h-5" />,
      bgColor: 'bg-green-50',
      borderColor: 'border-green-200',
      textColor: 'text-green-800',
      iconColor: 'text-green-600',
    },
    error: {
      icon: <XCircle className="w-5 h-5" />,
      bgColor: 'bg-red-50',
      borderColor: 'border-red-200',
      textColor: 'text-red-800',
      iconColor: 'text-red-600',
    },
    warning: {
      icon: <AlertCircle className="w-5 h-5" />,
      bgColor: 'bg-yellow-50',
      borderColor: 'border-yellow-200',
      textColor: 'text-yellow-800',
      iconColor: 'text-yellow-600',
    },
    info: {
      icon: <Info className="w-5 h-5" />,
      bgColor: 'bg-blue-50',
      borderColor: 'border-blue-200',
      textColor: 'text-blue-800',
      iconColor: 'text-blue-600',
    },
  }

  const config = typeConfig[type]

  return (
    <motion.div
      initial={{ opacity: 0, y: -20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95, transition: { duration: 0.2 } }}
      className={cn(
        'rounded-lg border p-4 shadow-lg max-w-sm',
        config.bgColor,
        config.borderColor
      )}
      role="alert"
      aria-live="polite"
    >
      <div className="flex items-start gap-3">
        <div className={cn('flex-shrink-0', config.iconColor)}>
          {icon || config.icon}
        </div>
        <div className="flex-1 min-w-0">
          {title && (
            <h4 className={cn('font-semibold mb-1', config.textColor)}>{title}</h4>
          )}
          <p className={cn('text-sm', config.textColor)}>{message}</p>
          {action && (
            <button
              onClick={action.onClick}
              className={cn('mt-2 text-sm font-medium underline', config.textColor)}
            >
              {action.label}
            </button>
          )}
        </div>
        <button
          onClick={() => onClose(id)}
          className={cn('flex-shrink-0 hover:opacity-70', config.textColor)}
          aria-label="Close notification"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </motion.div>
  )
}

export default Notification

