'use client'

import React from 'react'
import { CheckCircle, XCircle, Clock, AlertCircle } from 'lucide-react'
import { Badge } from '../ui'
import { clsx } from 'clsx'

interface StatusIndicatorProps {
  status: 'success' | 'error' | 'pending' | 'warning'
  label?: string
  size?: 'sm' | 'md' | 'lg'
  showLabel?: boolean
  className?: string
}

const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  status,
  label,
  size = 'md',
  showLabel = true,
  className,
}) => {
  const icons = {
    success: CheckCircle,
    error: XCircle,
    pending: Clock,
    warning: AlertCircle,
  }

  const colors = {
    success: 'text-green-600',
    error: 'text-red-600',
    pending: 'text-yellow-600',
    warning: 'text-orange-600',
  }

  const bgColors = {
    success: 'bg-green-100',
    error: 'bg-red-100',
    pending: 'bg-yellow-100',
    warning: 'bg-orange-100',
  }

  const labels = {
    success: 'Success',
    error: 'Error',
    pending: 'Pending',
    warning: 'Warning',
  }

  const sizes = {
    sm: 'w-4 h-4',
    md: 'w-5 h-5',
    lg: 'w-6 h-6',
  }

  const Icon = icons[status]
  const displayLabel = label || labels[status]

  return (
    <div className={clsx('flex items-center gap-2', className)}>
      <div className={clsx('p-1 rounded-full', bgColors[status])}>
        <Icon className={clsx(sizes[size], colors[status])} />
      </div>
      {showLabel && (
        <span className={clsx('text-sm font-medium', colors[status])}>
          {displayLabel}
        </span>
      )}
    </div>
  )
}

export default StatusIndicator




