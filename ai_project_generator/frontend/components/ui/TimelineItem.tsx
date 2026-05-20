'use client'

import { ReactNode } from 'react'
import { cn } from '@/lib/utils'

interface TimelineItemProps {
  title: string
  description?: string
  timestamp?: string
  icon?: ReactNode
  variant?: 'default' | 'success' | 'warning' | 'error'
  className?: string
  children?: ReactNode
}

const TimelineItem = ({
  title,
  description,
  timestamp,
  icon,
  variant = 'default',
  className,
  children,
}: TimelineItemProps) => {
  const variantClasses = {
    default: 'bg-primary-600',
    success: 'bg-green-600',
    warning: 'bg-yellow-600',
    error: 'bg-red-600',
  }

  return (
    <div className={cn('flex gap-4', className)}>
      <div className="flex flex-col items-center">
        <div
          className={cn(
            'w-10 h-10 rounded-full flex items-center justify-center text-white',
            variantClasses[variant]
          )}
        >
          {icon || <div className="w-2 h-2 rounded-full bg-white" />}
        </div>
        <div className="w-0.5 flex-1 bg-gray-300 mt-2" />
      </div>
      <div className="flex-1 pb-8">
        <div className="flex items-start justify-between mb-1">
          <h3 className="font-semibold text-gray-900">{title}</h3>
          {timestamp && <span className="text-sm text-gray-500">{timestamp}</span>}
        </div>
        {description && <p className="text-sm text-gray-600 mb-2">{description}</p>}
        {children}
      </div>
    </div>
  )
}

export default TimelineItem

