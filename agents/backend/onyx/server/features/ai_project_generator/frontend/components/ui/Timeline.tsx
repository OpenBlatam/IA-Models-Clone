'use client'

import { ReactNode } from 'react'
import { cn } from '@/lib/utils'

interface TimelineItem {
  title: string
  description?: string
  time?: string
  icon?: ReactNode
  status?: 'completed' | 'pending' | 'failed'
}

interface TimelineProps {
  items: TimelineItem[]
  className?: string
}

const Timeline = ({ items, className }: TimelineProps) => {
  const getStatusColor = (status?: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-500'
      case 'failed':
        return 'bg-red-500'
      default:
        return 'bg-gray-300'
    }
  }

  return (
    <div className={cn('relative', className)}>
      <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-gray-200" />
      <div className="space-y-6">
        {items.map((item, index) => (
          <div key={index} className="relative flex gap-4">
            <div className="flex-shrink-0">
              <div
                className={cn(
                  'w-8 h-8 rounded-full flex items-center justify-center border-2 border-white',
                  getStatusColor(item.status)
                )}
              >
                {item.icon || (
                  <div className={cn('w-3 h-3 rounded-full', getStatusColor(item.status))} />
                )}
              </div>
            </div>
            <div className="flex-1 pb-6">
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="font-semibold text-gray-900">{item.title}</h3>
                  {item.description && (
                    <p className="text-sm text-gray-600 mt-1">{item.description}</p>
                  )}
                </div>
                {item.time && (
                  <span className="text-xs text-gray-500 whitespace-nowrap ml-4">{item.time}</span>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default Timeline

