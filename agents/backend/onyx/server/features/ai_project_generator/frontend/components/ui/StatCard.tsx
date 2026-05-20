'use client'

import { ReactNode } from 'react'
import { TrendingUp, TrendingDown } from 'lucide-react'
import { cn } from '@/lib/utils'

interface StatCardProps {
  title: string
  value: string | number
  change?: {
    value: number
    type: 'increase' | 'decrease'
  }
  icon?: ReactNode
  description?: string
  className?: string
}

const StatCard = ({
  title,
  value,
  change,
  icon,
  description,
  className,
}: StatCardProps) => {
  return (
    <div className={cn('bg-white rounded-lg border border-gray-200 p-6', className)}>
      <div className="flex items-start justify-between mb-4">
        <div>
          <p className="text-sm font-medium text-gray-600 mb-1">{title}</p>
          <p className="text-3xl font-bold text-gray-900">{value}</p>
        </div>
        {icon && (
          <div className="p-3 bg-primary-50 rounded-lg text-primary-600">
            {icon}
          </div>
        )}
      </div>
      {change && (
        <div className="flex items-center gap-1 text-sm">
          {change.type === 'increase' ? (
            <TrendingUp className="w-4 h-4 text-green-600" />
          ) : (
            <TrendingDown className="w-4 h-4 text-red-600" />
          )}
          <span
            className={cn(
              'font-medium',
              change.type === 'increase' ? 'text-green-600' : 'text-red-600'
            )}
          >
            {Math.abs(change.value)}%
          </span>
          <span className="text-gray-500">vs last period</span>
        </div>
      )}
      {description && (
        <p className="text-sm text-gray-500 mt-2">{description}</p>
      )}
    </div>
  )
}

export default StatCard

