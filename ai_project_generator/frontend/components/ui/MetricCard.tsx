'use client'

import { ReactNode } from 'react'
import { cn } from '@/lib/utils'
import { Progress } from '@/components/ui'

interface MetricCardProps {
  label: string
  value: number
  max?: number
  unit?: string
  icon?: ReactNode
  color?: 'primary' | 'success' | 'warning' | 'danger'
  showProgress?: boolean
  className?: string
}

const MetricCard = ({
  label,
  value,
  max,
  unit,
  icon,
  color = 'primary',
  showProgress = false,
  className,
}: MetricCardProps) => {
  const colorClasses = {
    primary: 'text-primary-600 bg-primary-50',
    success: 'text-green-600 bg-green-50',
    warning: 'text-yellow-600 bg-yellow-50',
    danger: 'text-red-600 bg-red-50',
  }

  const percentage = max ? (value / max) * 100 : 0

  return (
    <div className={cn('bg-white rounded-lg border border-gray-200 p-4', className)}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-600">{label}</span>
        {icon && (
          <div className={cn('p-2 rounded', colorClasses[color])}>
            {icon}
          </div>
        )}
      </div>
      <div className="mb-2">
        <span className="text-2xl font-bold text-gray-900">{value}</span>
        {unit && <span className="text-sm text-gray-500 ml-1">{unit}</span>}
        {max && (
          <span className="text-sm text-gray-500 ml-2">/ {max}</span>
        )}
      </div>
      {showProgress && max && (
        <Progress
          value={percentage}
          color={color}
          size="sm"
          showLabel={false}
        />
      )}
    </div>
  )
}

export default MetricCard

