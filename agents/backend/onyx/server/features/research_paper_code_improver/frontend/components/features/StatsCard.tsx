'use client'

import React from 'react'
import { LucideIcon, TrendingUp, TrendingDown } from 'lucide-react'
import { Card } from '../ui'
import { clsx } from 'clsx'

interface StatsCardProps {
  title: string
  value: string | number
  icon: LucideIcon
  trend?: {
    value: number
    label: string
    isPositive?: boolean
  }
  color?: string
  bgColor?: string
  className?: string
}

const StatsCard: React.FC<StatsCardProps> = ({
  title,
  value,
  icon: Icon,
  trend,
  color = 'text-primary-600',
  bgColor = 'bg-primary-100',
  className,
}) => {
  return (
    <Card className={clsx('hover:shadow-md transition-shadow', className)}>
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-600 mb-1">{title}</p>
          <p className="text-3xl font-bold text-gray-900">{value}</p>
          {trend && (
            <div
              className={clsx(
                'flex items-center gap-1 mt-2 text-sm',
                trend.isPositive ? 'text-green-600' : 'text-red-600'
              )}
            >
              {trend.isPositive ? (
                <TrendingUp className="w-4 h-4" />
              ) : (
                <TrendingDown className="w-4 h-4" />
              )}
              <span>{trend.value}%</span>
              <span className="text-gray-500">{trend.label}</span>
            </div>
          )}
        </div>
        <div className={clsx('p-3 rounded-lg', bgColor, color)}>
          <Icon className="w-6 h-6" />
        </div>
      </div>
    </Card>
  )
}

export default StatsCard




