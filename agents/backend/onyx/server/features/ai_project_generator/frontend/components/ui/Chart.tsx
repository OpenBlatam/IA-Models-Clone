'use client'

import { ReactNode } from 'react'
import { cn } from '@/lib/utils'

interface ChartDataPoint {
  label: string
  value: number
  color?: string
}

interface ChartProps {
  data: ChartDataPoint[]
  type?: 'bar' | 'line' | 'pie' | 'area'
  className?: string
  showLabels?: boolean
  showValues?: boolean
  height?: number
}

const Chart = ({
  data,
  type = 'bar',
  className,
  showLabels = true,
  showValues = false,
  height = 300,
}: ChartProps) => {
  const maxValue = Math.max(...data.map((d) => d.value))
  const totalValue = data.reduce((sum, d) => sum + d.value, 0)

  if (type === 'pie') {
    let currentAngle = 0
    return (
      <div className={cn('relative', className)} style={{ width: height, height }}>
        <svg viewBox="0 0 200 200" className="transform -rotate-90">
          {data.map((point, index) => {
            const percentage = (point.value / totalValue) * 100
            const angle = (percentage / 100) * 360
            const largeArcFlag = angle > 180 ? 1 : 0
            const x1 = 100 + 100 * Math.cos((currentAngle * Math.PI) / 180)
            const y1 = 100 + 100 * Math.sin((currentAngle * Math.PI) / 180)
            const x2 = 100 + 100 * Math.cos(((currentAngle + angle) * Math.PI) / 180)
            const y2 = 100 + 100 * Math.sin(((currentAngle + angle) * Math.PI) / 180)

            const pathData = [
              `M 100 100`,
              `L ${x1} ${y1}`,
              `A 100 100 0 ${largeArcFlag} 1 ${x2} ${y2}`,
              'Z',
            ].join(' ')

            const color = point.color || `hsl(${(index * 360) / data.length}, 70%, 50%)`
            currentAngle += angle

            return (
              <path
                key={index}
                d={pathData}
                fill={color}
                stroke="white"
                strokeWidth="2"
              />
            )
          })}
        </svg>
        {showLabels && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <div className="text-2xl font-bold">{totalValue}</div>
              <div className="text-sm text-gray-500">Total</div>
            </div>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className={cn('space-y-2', className)}>
      {data.map((point, index) => {
        const percentage = (point.value / maxValue) * 100
        const color = point.color || `hsl(${(index * 240) / data.length}, 70%, 50%)`

        if (type === 'bar') {
          return (
            <div key={index} className="space-y-1">
              {showLabels && (
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-700">{point.label}</span>
                  {showValues && <span className="font-medium">{point.value}</span>}
                </div>
              )}
              <div className="w-full bg-gray-200 rounded-full h-6 overflow-hidden">
                <div
                  className="h-full transition-all duration-500 rounded-full"
                  style={{ width: `${percentage}%`, backgroundColor: color }}
                />
              </div>
            </div>
          )
        }

        if (type === 'line' || type === 'area') {
          return (
            <div key={index} className="flex items-end gap-1" style={{ height: `${height}px` }}>
              <div
                className={cn(
                  'transition-all duration-500 rounded-t',
                  type === 'area' ? 'w-full' : 'w-1'
                )}
                style={{
                  height: `${percentage}%`,
                  backgroundColor: color,
                }}
                title={`${point.label}: ${point.value}`}
              />
            </div>
          )
        }

        return null
      })}
    </div>
  )
}

export default Chart

