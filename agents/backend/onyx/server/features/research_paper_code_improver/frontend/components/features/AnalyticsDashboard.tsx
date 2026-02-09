'use client'

import React from 'react'
import { TrendingUp, TrendingDown, Activity, Zap } from 'lucide-react'
import { Card, StatsCard } from '../ui'
import StatisticsChart from './StatisticsChart'

interface AnalyticsDashboardProps {
  metrics?: {
    totalImprovements: number
    averageTime: number
    successRate: number
    papersProcessed: number
    trends?: {
      improvements: number[]
      time: number[]
    }
  }
}

const AnalyticsDashboard: React.FC<AnalyticsDashboardProps> = ({ metrics }) => {
  if (!metrics) {
    return (
      <Card>
        <div className="text-center py-12 text-gray-500">
          No analytics data available
        </div>
      </Card>
    )
  }

  const stats = [
    {
      title: 'Total Improvements',
      value: metrics.totalImprovements.toLocaleString(),
      icon: TrendingUp,
      color: 'text-blue-600',
      bgColor: 'bg-blue-100',
      trend: {
        value: 12,
        label: 'vs last month',
        isPositive: true,
      },
    },
    {
      title: 'Avg Processing Time',
      value: `${metrics.averageTime}ms`,
      icon: Zap,
      color: 'text-purple-600',
      bgColor: 'bg-purple-100',
      trend: {
        value: -5,
        label: 'vs last month',
        isPositive: true,
      },
    },
    {
      title: 'Success Rate',
      value: `${metrics.successRate}%`,
      icon: Activity,
      color: 'text-green-600',
      bgColor: 'bg-green-100',
      trend: {
        value: 3,
        label: 'vs last month',
        isPositive: true,
      },
    },
    {
      title: 'Papers Processed',
      value: metrics.papersProcessed.toLocaleString(),
      icon: TrendingUp,
      color: 'text-orange-600',
      bgColor: 'bg-orange-100',
    },
  ]

  const chartData = metrics.trends?.improvements.map((value, index) => ({
    name: `Day ${index + 1}`,
    value,
  })) || []

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => (
          <StatsCard
            key={stat.title}
            title={stat.title}
            value={stat.value}
            icon={stat.icon}
            color={stat.color}
            bgColor={stat.bgColor}
            trend={stat.trend}
          />
        ))}
      </div>

      {chartData.length > 0 && (
        <StatisticsChart
          data={chartData}
          type="line"
          title="Improvements Trend"
          color="#667eea"
        />
      )}
    </div>
  )
}

export default AnalyticsDashboard



