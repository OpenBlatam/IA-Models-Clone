'use client'

import React from 'react'
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import { Card } from '../ui'

interface StatisticsChartProps {
  data: Array<{
    name: string
    value: number
    [key: string]: string | number
  }>
  type?: 'line' | 'bar'
  title?: string
  dataKey?: string
  color?: string
}

const StatisticsChart: React.FC<StatisticsChartProps> = ({
  data,
  type = 'line',
  title,
  dataKey = 'value',
  color = '#667eea',
}) => {
  if (!data || data.length === 0) {
    return (
      <Card>
        <div className="text-center py-8 text-gray-500">
          No data available
        </div>
      </Card>
    )
  }

  return (
    <Card>
      {title && (
        <h3 className="text-lg font-semibold text-gray-900 mb-4">{title}</h3>
      )}
      <ResponsiveContainer width="100%" height={300}>
        {type === 'line' ? (
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey={dataKey}
              stroke={color}
              strokeWidth={2}
            />
          </LineChart>
        ) : (
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey={dataKey} fill={color} />
          </BarChart>
        )}
      </ResponsiveContainer>
    </Card>
  )
}

export default StatisticsChart




