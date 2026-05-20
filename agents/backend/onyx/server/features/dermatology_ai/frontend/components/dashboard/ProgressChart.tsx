'use client';

import React, { memo } from 'react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface ProgressChartProps {
  data: Array<{
    date: string;
    value: number;
    label?: string;
  }>;
  type?: 'line' | 'area';
  color?: string;
  showGrid?: boolean;
  height?: number;
}

export const ProgressChart: React.FC<ProgressChartProps> = memo(({
  data,
  type = 'line',
  color = '#0ea5e9',
  showGrid = true,
  height = 300,
}) => {
  const ChartComponent = type === 'area' ? AreaChart : LineChart;
  const DataComponent = type === 'area' ? Area : Line;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ChartComponent data={data}>
        {showGrid && <CartesianGrid strokeDasharray="3 3" className="opacity-30" />}
        <XAxis
          dataKey="date"
          tick={{ fill: 'currentColor' }}
          className="text-gray-600 dark:text-gray-400"
        />
        <YAxis
          tick={{ fill: 'currentColor' }}
          className="text-gray-600 dark:text-gray-400"
        />
        <Tooltip
          contentStyle={{
            backgroundColor: 'rgba(255, 255, 255, 0.95)',
            border: '1px solid #e5e7eb',
            borderRadius: '8px',
          }}
          labelStyle={{ color: '#374151' }}
        />
        <Legend />
        <DataComponent
          type="monotone"
          dataKey="value"
          stroke={color}
          fill={type === 'area' ? color : undefined}
          fillOpacity={type === 'area' ? 0.3 : undefined}
          strokeWidth={2}
          name="Score"
        />
      </ChartComponent>
    </ResponsiveContainer>
  );
});

ProgressChart.displayName = 'ProgressChart';

