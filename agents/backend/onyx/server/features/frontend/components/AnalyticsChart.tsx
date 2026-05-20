'use client';

import { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

interface AnalyticsChartProps {
  data: { label: string; value: number; color?: string }[];
  title: string;
  type?: 'bar' | 'line' | 'pie';
}

export default function AnalyticsChart({ data, title, type = 'bar' }: AnalyticsChartProps) {
  const maxValue = Math.max(...data.map((d) => d.value), 1);

  if (type === 'bar') {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">{title}</h3>
        <div className="space-y-3">
          {data.map((item, index) => (
            <motion.div
              key={item.label}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm text-gray-700 dark:text-gray-300">{item.label}</span>
                <span className="text-sm font-medium text-gray-900 dark:text-white">
                  {item.value}
                </span>
              </div>
              <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${(item.value / maxValue) * 100}%` }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  className={`h-full ${item.color || 'bg-primary-500'} rounded-full`}
                />
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">{title}</h3>
      <div className="text-center py-8 text-gray-500 dark:text-gray-400">
        Gráfico {type} (próximamente)
      </div>
    </div>
  );
}


