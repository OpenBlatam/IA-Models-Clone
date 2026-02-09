'use client';

import React from 'react';
import { clsx } from 'clsx';

interface HeatmapCell {
  date: string;
  value: number;
  label?: string;
}

interface HeatmapProps {
  data: HeatmapCell[];
  startDate?: Date;
  endDate?: Date;
  className?: string;
  colorScale?: (value: number) => string;
}

export const Heatmap: React.FC<HeatmapProps> = ({
  data,
  startDate,
  endDate,
  className,
  colorScale,
}) => {
  const defaultColorScale = (value: number) => {
    if (value === 0) return 'bg-gray-100 dark:bg-gray-800';
    if (value < 3) return 'bg-green-200 dark:bg-green-900';
    if (value < 6) return 'bg-green-400 dark:bg-green-700';
    if (value < 9) return 'bg-green-600 dark:bg-green-500';
    return 'bg-green-800 dark:bg-green-300';
  };

  const getColor = colorScale || defaultColorScale;
  const maxValue = Math.max(...data.map((d) => d.value), 0);

  return (
    <div className={clsx('w-full', className)}>
      <div className="grid grid-cols-7 gap-1">
        {data.map((cell, index) => (
          <div
            key={index}
            className={clsx(
              'aspect-square rounded text-xs flex items-center justify-center',
              getColor(cell.value),
              'hover:ring-2 hover:ring-primary-500 transition-all cursor-pointer'
            )}
            title={cell.label || `${cell.date}: ${cell.value}`}
          >
            {cell.value > 0 && (
              <span className="text-white dark:text-gray-900 font-medium">
                {cell.value}
              </span>
            )}
          </div>
        ))}
      </div>
      {maxValue > 0 && (
        <div className="flex items-center justify-end mt-4 space-x-2 text-xs text-gray-600 dark:text-gray-400">
          <span>Menos</span>
          <div className="flex space-x-1">
            {[0, 2, 4, 6, 8].map((val) => (
              <div
                key={val}
                className={clsx('w-3 h-3 rounded', getColor(val))}
              />
            ))}
          </div>
          <span>Más</span>
        </div>
      )}
    </div>
  );
};


