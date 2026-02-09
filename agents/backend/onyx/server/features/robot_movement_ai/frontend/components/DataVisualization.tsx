'use client';

import { useState } from 'react';
import { BarChart3, LineChart, PieChart, TrendingUp } from 'lucide-react';

export default function DataVisualization() {
  const [chartType, setChartType] = useState<'line' | 'bar' | 'pie' | 'area'>('line');
  const [timeRange, setTimeRange] = useState<'1h' | '24h' | '7d' | '30d'>('24h');

  const data = {
    labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
    values: [45, 52, 48, 61, 55, 58],
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Visualización de Datos</h3>
          </div>
          <div className="flex gap-2">
            {(['1h', '24h', '7d', '30d'] as const).map((range) => (
              <button
                key={range}
                onClick={() => setTimeRange(range)}
                className={`px-3 py-1 rounded-lg text-sm transition-colors ${
                  timeRange === range
                    ? 'bg-primary-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                {range}
              </button>
            ))}
          </div>
        </div>

        {/* Chart Type Selector */}
        <div className="mb-6 flex gap-2">
          {(['line', 'bar', 'pie', 'area'] as const).map((type) => (
            <button
              key={type}
              onClick={() => setChartType(type)}
              className={`px-3 py-1 rounded-lg text-sm transition-colors flex items-center gap-2 ${
                chartType === type
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {type === 'line' && <LineChart className="w-4 h-4" />}
              {type === 'bar' && <BarChart3 className="w-4 h-4" />}
              {type === 'pie' && <PieChart className="w-4 h-4" />}
              {type === 'area' && <TrendingUp className="w-4 h-4" />}
              {type === 'line' ? 'Línea' : type === 'bar' ? 'Barras' : type === 'pie' ? 'Circular' : 'Área'}
            </button>
          ))}
        </div>

        {/* Chart */}
        <div className="p-4 bg-gray-700/50 rounded-lg border border-gray-600">
          <h4 className="text-sm font-medium text-white mb-3">
            Gráfico {chartType} - Rango: {timeRange}
          </h4>
          <div className="h-64 bg-gray-800 rounded-lg flex items-center justify-center">
            <div className="text-center">
              <BarChart3 className="w-12 h-12 mx-auto mb-4 text-gray-400 opacity-50" />
              <p className="text-gray-400">Visualización de datos</p>
              <p className="text-xs text-gray-500 mt-2">
                Datos: {data.values.join(', ')}
              </p>
            </div>
          </div>
        </div>

        {/* Statistics */}
        <div className="grid grid-cols-4 gap-4 mt-6">
          <div className="p-3 bg-gray-700/50 rounded-lg border border-gray-600">
            <p className="text-xs text-gray-400 mb-1">Promedio</p>
            <p className="text-lg font-bold text-white">
              {(data.values.reduce((a, b) => a + b, 0) / data.values.length).toFixed(1)}
            </p>
          </div>
          <div className="p-3 bg-gray-700/50 rounded-lg border border-gray-600">
            <p className="text-xs text-gray-400 mb-1">Máximo</p>
            <p className="text-lg font-bold text-white">{Math.max(...data.values)}</p>
          </div>
          <div className="p-3 bg-gray-700/50 rounded-lg border border-gray-600">
            <p className="text-xs text-gray-400 mb-1">Mínimo</p>
            <p className="text-lg font-bold text-white">{Math.min(...data.values)}</p>
          </div>
          <div className="p-3 bg-gray-700/50 rounded-lg border border-gray-600">
            <p className="text-xs text-gray-400 mb-1">Total</p>
            <p className="text-lg font-bold text-white">{data.values.reduce((a, b) => a + b, 0)}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
