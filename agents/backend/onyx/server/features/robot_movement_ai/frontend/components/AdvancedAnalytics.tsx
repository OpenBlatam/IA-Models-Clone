'use client';

import { useState } from 'react';
import { TrendingUp, BarChart3, PieChart, LineChart } from 'lucide-react';

export default function AdvancedAnalytics() {
  const [timeRange, setTimeRange] = useState<'1h' | '24h' | '7d' | '30d'>('24h');
  const [chartType, setChartType] = useState<'line' | 'bar' | 'pie'>('line');

  const analytics = {
    totalMovements: 1247,
    averageSpeed: 0.65,
    energyConsumed: 85.3,
    efficiency: 92.5,
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Análisis Avanzado</h3>
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

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="p-4 bg-gray-700/50 rounded-lg border border-gray-600">
            <p className="text-sm text-gray-400 mb-1">Movimientos Totales</p>
            <p className="text-2xl font-bold text-white">{analytics.totalMovements}</p>
          </div>
          <div className="p-4 bg-gray-700/50 rounded-lg border border-gray-600">
            <p className="text-sm text-gray-400 mb-1">Velocidad Promedio</p>
            <p className="text-2xl font-bold text-white">{analytics.averageSpeed} m/s</p>
          </div>
          <div className="p-4 bg-gray-700/50 rounded-lg border border-gray-600">
            <p className="text-sm text-gray-400 mb-1">Energía Consumida</p>
            <p className="text-2xl font-bold text-white">{analytics.energyConsumed} kWh</p>
          </div>
          <div className="p-4 bg-gray-700/50 rounded-lg border border-gray-600">
            <p className="text-sm text-gray-400 mb-1">Eficiencia</p>
            <p className="text-2xl font-bold text-white">{analytics.efficiency}%</p>
          </div>
        </div>

        {/* Chart Type Selector */}
        <div className="mb-4 flex gap-2">
          {(['line', 'bar', 'pie'] as const).map((type) => (
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
              {type === 'line' ? 'Línea' : type === 'bar' ? 'Barras' : 'Circular'}
            </button>
          ))}
        </div>

        {/* Chart Placeholder */}
        <div className="p-4 bg-gray-700/50 rounded-lg border border-gray-600">
          <h4 className="text-sm font-medium text-white mb-3">Gráfico de Análisis</h4>
          <div className="h-64 bg-gray-800 rounded-lg flex items-center justify-center">
            <p className="text-gray-400">Gráfico {chartType} - Rango: {timeRange}</p>
          </div>
        </div>
      </div>
    </div>
  );
}


