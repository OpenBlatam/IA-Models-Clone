'use client';

import { useState, useEffect } from 'react';
import { useRobotStore } from '@/lib/store/robotStore';
import { TrendingUp, BarChart3, PieChart, LineChart } from 'lucide-react';
import {
  LineChart as RechartsLineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart as RechartsPieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

export default function DataAnalytics() {
  const { status } = useRobotStore();
  const [timeRange, setTimeRange] = useState<'hour' | 'day' | 'week' | 'month'>('day');
  const [data, setData] = useState<any[]>([]);

  useEffect(() => {
    // Generate mock analytics data
    const generateData = () => {
      const points = timeRange === 'hour' ? 60 : timeRange === 'day' ? 24 : timeRange === 'week' ? 7 : 30;
      return Array.from({ length: points }, (_, i) => ({
        time: i,
        movements: Math.floor(Math.random() * 100),
        errors: Math.floor(Math.random() * 10),
        efficiency: 85 + Math.random() * 15,
      }));
    };
    setData(generateData());
  }, [timeRange]);

  const efficiencyData = [
    { name: 'Alta', value: 75, color: '#10B981' },
    { name: 'Media', value: 20, color: '#F59E0B' },
    { name: 'Baja', value: 5, color: '#EF4444' },
  ];

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Análisis de Datos</h3>
          </div>
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value as any)}
            className="px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
          >
            <option value="hour">Última hora</option>
            <option value="day">Último día</option>
            <option value="week">Última semana</option>
            <option value="month">Último mes</option>
          </select>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="p-4 bg-gray-700/50 rounded-lg border border-gray-600">
            <p className="text-sm text-gray-400 mb-1">Movimientos Totales</p>
            <p className="text-2xl font-bold text-white">
              {data.reduce((sum, d) => sum + d.movements, 0)}
            </p>
          </div>
          <div className="p-4 bg-gray-700/50 rounded-lg border border-gray-600">
            <p className="text-sm text-gray-400 mb-1">Errores</p>
            <p className="text-2xl font-bold text-red-400">
              {data.reduce((sum, d) => sum + d.errors, 0)}
            </p>
          </div>
          <div className="p-4 bg-gray-700/50 rounded-lg border border-gray-600">
            <p className="text-sm text-gray-400 mb-1">Eficiencia Promedio</p>
            <p className="text-2xl font-bold text-green-400">
              {Math.round(data.reduce((sum, d) => sum + d.efficiency, 0) / data.length)}%
            </p>
          </div>
        </div>

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Movements Chart */}
          <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
            <h4 className="text-sm font-medium text-gray-300 mb-4 flex items-center gap-2">
              <BarChart3 className="w-4 h-4" />
              Movimientos
            </h4>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="time" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                />
                <Bar dataKey="movements" fill="#0EA5E9" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Efficiency Chart */}
          <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
            <h4 className="text-sm font-medium text-gray-300 mb-4 flex items-center gap-2">
              <LineChart className="w-4 h-4" />
              Eficiencia
            </h4>
            <ResponsiveContainer width="100%" height={250}>
              <AreaChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="time" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                />
                <Area
                  type="monotone"
                  dataKey="efficiency"
                  stroke="#10B981"
                  fill="#10B981"
                  fillOpacity={0.3}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Errors Chart */}
          <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
            <h4 className="text-sm font-medium text-gray-300 mb-4 flex items-center gap-2">
              <LineChart className="w-4 h-4" />
              Errores
            </h4>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="time" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                />
                <Line
                  type="monotone"
                  dataKey="errors"
                  stroke="#EF4444"
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Efficiency Distribution */}
          <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
            <h4 className="text-sm font-medium text-gray-300 mb-4 flex items-center gap-2">
              <PieChart className="w-4 h-4" />
              Distribución de Eficiencia
            </h4>
            <ResponsiveContainer width="100%" height={250}>
              <RechartsPieChart>
                <Pie
                  data={efficiencyData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {efficiencyData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                />
              </RechartsPieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}


