'use client';

import { useState, useEffect } from 'react';
import { Gauge, Activity, Zap, HardDrive } from 'lucide-react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

export default function PerformanceMonitor() {
  const [metrics, setMetrics] = useState({
    fps: 60,
    memory: 0,
    cpu: 0,
    network: 0,
  });
  const [history, setHistory] = useState<Array<{ time: string; fps: number; memory: number; cpu: number }>>([]);

  useEffect(() => {
    const interval = setInterval(() => {
      // Simulate performance metrics
      const newMetrics = {
        fps: 50 + Math.random() * 10,
        memory: performance.memory ? (performance.memory.usedJSHeapSize / 1048576).toFixed(2) : 0,
        cpu: 20 + Math.random() * 30,
        network: Math.random() * 100,
      };

      setMetrics(newMetrics);
      setHistory((prev) => {
        const newHistory = [
          ...prev,
          {
            time: new Date().toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
            fps: newMetrics.fps,
            memory: parseFloat(newMetrics.memory as string),
            cpu: newMetrics.cpu,
          },
        ];
        return newHistory.slice(-20); // Keep last 20 entries
      });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const chartData = history.map((h) => ({
    time: h.time.split(' ')[0],
    fps: h.fps,
    memory: h.memory,
    cpu: h.cpu,
  }));

  return (
    <div className="space-y-6">
      {/* Real-time Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
          <div className="flex items-center justify-between mb-2">
            <Gauge className="w-5 h-5 text-blue-400" />
            <span className="text-xs text-gray-400">FPS</span>
          </div>
          <p className="text-3xl font-bold text-white">{metrics.fps.toFixed(0)}</p>
          <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-400 transition-all"
              style={{ width: `${(metrics.fps / 60) * 100}%` }}
            />
          </div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
          <div className="flex items-center justify-between mb-2">
            <HardDrive className="w-5 h-5 text-green-400" />
            <span className="text-xs text-gray-400">Memoria</span>
          </div>
          <p className="text-3xl font-bold text-white">{metrics.memory} MB</p>
          <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-green-400 transition-all"
              style={{ width: `${Math.min((parseFloat(metrics.memory as string) / 100) * 100, 100)}%` }}
            />
          </div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
          <div className="flex items-center justify-between mb-2">
            <Activity className="w-5 h-5 text-yellow-400" />
            <span className="text-xs text-gray-400">CPU</span>
          </div>
          <p className="text-3xl font-bold text-white">{metrics.cpu.toFixed(0)}%</p>
          <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-yellow-400 transition-all"
              style={{ width: `${metrics.cpu}%` }}
            />
          </div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
          <div className="flex items-center justify-between mb-2">
            <Zap className="w-5 h-5 text-purple-400" />
            <span className="text-xs text-gray-400">Red</span>
          </div>
          <p className="text-3xl font-bold text-white">{metrics.network.toFixed(0)}%</p>
          <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-purple-400 transition-all"
              style={{ width: `${metrics.network}%` }}
            />
          </div>
        </div>
      </div>

      {/* Performance Chart */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4">Rendimiento en Tiempo Real</h3>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="time" stroke="#9CA3AF" />
            <YAxis stroke="#9CA3AF" />
            <Tooltip
              contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
            />
            <Area
              type="monotone"
              dataKey="fps"
              stroke="#0EA5E9"
              fill="#0EA5E9"
              fillOpacity={0.3}
              name="FPS"
            />
            <Area
              type="monotone"
              dataKey="cpu"
              stroke="#F59E0B"
              fill="#F59E0B"
              fillOpacity={0.3}
              name="CPU %"
            />
            <Area
              type="monotone"
              dataKey="memory"
              stroke="#10B981"
              fill="#10B981"
              fillOpacity={0.3}
              name="Memoria MB"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}


