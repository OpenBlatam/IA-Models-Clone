'use client';

import { useEffect, useRef, useState } from 'react';
import { useRobotStore } from '@/lib/store/robotStore';
import { Radio, TrendingUp, Activity } from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

export default function RealTimeVisualization() {
  const { status } = useRobotStore();
  const [data, setData] = useState<Array<{ time: string; x: number; y: number; z: number; velocity: number }>>([]);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    intervalRef.current = setInterval(() => {
      if (status?.robot_status.position) {
        const newData = {
          time: new Date().toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
          x: status.robot_status.position.x,
          y: status.robot_status.position.y,
          z: status.robot_status.position.z,
          velocity: status.robot_status.velocity
            ? Math.sqrt(
                status.robot_status.velocity[0] ** 2 +
                status.robot_status.velocity[1] ** 2 +
                status.robot_status.velocity[2] ** 2
              )
            : 0,
        };
        setData((prev) => {
          const updated = [...prev, newData];
          return updated.slice(-50); // Keep last 50 points
        });
      }
    }, 100); // Update every 100ms for real-time feel

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [status]);

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-4">
          <Radio className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Visualización en Tiempo Real</h3>
        </div>

        {/* Position Chart */}
        <div className="mb-6">
          <h4 className="text-sm font-medium text-gray-300 mb-3">Posición (X, Y, Z)</h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip
                contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
              />
              <Line
                type="monotone"
                dataKey="x"
                stroke="#0EA5E9"
                strokeWidth={2}
                dot={false}
                name="X"
              />
              <Line
                type="monotone"
                dataKey="y"
                stroke="#10B981"
                strokeWidth={2}
                dot={false}
                name="Y"
              />
              <Line
                type="monotone"
                dataKey="z"
                stroke="#F59E0B"
                strokeWidth={2}
                dot={false}
                name="Z"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Velocity Chart */}
        <div>
          <h4 className="text-sm font-medium text-gray-300 mb-3 flex items-center gap-2">
            <TrendingUp className="w-4 h-4" />
            Velocidad
          </h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip
                contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
              />
              <ReferenceLine y={1.0} stroke="#EF4444" strokeDasharray="5 5" label="Max" />
              <Line
                type="monotone"
                dataKey="velocity"
                stroke="#EF4444"
                strokeWidth={2}
                dot={false}
                name="Velocidad (m/s)"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Current Values */}
        {status?.robot_status.position && (
          <div className="mt-6 grid grid-cols-4 gap-4">
            <div className="p-3 bg-gray-700/50 rounded">
              <p className="text-xs text-gray-400 mb-1">X</p>
              <p className="text-lg font-bold text-white">
                {status.robot_status.position.x.toFixed(3)}m
              </p>
            </div>
            <div className="p-3 bg-gray-700/50 rounded">
              <p className="text-xs text-gray-400 mb-1">Y</p>
              <p className="text-lg font-bold text-white">
                {status.robot_status.position.y.toFixed(3)}m
              </p>
            </div>
            <div className="p-3 bg-gray-700/50 rounded">
              <p className="text-xs text-gray-400 mb-1">Z</p>
              <p className="text-lg font-bold text-white">
                {status.robot_status.position.z.toFixed(3)}m
              </p>
            </div>
            <div className="p-3 bg-gray-700/50 rounded">
              <p className="text-xs text-gray-400 mb-1 flex items-center gap-1">
                <Activity className="w-3 h-3" />
                Velocidad
              </p>
              <p className="text-lg font-bold text-white">
                {data[data.length - 1]?.velocity.toFixed(3) || '0.000'}m/s
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}


