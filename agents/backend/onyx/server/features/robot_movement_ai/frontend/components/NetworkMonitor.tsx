'use client';

import { useState, useEffect } from 'react';
import { Network, Activity, Wifi, WifiOff } from 'lucide-react';

interface NetworkStat {
  name: string;
  value: number;
  unit: string;
  trend: 'up' | 'down' | 'stable';
}

export default function NetworkMonitor() {
  const [stats, setStats] = useState<NetworkStat[]>([
    { name: 'Latencia', value: 45, unit: 'ms', trend: 'stable' },
    { name: 'Ancho de Banda', value: 85, unit: 'Mbps', trend: 'up' },
    { name: 'Paquetes Enviados', value: 1250, unit: 'pkts', trend: 'up' },
    { name: 'Paquetes Recibidos', value: 1180, unit: 'pkts', trend: 'up' },
  ]);
  const [isConnected, setIsConnected] = useState(true);

  useEffect(() => {
    const interval = setInterval(() => {
      setStats((prev) =>
        prev.map((stat) => ({
          ...stat,
          value: stat.value + (Math.random() - 0.5) * 10,
          trend: Math.random() > 0.5 ? 'up' : 'down',
        }))
      );
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case 'up':
        return 'text-green-400';
      case 'down':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Network className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Monitor de Red</h3>
          </div>
          <div className={`flex items-center gap-2 px-3 py-1 rounded-lg ${
            isConnected
              ? 'bg-green-500/20 text-green-400'
              : 'bg-red-500/20 text-red-400'
          }`}>
            {isConnected ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
            <span className="text-sm font-medium">
              {isConnected ? 'Conectado' : 'Desconectado'}
            </span>
          </div>
        </div>

        {/* Network Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          {stats.map((stat, index) => (
            <div
              key={index}
              className="p-4 bg-gray-700/50 rounded-lg border border-gray-600"
            >
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm text-gray-400">{stat.name}</p>
                <span className={`text-xs ${getTrendColor(stat.trend)}`}>
                  {stat.trend === 'up' ? '↑' : stat.trend === 'down' ? '↓' : '→'}
                </span>
              </div>
              <div className="flex items-baseline gap-2">
                <p className="text-2xl font-bold text-white">
                  {stat.value.toFixed(stat.unit === 'ms' ? 0 : 0)}
                </p>
                <span className="text-sm text-gray-400">{stat.unit}</span>
              </div>
            </div>
          ))}
        </div>

        {/* Network Activity */}
        <div className="p-4 bg-gray-700/50 rounded-lg border border-gray-600">
          <h4 className="text-sm font-medium text-white mb-3 flex items-center gap-2">
            <Activity className="w-4 h-4" />
            Actividad de Red
          </h4>
          <div className="h-32 bg-gray-800 rounded-lg flex items-center justify-center">
            <p className="text-gray-400">Gráfico de actividad de red en tiempo real</p>
          </div>
        </div>
      </div>
    </div>
  );
}


