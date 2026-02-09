'use client';

import { useState, useEffect } from 'react';
import { Heart, CheckCircle, XCircle, AlertTriangle } from 'lucide-react';

interface HealthCheck {
  id: string;
  name: string;
  status: 'healthy' | 'warning' | 'error';
  message: string;
  lastCheck: Date;
}

export default function SystemHealth() {
  const [checks, setChecks] = useState<HealthCheck[]>([
    {
      id: '1',
      name: 'Conexión Robot',
      status: 'healthy',
      message: 'Conexión estable',
      lastCheck: new Date(),
    },
    {
      id: '2',
      name: 'API Backend',
      status: 'healthy',
      message: 'Respuesta normal',
      lastCheck: new Date(),
    },
    {
      id: '3',
      name: 'Base de Datos',
      status: 'warning',
      message: 'Latencia elevada',
      lastCheck: new Date(),
    },
    {
      id: '4',
      name: 'Memoria',
      status: 'healthy',
      message: 'Uso normal',
      lastCheck: new Date(),
    },
    {
      id: '5',
      name: 'CPU',
      status: 'healthy',
      message: 'Carga normal',
      lastCheck: new Date(),
    },
  ]);

  useEffect(() => {
    const interval = setInterval(() => {
      setChecks((prev) =>
        prev.map((check) => ({
          ...check,
          lastCheck: new Date(),
        }))
      );
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-yellow-400" />;
      case 'error':
        return <XCircle className="w-5 h-5 text-red-400" />;
      default:
        return <CheckCircle className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'bg-green-500/10 border-green-500/50';
      case 'warning':
        return 'bg-yellow-500/10 border-yellow-500/50';
      case 'error':
        return 'bg-red-500/10 border-red-500/50';
      default:
        return 'bg-gray-700/50 border-gray-600';
    }
  };

  const healthyCount = checks.filter((c) => c.status === 'healthy').length;
  const totalCount = checks.length;
  const healthPercentage = (healthyCount / totalCount) * 100;

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Heart className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Salud del Sistema</h3>
        </div>

        {/* Overall Health */}
        <div className="mb-6 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-300">Salud General</span>
            <span className="text-lg font-bold text-white">{healthPercentage.toFixed(0)}%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div
              className={`h-2 rounded-full transition-all ${
                healthPercentage >= 80
                  ? 'bg-green-500'
                  : healthPercentage >= 50
                  ? 'bg-yellow-500'
                  : 'bg-red-500'
              }`}
              style={{ width: `${healthPercentage}%` }}
            />
          </div>
          <p className="text-xs text-gray-400 mt-2">
            {healthyCount} de {totalCount} sistemas saludables
          </p>
        </div>

        {/* Health Checks */}
        <div className="space-y-3">
          {checks.map((check) => (
            <div
              key={check.id}
              className={`p-4 rounded-lg border ${getStatusColor(check.status)}`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-3 flex-1">
                  {getStatusIcon(check.status)}
                  <div className="flex-1">
                    <h4 className="font-semibold text-white mb-1">{check.name}</h4>
                    <p className="text-sm text-gray-300">{check.message}</p>
                    <p className="text-xs text-gray-400 mt-1">
                      Última verificación: {check.lastCheck.toLocaleTimeString('es-ES')}
                    </p>
                  </div>
                </div>
                <span className={`px-2 py-1 rounded text-xs capitalize ${
                  check.status === 'healthy'
                    ? 'bg-green-500/20 text-green-400'
                    : check.status === 'warning'
                    ? 'bg-yellow-500/20 text-yellow-400'
                    : 'bg-red-500/20 text-red-400'
                }`}>
                  {check.status}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}


