'use client';

import { Bot, Activity, Battery, Thermometer, AlertCircle } from 'lucide-react';
import { type RobotStatus as RobotStatusType } from '@/lib/api/robot-api';

interface RobotStatusProps {
  status?: RobotStatusType;
  metrics?: any;
}

export function RobotStatus({ status, metrics }: RobotStatusProps) {
  const isConnected = status?.connected || false;

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
        <Bot className="w-6 h-6" />
        Estado del Robot
      </h2>

      <div className="space-y-4">
        {/* Connection Status */}
        <div className="flex items-center justify-between">
          <span className="text-gray-300">Estado</span>
          <div className="flex items-center gap-2">
            <div
              className={`w-3 h-3 rounded-full ${
                isConnected ? 'bg-green-400' : 'bg-red-400'
              }`}
            />
            <span className="text-white font-medium">
              {isConnected ? 'Conectado' : 'Desconectado'}
            </span>
          </div>
        </div>

        {/* Position */}
        {status?.position && (
          <div className="bg-white/5 rounded-lg p-4 space-y-2">
            <p className="text-sm text-gray-400 mb-2">Posición Actual</p>
            <div className="grid grid-cols-3 gap-2 text-sm">
              <div>
                <span className="text-gray-400">X:</span>
                <span className="text-white ml-2 font-mono">
                  {status.position.x.toFixed(2)}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Y:</span>
                <span className="text-white ml-2 font-mono">
                  {status.position.y.toFixed(2)}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Z:</span>
                <span className="text-white ml-2 font-mono">
                  {status.position.z.toFixed(2)}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Battery */}
        {status?.battery !== undefined && (
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Battery className="w-5 h-5 text-gray-300" />
              <span className="text-gray-300">Batería</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-24 bg-gray-700 rounded-full h-2">
                <div
                  className={`h-2 rounded-full ${
                    status.battery > 50
                      ? 'bg-green-400'
                      : status.battery > 20
                      ? 'bg-yellow-400'
                      : 'bg-red-400'
                  }`}
                  style={{ width: `${status.battery}%` }}
                />
              </div>
              <span className="text-white font-medium w-12 text-right">
                {status.battery}%
              </span>
            </div>
          </div>
        )}

        {/* Temperature */}
        {status?.temperature !== undefined && (
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Thermometer className="w-5 h-5 text-gray-300" />
              <span className="text-gray-300">Temperatura</span>
            </div>
            <span className="text-white font-medium">
              {status.temperature}°C
            </span>
          </div>
        )}

        {/* Status Message */}
        {status?.status && (
          <div className="bg-white/5 rounded-lg p-3">
            <p className="text-sm text-gray-400 mb-1">Estado</p>
            <p className="text-white">{status.status}</p>
          </div>
        )}

        {/* Errors */}
        {status?.errors && status.errors.length > 0 && (
          <div className="bg-red-500/20 border border-red-500/50 rounded-lg p-3">
            <div className="flex items-center gap-2 mb-2">
              <AlertCircle className="w-5 h-5 text-red-400" />
              <p className="text-sm font-semibold text-red-300">Errores</p>
            </div>
            <ul className="space-y-1">
              {status.errors.map((error, idx) => (
                <li key={idx} className="text-sm text-red-200">
                  • {error}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Metrics */}
        {metrics && (
          <div className="mt-4 pt-4 border-t border-white/20">
            <p className="text-sm text-gray-400 mb-2">Métricas</p>
            <div className="space-y-2 text-sm">
              {Object.entries(metrics).map(([key, value]) => (
                <div key={key} className="flex justify-between">
                  <span className="text-gray-300 capitalize">
                    {key.replace(/_/g, ' ')}
                  </span>
                  <span className="text-white font-mono">{String(value)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

