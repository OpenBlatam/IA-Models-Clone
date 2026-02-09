'use client';

import { useEffect } from 'react';
import { useRobotStore } from '@/lib/store/robotStore';
import { apiClient } from '@/lib/api/client';
import { useAsync } from '@/lib/hooks/useAsync';
import { RefreshCw, CheckCircle, XCircle, AlertCircle } from 'lucide-react';

export default function StatusPanel() {
  const { status, fetchStatus, isLoading } = useRobotStore();
  const { execute: loadHealth, data: health, loading: healthLoading } = useAsync(
    () => apiClient.getHealth()
  );
  const { execute: loadStatistics, data: statistics, loading: statsLoading } = useAsync(
    () => apiClient.getStatistics()
  );

  useEffect(() => {
    fetchStatus();
    loadHealth();
    loadStatistics();
  }, [fetchStatus, loadHealth, loadStatistics]);

  const handleRefresh = () => {
    fetchStatus();
    loadHealth();
    loadStatistics();
  };

  return (
    <div className="space-y-tesla-lg">
      {/* Header */}
      <div className="bg-white rounded-lg p-tesla-lg border border-gray-200 shadow-sm">
        <div className="flex items-center justify-between mb-tesla-lg">
          <h2 className="text-2xl font-semibold text-tesla-black">Estado del Sistema</h2>
          <button
            onClick={handleRefresh}
            disabled={isLoading}
            className="flex items-center gap-tesla-sm px-tesla-md py-tesla-sm bg-white border-2 border-gray-300 hover:border-gray-400 text-tesla-black rounded-md transition-all disabled:opacity-50 font-medium"
          >
            <RefreshCw
              className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`}
            />
            Actualizar
          </button>
        </div>

        {/* Robot Status */}
        {status && (
          <div className="space-y-tesla-lg">
            <div>
              <h3 className="text-lg font-medium text-tesla-black mb-tesla-md">
                Estado del Robot
              </h3>
              <div className="grid grid-cols-2 gap-tesla-md">
                <div className="p-tesla-lg bg-gray-50 rounded-md border border-gray-200">
                  <p className="text-tesla-gray-dark text-sm mb-tesla-sm font-medium">Conexión</p>
                  <div className="flex items-center gap-tesla-sm">
                    {status.robot_status.connected ? (
                      <CheckCircle className="w-5 h-5 text-green-600" />
                    ) : (
                      <XCircle className="w-5 h-5 text-red-600" />
                    )}
                    <span className="text-tesla-black font-semibold">
                      {status.robot_status.connected ? 'Conectado' : 'Desconectado'}
                    </span>
                  </div>
                </div>
                <div className="p-tesla-lg bg-gray-50 rounded-md border border-gray-200">
                  <p className="text-tesla-gray-dark text-sm mb-tesla-sm font-medium">Movimiento</p>
                  <div className="flex items-center gap-tesla-sm">
                    {status.robot_status.moving ? (
                      <AlertCircle className="w-5 h-5 text-yellow-600" />
                    ) : (
                      <CheckCircle className="w-5 h-5 text-green-600" />
                    )}
                    <span className="text-tesla-black font-semibold">
                      {status.robot_status.moving ? 'En Movimiento' : 'Detenido'}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Position */}
            {status.robot_status.position && (
              <div>
                <h3 className="text-lg font-medium text-tesla-black mb-tesla-md">Posición</h3>
                <div className="p-tesla-lg bg-gray-50 rounded-md border border-gray-200">
                  <div className="grid grid-cols-3 gap-tesla-md text-sm">
                    <div>
                      <p className="text-tesla-gray-dark mb-tesla-xs font-medium">X</p>
                      <p className="text-tesla-black font-semibold text-base">
                        {status.robot_status.position.x.toFixed(3)} m
                      </p>
                    </div>
                    <div>
                      <p className="text-tesla-gray-dark mb-1 font-medium">Y</p>
                      <p className="text-tesla-black font-semibold text-base">
                        {status.robot_status.position.y.toFixed(3)} m
                      </p>
                    </div>
                    <div>
                      <p className="text-tesla-gray-dark mb-1 font-medium">Z</p>
                      <p className="text-tesla-black font-semibold text-base">
                        {status.robot_status.position.z.toFixed(3)} m
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Joint Angles */}
            {status.robot_status.joint_angles && (
              <div>
                <h3 className="text-lg font-medium text-tesla-black mb-tesla-md">
                  Ángulos de Articulación
                </h3>
                <div className="p-tesla-lg bg-gray-50 rounded-md border border-gray-200">
                  <div className="grid grid-cols-6 gap-tesla-sm text-sm">
                    {status.robot_status.joint_angles.map((angle: number, i: number) => (
                      <div key={i}>
                        <p className="text-tesla-gray-dark mb-tesla-xs font-medium">J{i + 1}</p>
                        <p className="text-tesla-black font-semibold">
                          {(angle * (180 / Math.PI)).toFixed(1)}°
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Health Status */}
      {health && (
        <div className="bg-white rounded-lg p-tesla-lg border border-gray-200 shadow-sm">
          <h3 className="text-lg font-medium text-tesla-black mb-tesla-md">Estado de Salud</h3>
          <div className="space-y-tesla-sm">
            <div className="flex items-center justify-between p-tesla-md bg-gray-50 rounded-md border border-gray-200">
              <span className="text-tesla-black font-medium">Estado General</span>
              <span
                className={`font-semibold ${
                  health.status === 'healthy' ? 'text-green-600' : 'text-red-600'
                }`}
              >
                {health.status}
              </span>
            </div>
            {health.components && (
              <div className="space-y-tesla-sm">
                {Object.entries(health.components).map(([key, value]: [string, any]) => (
                  <div
                    key={key}
                    className="flex items-center justify-between p-tesla-md bg-gray-50 rounded-md border border-gray-200"
                  >
                    <span className="text-tesla-black capitalize font-medium">{key}</span>
                    <span
                      className={`font-semibold ${
                        value.status === 'healthy' ? 'text-green-600' : 'text-red-600'
                      }`}
                    >
                      {value.status || 'unknown'}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Statistics */}
      {statistics && (
        <div className="bg-white rounded-lg p-tesla-lg border border-gray-200 shadow-sm">
          <h3 className="text-lg font-medium text-tesla-black mb-tesla-md">Estadísticas</h3>
          <div className="space-y-tesla-sm">
            {Object.entries(statistics).map(([key, value]: [string, any]) => (
              <details key={key} className="p-tesla-md bg-gray-50 rounded-md border border-gray-200">
                <summary className="text-tesla-black font-medium cursor-pointer hover:text-tesla-blue transition-colors">
                  {key.replace(/_/g, ' ').toUpperCase()}
                </summary>
                <pre className="mt-tesla-sm text-xs text-tesla-gray-dark overflow-x-auto bg-white p-tesla-sm rounded border border-gray-200">
                  {JSON.stringify(value, null, 2)}
                </pre>
              </details>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

