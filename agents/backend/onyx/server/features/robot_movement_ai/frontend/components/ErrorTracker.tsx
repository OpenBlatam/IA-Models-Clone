'use client';

import { useState } from 'react';
import { Bug, AlertCircle, XCircle, CheckCircle } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface Error {
  id: string;
  message: string;
  type: 'error' | 'warning' | 'info';
  source: string;
  timestamp: Date;
  resolved: boolean;
}

export default function ErrorTracker() {
  const [errors, setErrors] = useState<Error[]>([
    {
      id: '1',
      message: 'Error de conexión con el robot',
      type: 'error',
      source: 'Robot Control',
      timestamp: new Date(),
      resolved: false,
    },
    {
      id: '2',
      message: 'Timeout en solicitud API',
      type: 'warning',
      source: 'API Client',
      timestamp: new Date(Date.now() - 3600000),
      resolved: false,
    },
    {
      id: '3',
      message: 'Batería baja detectada',
      type: 'warning',
      source: 'Battery Monitor',
      timestamp: new Date(Date.now() - 7200000),
      resolved: true,
    },
    {
      id: '4',
      message: 'Actualización de firmware disponible',
      type: 'info',
      source: 'System',
      timestamp: new Date(Date.now() - 10800000),
      resolved: false,
    },
  ]);

  const handleResolve = (id: string) => {
    setErrors((prev) =>
      prev.map((e) => (e.id === id ? { ...e, resolved: true } : e))
    );
    toast.success('Error resuelto');
  };

  const handleDelete = (id: string) => {
    setErrors(errors.filter((e) => e.id !== id));
    toast.success('Error eliminado');
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'error':
        return <XCircle className="w-5 h-5 text-red-400" />;
      case 'warning':
        return <AlertCircle className="w-5 h-5 text-yellow-400" />;
      default:
        return <CheckCircle className="w-5 h-5 text-blue-400" />;
    }
  };

  const getTypeColor = (type: string, resolved: boolean) => {
    if (resolved) return 'bg-gray-700/50 border-gray-600 opacity-50';
    switch (type) {
      case 'error':
        return 'bg-red-500/10 border-red-500/50';
      case 'warning':
        return 'bg-yellow-500/10 border-yellow-500/50';
      default:
        return 'bg-blue-500/10 border-blue-500/50';
    }
  };

  const unresolvedCount = errors.filter((e) => !e.resolved).length;

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Bug className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Rastreador de Errores</h3>
          </div>
          {unresolvedCount > 0 && (
            <span className="px-3 py-1 bg-red-500/20 text-red-400 rounded-lg text-sm font-semibold">
              {unresolvedCount} sin resolver
            </span>
          )}
        </div>

        {/* Errors List */}
        <div className="space-y-3">
          {errors.map((error) => (
            <div
              key={error.id}
              className={`p-4 rounded-lg border ${getTypeColor(error.type, error.resolved)}`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-3 flex-1">
                  {getTypeIcon(error.type)}
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <h4 className="font-semibold text-white">{error.message}</h4>
                      {error.resolved && (
                        <span className="px-2 py-0.5 bg-green-500/20 text-green-400 text-xs rounded">
                          Resuelto
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-gray-300 mb-1">Fuente: {error.source}</p>
                    <p className="text-xs text-gray-400">
                      {error.timestamp.toLocaleString('es-ES')}
                    </p>
                  </div>
                </div>
                <div className="flex gap-2">
                  {!error.resolved && (
                    <button
                      onClick={() => handleResolve(error.id)}
                      className="px-3 py-1 bg-green-600 hover:bg-green-700 text-white text-sm rounded-lg transition-colors"
                    >
                      Resolver
                    </button>
                  )}
                  <button
                    onClick={() => handleDelete(error.id)}
                    className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded-lg transition-colors"
                  >
                    Eliminar
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>

        {errors.length === 0 && (
          <div className="text-center py-12 text-gray-400">
            <Bug className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No hay errores registrados</p>
          </div>
        )}
      </div>
    </div>
  );
}


