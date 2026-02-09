'use client';

import { useState, useEffect } from 'react';
import { Server, CheckCircle, XCircle, AlertTriangle, RefreshCw } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface Service {
  id: string;
  name: string;
  url: string;
  status: 'online' | 'offline' | 'degraded';
  responseTime?: number;
  lastCheck: Date;
}

export default function ServiceStatus() {
  const [services, setServices] = useState<Service[]>([
    {
      id: '1',
      name: 'API Principal',
      url: 'https://api.example.com',
      status: 'online',
      responseTime: 45,
      lastCheck: new Date(),
    },
    {
      id: '2',
      name: 'Base de Datos',
      url: 'localhost:5432',
      status: 'online',
      responseTime: 12,
      lastCheck: new Date(),
    },
    {
      id: '3',
      name: 'Servicio de Notificaciones',
      url: 'https://notifications.example.com',
      status: 'degraded',
      responseTime: 250,
      lastCheck: new Date(),
    },
    {
      id: '4',
      name: 'Servicio Externo',
      url: 'https://external.example.com',
      status: 'offline',
      lastCheck: new Date(Date.now() - 3600000),
    },
  ]);

  const handleRefresh = async (id: string) => {
    setServices((prev) =>
      prev.map((s) =>
        s.id === id ? { ...s, lastCheck: new Date() } : s
      )
    );

    // Simulate check
    await new Promise((resolve) => setTimeout(resolve, 1000));

    const statuses: ('online' | 'offline' | 'degraded')[] = ['online', 'online', 'degraded', 'offline'];
    const randomStatus = statuses[Math.floor(Math.random() * statuses.length)];

    setServices((prev) =>
      prev.map((s) =>
        s.id === id
          ? {
              ...s,
              status: randomStatus,
              responseTime: randomStatus === 'online' ? Math.floor(Math.random() * 100) : undefined,
            }
          : s
      )
    );

    toast.success('Estado actualizado');
  };

  const handleRefreshAll = async () => {
    toast.info('Verificando todos los servicios...');
    for (const service of services) {
      await handleRefresh(service.id);
      await new Promise((resolve) => setTimeout(resolve, 300));
    }
    toast.success('Verificación completa');
  };

  useEffect(() => {
    const interval = setInterval(() => {
      handleRefreshAll();
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'online':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'degraded':
        return <AlertTriangle className="w-5 h-5 text-yellow-400" />;
      case 'offline':
        return <XCircle className="w-5 h-5 text-red-400" />;
      default:
        return <XCircle className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online':
        return 'bg-green-500/10 border-green-500/50';
      case 'degraded':
        return 'bg-yellow-500/10 border-yellow-500/50';
      case 'offline':
        return 'bg-red-500/10 border-red-500/50';
      default:
        return 'bg-gray-700/50 border-gray-600';
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Server className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Estado de Servicios</h3>
          </div>
          <button
            onClick={handleRefreshAll}
            className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <RefreshCw className="w-4 h-4" />
            Actualizar Todo
          </button>
        </div>

        {/* Services List */}
        <div className="space-y-3">
          {services.map((service) => (
            <div
              key={service.id}
              className={`p-4 rounded-lg border ${getStatusColor(service.status)}`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-3 flex-1">
                  {getStatusIcon(service.status)}
                  <div className="flex-1">
                    <h4 className="font-semibold text-white mb-1">{service.name}</h4>
                    <p className="text-sm text-gray-300 mb-1">{service.url}</p>
                    {service.responseTime && (
                      <p className="text-xs text-gray-400">
                        Tiempo de respuesta: {service.responseTime}ms
                      </p>
                    )}
                    <p className="text-xs text-gray-400">
                      Última verificación: {service.lastCheck.toLocaleTimeString('es-ES')}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className={`px-2 py-1 rounded text-xs capitalize ${
                    service.status === 'online'
                      ? 'bg-green-500/20 text-green-400'
                      : service.status === 'degraded'
                      ? 'bg-yellow-500/20 text-yellow-400'
                      : 'bg-red-500/20 text-red-400'
                  }`}>
                    {service.status}
                  </span>
                  <button
                    onClick={() => handleRefresh(service.id)}
                    className="p-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
                  >
                    <RefreshCw className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}


