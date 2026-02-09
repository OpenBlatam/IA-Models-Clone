'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api/client';
import { Plug, CheckCircle, XCircle, RefreshCw, Database, Server } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface IntegrationStatus {
  name: string;
  endpoint: string;
  status: 'connected' | 'disconnected' | 'error';
  lastCheck: string;
  responseTime?: number;
}

export default function BackendIntegrations() {
  const [integrations, setIntegrations] = useState<IntegrationStatus[]>([]);
  const [isChecking, setIsChecking] = useState(false);

  const endpoints = [
    { name: 'Robot Control API', endpoint: '/api/v1/status' },
    { name: 'Metrics API', endpoint: '/api/v1/metrics' },
    { name: 'Resources API', endpoint: '/api/v1/resources' },
    { name: 'System API', endpoint: '/api/v1/system/version' },
    { name: 'Analytics API', endpoint: '/api/v1/analytics/learning/statistics' },
    { name: 'Tasks API', endpoint: '/api/v1/tasks/scheduled' },
    { name: 'Notifications API', endpoint: '/api/v1/notifications' },
    { name: 'Monitoring API', endpoint: '/api/v1/monitoring/performance' },
    { name: 'Health API', endpoint: '/health' },
    { name: 'Deep Learning API', endpoint: '/api/v1/deep-learning/models' },
    { name: 'LLM API', endpoint: '/api/v1/llm/models' },
    { name: 'Universal Robot API', endpoint: '/api/v1/universal-robot/list' },
  ];

  const checkIntegration = async (endpoint: string) => {
    const startTime = performance.now();
    try {
      const response = await apiClient.client.get(endpoint, { timeout: 5000 });
      const responseTime = performance.now() - startTime;
      return {
        status: 'connected' as const,
        responseTime: Math.round(responseTime),
      };
    } catch (error) {
      return {
        status: 'error' as const,
        responseTime: undefined,
      };
    }
  };

  const checkAllIntegrations = async () => {
    setIsChecking(true);
    const results = await Promise.all(
      endpoints.map(async (endpoint) => {
        const result = await checkIntegration(endpoint.endpoint);
        return {
          name: endpoint.name,
          endpoint: endpoint.endpoint,
          ...result,
          lastCheck: new Date().toISOString(),
        };
      })
    );
    setIntegrations(results);
    setIsChecking(false);
    toast.success('Integraciones verificadas');
  };

  useEffect(() => {
    checkAllIntegrations();
  }, []);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'connected':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'error':
        return <XCircle className="w-5 h-5 text-red-400" />;
      default:
        return <XCircle className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected':
        return 'border-green-500/50 bg-green-500/10';
      case 'error':
        return 'border-red-500/50 bg-red-500/10';
      default:
        return 'border-gray-500/50 bg-gray-500/10';
    }
  };

  const connectedCount = integrations.filter((i) => i.status === 'connected').length;
  const totalCount = integrations.length;

  return (
    <div className="space-y-6">
      {/* Summary */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Plug className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Integraciones con Backend</h3>
          </div>
          <button
            onClick={checkAllIntegrations}
            disabled={isChecking}
            className="flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${isChecking ? 'animate-spin' : ''}`} />
            Verificar
          </button>
        </div>

        <div className="grid grid-cols-3 gap-4">
          <div className="p-4 bg-gray-700/50 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Server className="w-5 h-5 text-blue-400" />
              <span className="text-gray-400">Total</span>
            </div>
            <p className="text-2xl font-bold text-white">{totalCount}</p>
          </div>
          <div className="p-4 bg-gray-700/50 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <CheckCircle className="w-5 h-5 text-green-400" />
              <span className="text-gray-400">Conectadas</span>
            </div>
            <p className="text-2xl font-bold text-green-400">{connectedCount}</p>
          </div>
          <div className="p-4 bg-gray-700/50 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <XCircle className="w-5 h-5 text-red-400" />
              <span className="text-gray-400">Desconectadas</span>
            </div>
            <p className="text-2xl font-bold text-red-400">{totalCount - connectedCount}</p>
          </div>
        </div>
      </div>

      {/* Integrations List */}
      <div className="space-y-3">
        {integrations.map((integration, index) => (
          <div
            key={index}
            className={`p-4 rounded-lg border ${getStatusColor(integration.status)}`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3 flex-1">
                {getStatusIcon(integration.status)}
                <div className="flex-1">
                  <h4 className="font-semibold text-white">{integration.name}</h4>
                  <p className="text-sm text-gray-400 font-mono">{integration.endpoint}</p>
                  {integration.responseTime && (
                    <p className="text-xs text-gray-500 mt-1">
                      Tiempo de respuesta: {integration.responseTime}ms
                    </p>
                  )}
                </div>
              </div>
              <div className="text-right">
                <p className="text-xs text-gray-400">
                  {new Date(integration.lastCheck).toLocaleTimeString('es-ES')}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}


