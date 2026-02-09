'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api/client';
import { Bell, AlertTriangle, CheckCircle, XCircle, Info, Settings } from 'lucide-react';
import { format } from 'date-fns';

interface Alert {
  id: string;
  type: 'error' | 'warning' | 'info' | 'success';
  title: string;
  message: string;
  timestamp: string;
  acknowledged: boolean;
  source?: string;
}

export default function AlertsPanel() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [filter, setFilter] = useState<'all' | 'error' | 'warning' | 'info' | 'success'>('all');
  const [autoAcknowledge, setAutoAcknowledge] = useState(false);

  useEffect(() => {
    // Simular alertas (en producción vendría del backend)
    const interval = setInterval(() => {
      if (Math.random() > 0.7) {
        const types: Alert['type'][] = ['error', 'warning', 'info', 'success'];
        const newAlert: Alert = {
          id: `alert-${Date.now()}`,
          type: types[Math.floor(Math.random() * types.length)],
          title: `Alert ${alerts.length + 1}`,
          message: 'This is a sample alert message',
          timestamp: new Date().toISOString(),
          acknowledged: false,
          source: 'robot-system',
        };
        setAlerts((prev) => [newAlert, ...prev.slice(0, 49)]);
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [alerts.length]);

  const filteredAlerts = alerts.filter(
    (alert) => (filter === 'all' || alert.type === filter) && !alert.acknowledged
  );

  const getTypeIcon = (type: Alert['type']) => {
    switch (type) {
      case 'error':
        return <XCircle className="w-5 h-5 text-red-400" />;
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-yellow-400" />;
      case 'success':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      default:
        return <Info className="w-5 h-5 text-blue-400" />;
    }
  };

  const getTypeColor = (type: Alert['type']) => {
    switch (type) {
      case 'error':
        return 'border-red-500/50 bg-red-500/10';
      case 'warning':
        return 'border-yellow-500/50 bg-yellow-500/10';
      case 'success':
        return 'border-green-500/50 bg-green-500/10';
      default:
        return 'border-blue-500/50 bg-blue-500/10';
    }
  };

  const handleAcknowledge = (id: string) => {
    setAlerts((prev) =>
      prev.map((alert) =>
        alert.id === id ? { ...alert, acknowledged: true } : alert
      )
    );
  };

  const handleAcknowledgeAll = () => {
    setAlerts((prev) =>
      prev.map((alert) => ({ ...alert, acknowledged: true }))
    );
  };

  useEffect(() => {
    if (autoAcknowledge && alerts.length > 0) {
      const timer = setTimeout(() => {
        setAlerts((prev) =>
          prev.map((alert) => ({ ...alert, acknowledged: true }))
        );
      }, 10000); // Auto-acknowledge after 10 seconds
      return () => clearTimeout(timer);
    }
  }, [autoAcknowledge, alerts.length]);

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-4 border border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Bell className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Alertas del Sistema</h3>
            <span className="px-2 py-1 bg-red-500/20 text-red-400 rounded text-sm">
              {filteredAlerts.length}
            </span>
          </div>
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-sm text-gray-300">
              <input
                type="checkbox"
                checked={autoAcknowledge}
                onChange={(e) => setAutoAcknowledge(e.target.checked)}
                className="rounded"
              />
              Auto-reconocer
            </label>
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value as any)}
              className="px-3 py-1 bg-gray-700 border border-gray-600 rounded text-white text-sm"
            >
              <option value="all">Todas</option>
              <option value="error">Errores</option>
              <option value="warning">Advertencias</option>
              <option value="info">Info</option>
              <option value="success">Éxito</option>
            </select>
            <button
              onClick={handleAcknowledgeAll}
              className="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition-colors"
            >
              Reconocer Todas
            </button>
          </div>
        </div>
      </div>

      {/* Alerts List */}
      <div className="space-y-2 max-h-[500px] overflow-y-auto">
        {filteredAlerts.length === 0 ? (
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-8 border border-gray-700 text-center text-gray-400">
            No hay alertas activas
          </div>
        ) : (
          filteredAlerts.map((alert) => (
            <div
              key={alert.id}
              className={`p-4 rounded-lg border ${getTypeColor(alert.type)}`}
            >
              <div className="flex items-start gap-3">
                {getTypeIcon(alert.type)}
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <h4 className="font-semibold text-white">{alert.title}</h4>
                    <span className="text-xs text-gray-400">
                      {format(new Date(alert.timestamp), 'HH:mm:ss')}
                    </span>
                  </div>
                  <p className="text-sm text-gray-300 mb-2">{alert.message}</p>
                  {alert.source && (
                    <p className="text-xs text-gray-400">Fuente: {alert.source}</p>
                  )}
                </div>
                <button
                  onClick={() => handleAcknowledge(alert.id)}
                  className="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition-colors"
                >
                  Reconocer
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

