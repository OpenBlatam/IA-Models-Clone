'use client';

import { useState } from 'react';
import { Bell, Plus, Trash2, Edit, CheckCircle, XCircle } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface Alert {
  id: string;
  name: string;
  condition: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  enabled: boolean;
  acknowledged: boolean;
  timestamp?: Date;
}

export default function AlertManager() {
  const [alerts, setAlerts] = useState<Alert[]>([
    {
      id: '1',
      name: 'Velocidad Alta',
      condition: 'velocity > 0.8',
      severity: 'warning',
      enabled: true,
      acknowledged: false,
      timestamp: new Date(),
    },
    {
      id: '2',
      name: 'Batería Baja',
      condition: 'battery < 20',
      severity: 'error',
      enabled: true,
      acknowledged: true,
      timestamp: new Date(Date.now() - 3600000),
    },
    {
      id: '3',
      name: 'Conexión Perdida',
      condition: 'connection == false',
      severity: 'critical',
      enabled: true,
      acknowledged: false,
      timestamp: new Date(Date.now() - 1800000),
    },
  ]);
  const [showAdd, setShowAdd] = useState(false);

  const handleAcknowledge = (id: string) => {
    setAlerts((prev) =>
      prev.map((a) => (a.id === id ? { ...a, acknowledged: true } : a))
    );
    toast.success('Alerta reconocida');
  };

  const handleDelete = (id: string) => {
    setAlerts(alerts.filter((a) => a.id !== id));
    toast.success('Alerta eliminada');
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-500/20 border-red-500/50 text-red-400';
      case 'error':
        return 'bg-red-500/10 border-red-500/30 text-red-400';
      case 'warning':
        return 'bg-yellow-500/10 border-yellow-500/30 text-yellow-400';
      default:
        return 'bg-blue-500/10 border-blue-500/30 text-blue-400';
    }
  };

  const unacknowledgedCount = alerts.filter((a) => !a.acknowledged).length;

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Bell className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Gestor de Alertas</h3>
            {unacknowledgedCount > 0 && (
              <span className="px-2 py-1 bg-red-500/20 text-red-400 rounded-lg text-xs font-semibold">
                {unacknowledgedCount} sin reconocer
              </span>
            )}
          </div>
          <button
            onClick={() => setShowAdd(!showAdd)}
            className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Nueva Alerta
          </button>
        </div>

        {/* Alerts List */}
        <div className="space-y-3">
          {alerts.map((alert) => (
            <div
              key={alert.id}
              className={`p-4 rounded-lg border ${getSeverityColor(alert.severity)} ${
                alert.acknowledged ? 'opacity-60' : ''
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <h4 className="font-semibold text-white">{alert.name}</h4>
                    <span className="px-2 py-0.5 bg-gray-700 text-gray-300 text-xs rounded capitalize">
                      {alert.severity}
                    </span>
                    {alert.acknowledged && (
                      <CheckCircle className="w-4 h-4 text-green-400" />
                    )}
                  </div>
                  <p className="text-sm text-gray-300 font-mono mb-1">{alert.condition}</p>
                  {alert.timestamp && (
                    <p className="text-xs text-gray-400">
                      {alert.timestamp.toLocaleString('es-ES')}
                    </p>
                  )}
                </div>
                <div className="flex gap-2">
                  {!alert.acknowledged && (
                    <button
                      onClick={() => handleAcknowledge(alert.id)}
                      className="px-3 py-1 bg-green-600 hover:bg-green-700 text-white text-sm rounded-lg transition-colors flex items-center gap-2"
                    >
                      <CheckCircle className="w-3 h-3" />
                      Reconocer
                    </button>
                  )}
                  <button
                    onClick={() => handleDelete(alert.id)}
                    className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded-lg transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
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


