'use client';

import { useState } from 'react';
import { Bell, Plus, Trash2, Edit, AlertTriangle, Info, CheckCircle } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface Alert {
  id: string;
  name: string;
  condition: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  enabled: boolean;
  lastTriggered?: Date;
}

export default function SmartAlerts() {
  const [alerts, setAlerts] = useState<Alert[]>([
    {
      id: '1',
      name: 'Velocidad Alta',
      condition: 'velocity > 0.8',
      severity: 'warning',
      enabled: true,
      lastTriggered: new Date(Date.now() - 3600000),
    },
    {
      id: '2',
      name: 'Batería Baja',
      condition: 'battery < 20',
      severity: 'error',
      enabled: true,
    },
    {
      id: '3',
      name: 'Conexión Perdida',
      condition: 'connection == false',
      severity: 'critical',
      enabled: true,
    },
  ]);
  const [showAdd, setShowAdd] = useState(false);
  const [newName, setNewName] = useState('');
  const [newCondition, setNewCondition] = useState('');
  const [newSeverity, setNewSeverity] = useState<'info' | 'warning' | 'error' | 'critical'>('warning');

  const handleAdd = () => {
    if (!newName.trim() || !newCondition.trim()) {
      toast.error('Nombre y condición son requeridos');
      return;
    }

    const newAlert: Alert = {
      id: Date.now().toString(),
      name: newName,
      condition: newCondition,
      severity: newSeverity,
      enabled: true,
    };

    setAlerts([...alerts, newAlert]);
    setNewName('');
    setNewCondition('');
    setShowAdd(false);
    toast.success('Alerta creada');
  };

  const handleDelete = (id: string) => {
    setAlerts(alerts.filter((a) => a.id !== id));
    toast.success('Alerta eliminada');
  };

  const handleToggle = (id: string) => {
    setAlerts((prev) =>
      prev.map((a) => (a.id === id ? { ...a, enabled: !a.enabled } : a))
    );
    toast.success('Alerta actualizada');
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
        return <AlertTriangle className="w-5 h-5 text-red-500" />;
      case 'error':
        return <AlertTriangle className="w-5 h-5 text-red-400" />;
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-yellow-400" />;
      default:
        return <Info className="w-5 h-5 text-blue-400" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-500/20 border-red-500/50';
      case 'error':
        return 'bg-red-500/10 border-red-500/30';
      case 'warning':
        return 'bg-yellow-500/10 border-yellow-500/30';
      default:
        return 'bg-blue-500/10 border-blue-500/30';
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Bell className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Alertas Inteligentes</h3>
          </div>
          <button
            onClick={() => setShowAdd(!showAdd)}
            className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Nueva Alerta
          </button>
        </div>

        {/* Add Form */}
        {showAdd && (
          <div className="mb-6 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
            <h4 className="text-sm font-medium text-white mb-3">Crear Nueva Alerta</h4>
            <div className="space-y-3">
              <input
                type="text"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="Nombre de la alerta"
                className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
              <input
                type="text"
                value={newCondition}
                onChange={(e) => setNewCondition(e.target.value)}
                placeholder="Condición (ej: velocity > 0.8)"
                className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500 font-mono text-sm"
              />
              <select
                value={newSeverity}
                onChange={(e) => setNewSeverity(e.target.value as any)}
                className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
              >
                <option value="info">Info</option>
                <option value="warning">Advertencia</option>
                <option value="error">Error</option>
                <option value="critical">Crítico</option>
              </select>
              <div className="flex gap-2">
                <button
                  onClick={handleAdd}
                  className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
                >
                  Crear
                </button>
                <button
                  onClick={() => {
                    setShowAdd(false);
                    setNewName('');
                    setNewCondition('');
                  }}
                  className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
                >
                  Cancelar
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Alerts List */}
        <div className="space-y-3">
          {alerts.map((alert) => (
            <div
              key={alert.id}
              className={`p-4 rounded-lg border ${getSeverityColor(alert.severity)} ${
                !alert.enabled ? 'opacity-50' : ''
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-3 flex-1">
                  {getSeverityIcon(alert.severity)}
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <h4 className="font-semibold text-white">{alert.name}</h4>
                      <span className="px-2 py-0.5 bg-gray-700 text-gray-300 text-xs rounded capitalize">
                        {alert.severity}
                      </span>
                    </div>
                    <p className="text-sm text-gray-300 font-mono mb-1">{alert.condition}</p>
                    {alert.lastTriggered && (
                      <p className="text-xs text-gray-400">
                        Última activación: {alert.lastTriggered.toLocaleString('es-ES')}
                      </p>
                    )}
                  </div>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => handleToggle(alert.id)}
                    className={`px-3 py-1 text-sm rounded-lg transition-colors ${
                      alert.enabled
                        ? 'bg-green-600 hover:bg-green-700 text-white'
                        : 'bg-gray-600 hover:bg-gray-700 text-gray-300'
                    }`}
                  >
                    {alert.enabled ? 'Activa' : 'Inactiva'}
                  </button>
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


