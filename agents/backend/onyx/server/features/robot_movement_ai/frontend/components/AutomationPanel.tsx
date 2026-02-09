'use client';

import { useState } from 'react';
import { useLocalStorage } from '@/lib/hooks/useLocalStorage';
import { Bot, Plus, Trash2, Play, Pause, Edit } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface Automation {
  id: string;
  name: string;
  description: string;
  trigger: 'time' | 'event' | 'condition';
  action: string;
  enabled: boolean;
  schedule?: string;
}

export default function AutomationPanel() {
  const [automations, setAutomations] = useLocalStorage<Automation[]>('automations', [
    {
      id: '1',
      name: 'Backup Diario',
      description: 'Crear backup automático cada día a las 2 AM',
      trigger: 'time',
      action: 'createBackup()',
      enabled: true,
      schedule: '0 2 * * *',
    },
    {
      id: '2',
      name: 'Verificación de Salud',
      description: 'Verificar salud del sistema cada hora',
      trigger: 'time',
      action: 'checkHealth()',
      enabled: true,
      schedule: '0 * * * *',
    },
  ]);
  const [newAutomation, setNewAutomation] = useState({
    name: '',
    description: '',
    trigger: 'time' as const,
    action: '',
    schedule: '',
  });

  const handleCreateAutomation = () => {
    if (!newAutomation.name.trim() || !newAutomation.action.trim()) {
      toast.error('Completa todos los campos requeridos');
      return;
    }
    const automation: Automation = {
      id: Date.now().toString(),
      ...newAutomation,
      enabled: false,
    };
    setAutomations([...automations, automation]);
    setNewAutomation({ name: '', description: '', trigger: 'time', action: '', schedule: '' });
    toast.success('Automatización creada');
  };

  const handleToggleAutomation = (id: string) => {
    setAutomations((prev) =>
      prev.map((a) => (a.id === id ? { ...a, enabled: !a.enabled } : a))
    );
    const automation = automations.find((a) => a.id === id);
    toast.success(`${automation?.name} ${automation?.enabled ? 'desactivada' : 'activada'}`);
  };

  const handleDeleteAutomation = (id: string) => {
    setAutomations(automations.filter((a) => a.id !== id));
    toast.success('Automatización eliminada');
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Bot className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Panel de Automatización</h3>
        </div>

        {/* Create Automation */}
        <div className="mb-6 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
          <h4 className="text-sm font-semibold text-white mb-3">Crear Automatización</h4>
          <div className="space-y-3">
            <input
              type="text"
              value={newAutomation.name}
              onChange={(e) => setNewAutomation({ ...newAutomation, name: e.target.value })}
              placeholder="Nombre de la automatización"
              className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
            <textarea
              value={newAutomation.description}
              onChange={(e) => setNewAutomation({ ...newAutomation, description: e.target.value })}
              placeholder="Descripción"
              rows={2}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500 resize-none"
            />
            <select
              value={newAutomation.trigger}
              onChange={(e) => setNewAutomation({ ...newAutomation, trigger: e.target.value as any })}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              <option value="time">Por tiempo</option>
              <option value="event">Por evento</option>
              <option value="condition">Por condición</option>
            </select>
            {newAutomation.trigger === 'time' && (
              <input
                type="text"
                value={newAutomation.schedule}
                onChange={(e) => setNewAutomation({ ...newAutomation, schedule: e.target.value })}
                placeholder="Cron schedule (ej: 0 2 * * *)"
                className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white font-mono text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
            )}
            <input
              type="text"
              value={newAutomation.action}
              onChange={(e) => setNewAutomation({ ...newAutomation, action: e.target.value })}
              placeholder="Acción a ejecutar"
              className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white font-mono text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
            <button
              onClick={handleCreateAutomation}
              className="w-full px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors flex items-center justify-center gap-2"
            >
              <Plus className="w-4 h-4" />
              Crear Automatización
            </button>
          </div>
        </div>

        {/* Automations List */}
        <div className="space-y-3">
          {automations.length === 0 ? (
            <div className="text-center py-12 text-gray-400">
              <Bot className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>No hay automatizaciones configuradas</p>
            </div>
          ) : (
            automations.map((automation) => (
              <div
                key={automation.id}
                className={`p-4 rounded-lg border ${
                  automation.enabled
                    ? 'bg-gray-700/50 border-gray-600'
                    : 'bg-gray-800/50 border-gray-700 opacity-60'
                }`}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <h4 className="font-semibold text-white">{automation.name}</h4>
                      {automation.enabled && (
                        <span className="px-2 py-0.5 bg-green-500/20 text-green-400 text-xs rounded">
                          Activa
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-gray-300 mb-2">{automation.description}</p>
                    <div className="flex gap-4 text-xs text-gray-400">
                      <span>Trigger: {automation.trigger}</span>
                      {automation.schedule && <span>Schedule: {automation.schedule}</span>}
                    </div>
                    <p className="text-xs text-gray-400 mt-1 font-mono">{automation.action}</p>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleToggleAutomation(automation.id)}
                      className={`p-2 rounded transition-colors ${
                        automation.enabled
                          ? 'bg-yellow-600 hover:bg-yellow-700 text-white'
                          : 'bg-green-600 hover:bg-green-700 text-white'
                      }`}
                      title={automation.enabled ? 'Pausar' : 'Activar'}
                    >
                      {automation.enabled ? (
                        <Pause className="w-4 h-4" />
                      ) : (
                        <Play className="w-4 h-4" />
                      )}
                    </button>
                    <button
                      className="p-2 bg-gray-600 hover:bg-gray-700 text-white rounded transition-colors"
                      title="Editar"
                    >
                      <Edit className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => handleDeleteAutomation(automation.id)}
                      className="p-2 bg-red-600 hover:bg-red-700 text-white rounded transition-colors"
                      title="Eliminar"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}


