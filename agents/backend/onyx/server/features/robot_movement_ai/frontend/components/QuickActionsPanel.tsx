'use client';

import { useState } from 'react';
import { Zap, Plus, Trash2, Edit } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface QuickAction {
  id: string;
  name: string;
  icon: string;
  action: string;
  color: string;
}

export default function QuickActionsPanel() {
  const [actions, setActions] = useState<QuickAction[]>([
    {
      id: '1',
      name: 'Home',
      icon: '🏠',
      action: 'moveToHome',
      color: 'bg-blue-500',
    },
    {
      id: '2',
      name: 'Stop',
      icon: '⏹️',
      action: 'stop',
      color: 'bg-red-500',
    },
    {
      id: '3',
      name: 'Reset',
      icon: '🔄',
      action: 'reset',
      color: 'bg-yellow-500',
    },
    {
      id: '4',
      name: 'Calibrate',
      icon: '🎯',
      action: 'calibrate',
      color: 'bg-green-500',
    },
  ]);
  const [showAdd, setShowAdd] = useState(false);
  const [newName, setNewName] = useState('');
  const [newIcon, setNewIcon] = useState('');
  const [newAction, setNewAction] = useState('');

  const handleAdd = () => {
    if (!newName.trim() || !newAction.trim()) {
      toast.error('Nombre y acción son requeridos');
      return;
    }

    const newQuickAction: QuickAction = {
      id: Date.now().toString(),
      name: newName,
      icon: newIcon || '⚡',
      action: newAction,
      color: 'bg-primary-500',
    };

    setActions([...actions, newQuickAction]);
    setNewName('');
    setNewIcon('');
    setNewAction('');
    setShowAdd(false);
    toast.success('Acción rápida creada');
  };

  const handleDelete = (id: string) => {
    setActions(actions.filter((a) => a.id !== id));
    toast.success('Acción eliminada');
  };

  const handleExecute = (action: string) => {
    toast.info(`Ejecutando: ${action}`);
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Panel de Acciones Rápidas</h3>
          </div>
          <button
            onClick={() => setShowAdd(!showAdd)}
            className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Nueva Acción
          </button>
        </div>

        {/* Add Form */}
        {showAdd && (
          <div className="mb-6 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
            <h4 className="text-sm font-medium text-white mb-3">Crear Nueva Acción Rápida</h4>
            <div className="space-y-3">
              <input
                type="text"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="Nombre"
                className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
              <input
                type="text"
                value={newIcon}
                onChange={(e) => setNewIcon(e.target.value)}
                placeholder="Icono (emoji)"
                className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
              <input
                type="text"
                value={newAction}
                onChange={(e) => setNewAction(e.target.value)}
                placeholder="Acción (ej: moveToHome)"
                className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500 font-mono text-sm"
              />
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
                    setNewIcon('');
                    setNewAction('');
                  }}
                  className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
                >
                  Cancelar
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Actions Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {actions.map((action) => (
            <div
              key={action.id}
              className="p-4 bg-gray-700/50 rounded-lg border border-gray-600 hover:border-primary-500/50 transition-colors cursor-pointer group"
              onClick={() => handleExecute(action.action)}
            >
              <div className="flex flex-col items-center text-center">
                <div className={`w-16 h-16 rounded-full ${action.color} flex items-center justify-center text-2xl mb-2 group-hover:scale-110 transition-transform`}>
                  {action.icon}
                </div>
                <h4 className="font-semibold text-white mb-1">{action.name}</h4>
                <p className="text-xs text-gray-400 font-mono">{action.action}</p>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDelete(action.id);
                  }}
                  className="mt-2 px-2 py-1 bg-red-600 hover:bg-red-700 text-white text-xs rounded transition-colors opacity-0 group-hover:opacity-100"
                >
                  <Trash2 className="w-3 h-3" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}


