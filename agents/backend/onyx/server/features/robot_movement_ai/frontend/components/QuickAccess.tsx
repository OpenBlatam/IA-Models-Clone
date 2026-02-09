'use client';

import { useState } from 'react';
import { useLocalStorage } from '@/lib/hooks/useLocalStorage';
import { Zap, Plus, X, Edit } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface QuickAccessItem {
  id: string;
  name: string;
  action: string;
  icon?: string;
}

export default function QuickAccess() {
  const [items, setItems] = useLocalStorage<QuickAccessItem[]>('quick-access', [
    { id: '1', name: 'Home', action: 'moveTo(0,0,0)' },
    { id: '2', name: 'Stop', action: 'stop()' },
    { id: '3', name: 'Status', action: 'showTab("status")' },
  ]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [newItem, setNewItem] = useState({ name: '', action: '' });

  const handleAddItem = () => {
    if (!newItem.name.trim() || !newItem.action.trim()) {
      toast.error('Completa todos los campos');
      return;
    }
    const item: QuickAccessItem = {
      id: Date.now().toString(),
      ...newItem,
    };
    setItems([...items, item]);
    setNewItem({ name: '', action: '' });
    toast.success('Acceso rápido agregado');
  };

  const handleDeleteItem = (id: string) => {
    setItems(items.filter((item) => item.id !== id));
    toast.success('Acceso rápido eliminado');
  };

  const handleExecute = (action: string) => {
    // Would execute action
    toast.info(`Ejecutando: ${action}`);
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Zap className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Acceso Rápido</h3>
        </div>

        {/* Add Item */}
        <div className="mb-6 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
          <h4 className="text-sm font-semibold text-white mb-3">Agregar Acceso Rápido</h4>
          <div className="space-y-3">
            <input
              type="text"
              value={newItem.name}
              onChange={(e) => setNewItem({ ...newItem, name: e.target.value })}
              placeholder="Nombre"
              className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
            <input
              type="text"
              value={newItem.action}
              onChange={(e) => setNewItem({ ...newItem, action: e.target.value })}
              placeholder="Acción (ej: moveTo(0,0,0))"
              className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white font-mono text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
            <button
              onClick={handleAddItem}
              className="w-full px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors flex items-center justify-center gap-2"
            >
              <Plus className="w-4 h-4" />
              Agregar
            </button>
          </div>
        </div>

        {/* Items Grid */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
          {items.map((item) => (
            <div
              key={item.id}
              className="p-4 bg-gray-700/50 rounded-lg border border-gray-600 hover:border-primary-500/50 transition-colors cursor-pointer group"
              onClick={() => handleExecute(item.action)}
            >
              <div className="flex items-start justify-between mb-2">
                <h4 className="font-semibold text-white text-sm">{item.name}</h4>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDeleteItem(item.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 text-gray-400 hover:text-red-400 transition-colors"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
              <p className="text-xs text-gray-400 font-mono truncate">{item.action}</p>
            </div>
          ))}
        </div>

        {items.length === 0 && (
          <div className="text-center py-12 text-gray-400">
            <Zap className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No hay accesos rápidos configurados</p>
          </div>
        )}
      </div>
    </div>
  );
}


