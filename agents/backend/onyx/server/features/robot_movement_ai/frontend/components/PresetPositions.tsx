'use client';

import { useState } from 'react';
import { useRobotStore } from '@/lib/store/robotStore';
import { MapPin, Plus, Trash2, Play, Save } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface Preset {
  id: string;
  name: string;
  position: { x: number; y: number; z: number };
}

export default function PresetPositions() {
  const { moveTo } = useRobotStore();
  const [presets, setPresets] = useState<Preset[]>([
    { id: '1', name: 'Home', position: { x: 0, y: 0, z: 0 } },
    { id: '2', name: 'Punto A', position: { x: 0.5, y: 0.3, z: 0.2 } },
    { id: '3', name: 'Punto B', position: { x: -0.5, y: 0.3, z: 0.2 } },
  ]);
  const [newPreset, setNewPreset] = useState({ name: '', x: 0, y: 0, z: 0 });
  const [showAdd, setShowAdd] = useState(false);

  const handleMoveToPreset = (preset: Preset) => {
    moveTo(preset.position);
    toast.success(`Moviendo a ${preset.name}`);
  };

  const handleAddPreset = () => {
    if (!newPreset.name.trim()) {
      toast.error('El nombre es requerido');
      return;
    }
    const preset: Preset = {
      id: Date.now().toString(),
      name: newPreset.name,
      position: { x: newPreset.x, y: newPreset.y, z: newPreset.z },
    };
    setPresets([...presets, preset]);
    setNewPreset({ name: '', x: 0, y: 0, z: 0 });
    setShowAdd(false);
    toast.success('Preset agregado');
  };

  const handleDeletePreset = (id: string) => {
    setPresets(presets.filter((p) => p.id !== id));
    toast.success('Preset eliminado');
  };

  const handleSaveCurrent = () => {
    // Would get current position from store
    toast.info('Funcionalidad de guardar posición actual próximamente');
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <MapPin className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Posiciones Predefinidas</h3>
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleSaveCurrent}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors flex items-center gap-2"
            >
              <Save className="w-4 h-4" />
              Guardar Actual
            </button>
            <button
              onClick={() => setShowAdd(!showAdd)}
              className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors flex items-center gap-2"
            >
              <Plus className="w-4 h-4" />
              Agregar
            </button>
          </div>
        </div>

        {/* Add New Preset */}
        {showAdd && (
          <div className="mb-6 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
            <h4 className="text-sm font-semibold text-white mb-3">Nueva Posición</h4>
            <div className="space-y-3">
              <input
                type="text"
                value={newPreset.name}
                onChange={(e) => setNewPreset({ ...newPreset, name: e.target.value })}
                placeholder="Nombre del preset"
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
              <div className="grid grid-cols-3 gap-3">
                <input
                  type="number"
                  step="0.01"
                  value={newPreset.x}
                  onChange={(e) => setNewPreset({ ...newPreset, x: parseFloat(e.target.value) || 0 })}
                  placeholder="X"
                  className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
                <input
                  type="number"
                  step="0.01"
                  value={newPreset.y}
                  onChange={(e) => setNewPreset({ ...newPreset, y: parseFloat(e.target.value) || 0 })}
                  placeholder="Y"
                  className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
                <input
                  type="number"
                  step="0.01"
                  value={newPreset.z}
                  onChange={(e) => setNewPreset({ ...newPreset, z: parseFloat(e.target.value) || 0 })}
                  placeholder="Z"
                  className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
              </div>
              <div className="flex gap-2">
                <button
                  onClick={handleAddPreset}
                  className="flex-1 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
                >
                  Guardar
                </button>
                <button
                  onClick={() => {
                    setShowAdd(false);
                    setNewPreset({ name: '', x: 0, y: 0, z: 0 });
                  }}
                  className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
                >
                  Cancelar
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Presets List */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {presets.map((preset) => (
            <div
              key={preset.id}
              className="p-4 bg-gray-700/50 rounded-lg border border-gray-600 hover:border-primary-500/50 transition-colors"
            >
              <div className="flex items-start justify-between mb-3">
                <h4 className="font-semibold text-white">{preset.name}</h4>
                <button
                  onClick={() => handleDeletePreset(preset.id)}
                  className="p-1 text-gray-400 hover:text-red-400 transition-colors"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
              <div className="space-y-1 mb-3">
                <p className="text-sm text-gray-300">
                  X: <span className="font-mono">{preset.position.x.toFixed(3)}</span>
                </p>
                <p className="text-sm text-gray-300">
                  Y: <span className="font-mono">{preset.position.y.toFixed(3)}</span>
                </p>
                <p className="text-sm text-gray-300">
                  Z: <span className="font-mono">{preset.position.z.toFixed(3)}</span>
                </p>
              </div>
              <button
                onClick={() => handleMoveToPreset(preset)}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
              >
                <Play className="w-4 h-4" />
                Mover
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}


