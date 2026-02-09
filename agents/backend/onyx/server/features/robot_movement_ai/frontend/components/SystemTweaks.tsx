'use client';

import { useState } from 'react';
import { Sliders, Save, RotateCcw } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface Tweak {
  id: string;
  name: string;
  description: string;
  category: 'performance' | 'ui' | 'network' | 'security';
  value: number;
  min: number;
  max: number;
  unit: string;
}

export default function SystemTweaks() {
  const [tweaks, setTweaks] = useState<Tweak[]>([
    {
      id: '1',
      name: 'Intervalo de Actualización',
      description: 'Frecuencia de actualización de la UI',
      category: 'performance',
      value: 1000,
      min: 100,
      max: 5000,
      unit: 'ms',
    },
    {
      id: '2',
      name: 'Tamaño de Fuente',
      description: 'Tamaño de fuente de la interfaz',
      category: 'ui',
      value: 14,
      min: 10,
      max: 20,
      unit: 'px',
    },
    {
      id: '3',
      name: 'Timeout de Red',
      description: 'Tiempo máximo de espera para requests',
      category: 'network',
      value: 5000,
      min: 1000,
      max: 30000,
      unit: 'ms',
    },
    {
      id: '4',
      name: 'Nivel de Logging',
      description: 'Nivel de detalle de los logs',
      category: 'security',
      value: 2,
      min: 0,
      max: 5,
      unit: '',
    },
  ]);

  const handleSave = () => {
    toast.success('Ajustes guardados');
  };

  const handleReset = () => {
    toast.info('Ajustes restaurados a valores por defecto');
  };

  const updateTweak = (id: string, value: number) => {
    setTweaks((prev) =>
      prev.map((t) => (t.id === id ? { ...t, value } : t))
    );
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'performance':
        return 'bg-blue-500/20 text-blue-400 border-blue-500/50';
      case 'ui':
        return 'bg-purple-500/20 text-purple-400 border-purple-500/50';
      case 'network':
        return 'bg-green-500/20 text-green-400 border-green-500/50';
      default:
        return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50';
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Sliders className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Ajustes del Sistema</h3>
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleReset}
              className="px-3 py-1 bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg transition-colors flex items-center gap-2 text-sm"
            >
              <RotateCcw className="w-4 h-4" />
              Restaurar
            </button>
            <button
              onClick={handleSave}
              className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors flex items-center gap-2"
            >
              <Save className="w-4 h-4" />
              Guardar
            </button>
          </div>
        </div>

        {/* Tweaks List */}
        <div className="space-y-4">
          {tweaks.map((tweak) => (
            <div
              key={tweak.id}
              className="p-4 bg-gray-700/50 rounded-lg border border-gray-600"
            >
              <div className="flex items-center gap-2 mb-2">
                <h4 className="font-semibold text-white">{tweak.name}</h4>
                <span className={`px-2 py-0.5 rounded text-xs border ${getCategoryColor(tweak.category)}`}>
                  {tweak.category}
                </span>
              </div>
              <p className="text-sm text-gray-300 mb-3">{tweak.description}</p>
              <div className="flex items-center gap-4">
                <input
                  type="range"
                  min={tweak.min}
                  max={tweak.max}
                  value={tweak.value}
                  onChange={(e) => updateTweak(tweak.id, Number(e.target.value))}
                  className="flex-1"
                />
                <div className="w-24 text-right">
                  <span className="text-lg font-bold text-white">
                    {tweak.value}
                  </span>
                  <span className="text-sm text-gray-400 ml-1">{tweak.unit}</span>
                </div>
              </div>
              <div className="flex items-center justify-between text-xs text-gray-400 mt-1">
                <span>{tweak.min}{tweak.unit}</span>
                <span>{tweak.max}{tweak.unit}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}


