/**
 * Preset Selector Component
 * @module robot-3d-view/controls/preset-selector
 */

'use client';

import { memo } from 'react';
import { getPreset, type PresetType } from '../lib/presets';
import { notify } from '../utils/notifications';

/**
 * Props for PresetSelector component
 */
interface PresetSelectorProps {
  onPresetChange: (config: ReturnType<typeof getPreset>) => void;
}

/**
 * Preset Selector Component
 * 
 * Allows users to quickly apply preset configurations.
 * 
 * @param props - Component props
 * @returns Preset selector component
 */
export const PresetSelector = memo(({ onPresetChange }: PresetSelectorProps) => {
  const presets: Array<{ type: PresetType; label: string; description: string }> = [
    { type: 'minimal', label: 'Minimal', description: 'Solo lo esencial' },
    { type: 'standard', label: 'Estándar', description: 'Configuración recomendada' },
    { type: 'detailed', label: 'Detallado', description: 'Todas las características' },
    { type: 'performance', label: 'Rendimiento', description: 'Optimizado para velocidad' },
    { type: 'quality', label: 'Calidad', description: 'Máxima calidad visual' },
  ];

  const handlePresetSelect = (preset: PresetType) => {
    const config = getPreset(preset);
    onPresetChange(config);
    notify.success(`Preset "${presets.find((p) => p.type === preset)?.label}" aplicado`);
  };

  return (
    <div className="absolute top-4 left-1/2 -translate-x-1/2 z-40">
      <div className="bg-gray-800/95 backdrop-blur-md border border-gray-700/50 rounded-lg p-2 shadow-lg">
        <div className="text-[10px] text-gray-400 mb-2 px-2">Presets:</div>
        <div className="flex gap-1 flex-wrap">
          {presets.map(({ type, label, description }) => (
            <button
              key={type}
              onClick={() => handlePresetSelect(type)}
              className="px-2 py-1 text-[10px] rounded bg-gray-700/50 hover:bg-gray-600 transition-all"
              title={description}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
});

PresetSelector.displayName = 'PresetSelector';



