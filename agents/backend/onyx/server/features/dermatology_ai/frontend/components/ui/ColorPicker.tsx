'use client';

import React, { useState } from 'react';
import { clsx } from 'clsx';

interface ColorPickerProps {
  value?: string;
  onChange?: (color: string) => void;
  className?: string;
  showPresets?: boolean;
  presets?: string[];
}

export const ColorPicker: React.FC<ColorPickerProps> = ({
  value = '#000000',
  onChange,
  className,
  showPresets = true,
  presets = [
    '#000000',
    '#ffffff',
    '#ef4444',
    '#f59e0b',
    '#eab308',
    '#22c55e',
    '#0ea5e9',
    '#3b82f6',
    '#6366f1',
    '#8b5cf6',
    '#ec4899',
  ],
}) => {
  const [color, setColor] = useState(value);

  const handleChange = (newColor: string) => {
    setColor(newColor);
    onChange?.(newColor);
  };

  return (
    <div className={clsx('flex flex-col space-y-4', className)}>
      <div className="flex items-center space-x-4">
        <input
          type="color"
          value={color}
          onChange={(e) => handleChange(e.target.value)}
          className="w-16 h-16 rounded-lg border border-gray-300 dark:border-gray-600 cursor-pointer"
        />
        <div className="flex-1">
          <input
            type="text"
            value={color}
            onChange={(e) => handleChange(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white font-mono text-sm"
            placeholder="#000000"
          />
        </div>
      </div>
      {showPresets && (
        <div>
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 block">
            Colores predefinidos
          </label>
          <div className="flex flex-wrap gap-2">
            {presets.map((preset) => (
              <button
                key={preset}
                onClick={() => handleChange(preset)}
                className={clsx(
                  'w-8 h-8 rounded border-2 transition-all',
                  color === preset
                    ? 'border-primary-600 scale-110'
                    : 'border-gray-300 dark:border-gray-600 hover:scale-105'
                )}
                style={{ backgroundColor: preset }}
                aria-label={`Seleccionar color ${preset}`}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};


