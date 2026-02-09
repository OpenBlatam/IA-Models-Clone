'use client';

import { useState } from 'react';
import { useThemeStore } from '@/lib/store/themeStore';
import { Palette, Moon, Sun, Monitor } from 'lucide-react';

export default function ThemeCustomizer() {
  const { theme, setTheme } = useThemeStore();
  const [accentColor, setAccentColor] = useState('#0EA5E9');

  const accentColors = [
    { name: 'Azul', value: '#0EA5E9' },
    { name: 'Verde', value: '#10B981' },
    { name: 'Púrpura', value: '#8B5CF6' },
    { name: 'Rosa', value: '#EC4899' },
    { name: 'Naranja', value: '#F59E0B' },
    { name: 'Rojo', value: '#EF4444' },
  ];

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Palette className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Personalización de Tema</h3>
        </div>

        {/* Theme Mode */}
        <div className="mb-6">
          <h4 className="text-sm font-medium text-gray-300 mb-3">Modo de Tema</h4>
          <div className="grid grid-cols-3 gap-3">
            <button
              onClick={() => setTheme('light')}
              className={`p-4 rounded-lg border-2 transition-colors flex flex-col items-center gap-2 ${
                theme === 'light'
                  ? 'border-primary-500 bg-primary-500/10'
                  : 'border-gray-600 bg-gray-700/50 hover:border-gray-500'
              }`}
            >
              <Sun className="w-6 h-6 text-yellow-400" />
              <span className="text-white text-sm">Claro</span>
            </button>
            <button
              onClick={() => setTheme('dark')}
              className={`p-4 rounded-lg border-2 transition-colors flex flex-col items-center gap-2 ${
                theme === 'dark'
                  ? 'border-primary-500 bg-primary-500/10'
                  : 'border-gray-600 bg-gray-700/50 hover:border-gray-500'
              }`}
            >
              <Moon className="w-6 h-6 text-blue-400" />
              <span className="text-white text-sm">Oscuro</span>
            </button>
            <button
              onClick={() => setTheme('system')}
              className={`p-4 rounded-lg border-2 transition-colors flex flex-col items-center gap-2 ${
                theme === 'system'
                  ? 'border-primary-500 bg-primary-500/10'
                  : 'border-gray-600 bg-gray-700/50 hover:border-gray-500'
              }`}
            >
              <Monitor className="w-6 h-6 text-gray-400" />
              <span className="text-white text-sm">Sistema</span>
            </button>
          </div>
        </div>

        {/* Accent Color */}
        <div>
          <h4 className="text-sm font-medium text-gray-300 mb-3">Color de Acento</h4>
          <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
            {accentColors.map((color) => (
              <button
                key={color.value}
                onClick={() => setAccentColor(color.value)}
                className={`p-4 rounded-lg border-2 transition-colors ${
                  accentColor === color.value
                    ? 'border-white ring-2 ring-offset-2 ring-offset-gray-800 ring-white'
                    : 'border-gray-600 hover:border-gray-500'
                }`}
                style={{ backgroundColor: color.value }}
                title={color.name}
              />
            ))}
          </div>
          <p className="mt-3 text-xs text-gray-400">
            Color seleccionado: {accentColors.find((c) => c.value === accentColor)?.name}
          </p>
        </div>

        {/* Preview */}
        <div className="mt-6 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
          <h4 className="text-sm font-medium text-gray-300 mb-3">Vista Previa</h4>
          <div className="space-y-2">
            <div
              className="px-4 py-2 rounded-lg text-white"
              style={{ backgroundColor: accentColor }}
            >
              Botón de ejemplo
            </div>
            <div className="p-3 rounded-lg border" style={{ borderColor: accentColor }}>
              <p className="text-white">Borde de ejemplo</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}


