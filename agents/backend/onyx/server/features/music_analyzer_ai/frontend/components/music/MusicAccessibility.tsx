'use client';

import { useState } from 'react';
import { Accessibility, Eye, Volume2, Mouse } from 'lucide-react';

export function MusicAccessibility() {
  const [settings, setSettings] = useState({
    highContrast: false,
    largeText: false,
    screenReader: false,
    reducedMotion: false,
  });

  const handleSettingChange = (key: string, value: boolean) => {
    setSettings({ ...settings, [key]: value });
    // Aplicar cambios de accesibilidad
    if (key === 'highContrast') {
      document.documentElement.classList.toggle('high-contrast', value);
    }
    if (key === 'largeText') {
      document.documentElement.classList.toggle('large-text', value);
    }
    if (key === 'reducedMotion') {
      document.documentElement.classList.toggle('reduced-motion', value);
    }
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Accessibility className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Accesibilidad</h3>
      </div>

      <div className="space-y-4">
        <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
          <div className="flex items-center gap-3">
            <Eye className="w-5 h-5 text-purple-300" />
            <div>
              <p className="text-white font-medium">Alto Contraste</p>
              <p className="text-sm text-gray-400">Mejora la visibilidad</p>
            </div>
          </div>
          <input
            type="checkbox"
            checked={settings.highContrast}
            onChange={(e) => handleSettingChange('highContrast', e.target.checked)}
            className="w-5 h-5 rounded accent-purple-500"
          />
        </div>

        <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
          <div className="flex items-center gap-3">
            <Eye className="w-5 h-5 text-purple-300" />
            <div>
              <p className="text-white font-medium">Texto Grande</p>
              <p className="text-sm text-gray-400">Aumenta el tamaño del texto</p>
            </div>
          </div>
          <input
            type="checkbox"
            checked={settings.largeText}
            onChange={(e) => handleSettingChange('largeText', e.target.checked)}
            className="w-5 h-5 rounded accent-purple-500"
          />
        </div>

        <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
          <div className="flex items-center gap-3">
            <Volume2 className="w-5 h-5 text-purple-300" />
            <div>
              <p className="text-white font-medium">Lector de Pantalla</p>
              <p className="text-sm text-gray-400">Optimizado para lectores de pantalla</p>
            </div>
          </div>
          <input
            type="checkbox"
            checked={settings.screenReader}
            onChange={(e) => handleSettingChange('screenReader', e.target.checked)}
            className="w-5 h-5 rounded accent-purple-500"
          />
        </div>

        <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
          <div className="flex items-center gap-3">
            <Mouse className="w-5 h-5 text-purple-300" />
            <div>
              <p className="text-white font-medium">Reducir Animaciones</p>
              <p className="text-sm text-gray-400">Reduce las animaciones</p>
            </div>
          </div>
          <input
            type="checkbox"
            checked={settings.reducedMotion}
            onChange={(e) => handleSettingChange('reducedMotion', e.target.checked)}
            className="w-5 h-5 rounded accent-purple-500"
          />
        </div>
      </div>
    </div>
  );
}


