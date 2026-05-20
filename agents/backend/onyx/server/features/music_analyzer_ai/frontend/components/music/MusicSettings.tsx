'use client';

import { useState } from 'react';
import { Settings, Volume2, Music, Bell, Moon, Sun } from 'lucide-react';

export function MusicSettings() {
  const [settings, setSettings] = useState({
    autoPlay: false,
    crossfade: true,
    crossfadeDuration: 3,
    notifications: true,
    theme: 'dark',
    volume: 0.8,
    quality: 'high',
  });

  const handleSettingChange = (key: string, value: any) => {
    setSettings({ ...settings, [key]: value });
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Settings className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Configuración</h3>
      </div>

      <div className="space-y-6">
        <div>
          <label className="flex items-center justify-between mb-2">
            <span className="text-white">Reproducción Automática</span>
            <input
              type="checkbox"
              checked={settings.autoPlay}
              onChange={(e) => handleSettingChange('autoPlay', e.target.checked)}
              className="w-4 h-4 rounded accent-purple-500"
            />
          </label>
        </div>

        <div>
          <label className="flex items-center justify-between mb-2">
            <span className="text-white">Crossfade</span>
            <input
              type="checkbox"
              checked={settings.crossfade}
              onChange={(e) => handleSettingChange('crossfade', e.target.checked)}
              className="w-4 h-4 rounded accent-purple-500"
            />
          </label>
          {settings.crossfade && (
            <div className="mt-2">
              <input
                type="range"
                min="0"
                max="12"
                value={settings.crossfadeDuration}
                onChange={(e) => handleSettingChange('crossfadeDuration', parseInt(e.target.value))}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
              />
              <div className="flex justify-between text-xs text-gray-400 mt-1">
                <span>0s</span>
                <span>{settings.crossfadeDuration}s</span>
                <span>12s</span>
              </div>
            </div>
          )}
        </div>

        <div>
          <label className="flex items-center justify-between mb-2">
            <span className="text-white">Notificaciones</span>
            <input
              type="checkbox"
              checked={settings.notifications}
              onChange={(e) => handleSettingChange('notifications', e.target.checked)}
              className="w-4 h-4 rounded accent-purple-500"
            />
          </label>
        </div>

        <div>
          <label className="block text-white mb-2">Tema</label>
          <div className="flex gap-2">
            <button
              onClick={() => handleSettingChange('theme', 'dark')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                settings.theme === 'dark'
                  ? 'bg-purple-600 text-white'
                  : 'bg-white/10 text-gray-300 hover:bg-white/20'
              }`}
            >
              <Moon className="w-4 h-4" />
              Oscuro
            </button>
            <button
              onClick={() => handleSettingChange('theme', 'light')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                settings.theme === 'light'
                  ? 'bg-purple-600 text-white'
                  : 'bg-white/10 text-gray-300 hover:bg-white/20'
              }`}
            >
              <Sun className="w-4 h-4" />
              Claro
            </button>
          </div>
        </div>

        <div>
          <label className="block text-white mb-2">Volumen por Defecto</label>
          <div className="flex items-center gap-2">
            <Volume2 className="w-4 h-4 text-gray-400" />
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={settings.volume}
              onChange={(e) => handleSettingChange('volume', parseFloat(e.target.value))}
              className="flex-1 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
            />
            <span className="text-sm text-white w-12 text-right">
              {Math.round(settings.volume * 100)}%
            </span>
          </div>
        </div>

        <div>
          <label className="block text-white mb-2">Calidad de Audio</label>
          <select
            value={settings.quality}
            onChange={(e) => handleSettingChange('quality', e.target.value)}
            className="w-full px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-400"
          >
            <option value="low">Baja</option>
            <option value="medium">Media</option>
            <option value="high">Alta</option>
          </select>
        </div>
      </div>
    </div>
  );
}


