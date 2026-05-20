'use client';

import { useState } from 'react';
import { Palette, Moon, Sun, Sparkles } from 'lucide-react';

export function MusicThemes() {
  const [selectedTheme, setSelectedTheme] = useState('dark');

  const themes = [
    { id: 'dark', name: 'Oscuro', icon: Moon, gradient: 'from-gray-900 to-gray-800' },
    { id: 'light', name: 'Claro', icon: Sun, gradient: 'from-blue-50 to-white' },
    { id: 'purple', name: 'Púrpura', icon: Sparkles, gradient: 'from-purple-900 to-pink-900' },
    { id: 'blue', name: 'Azul', icon: Sparkles, gradient: 'from-blue-900 to-cyan-900' },
    { id: 'green', name: 'Verde', icon: Sparkles, gradient: 'from-green-900 to-emerald-900' },
  ];

  const handleThemeChange = (themeId: string) => {
    setSelectedTheme(themeId);
    // Aplicar tema (en producción, esto cambiaría las clases CSS globales)
    document.documentElement.setAttribute('data-theme', themeId);
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Palette className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Temas</h3>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        {themes.map((theme) => {
          const Icon = theme.icon;
          return (
            <button
              key={theme.id}
              onClick={() => handleThemeChange(theme.id)}
              className={`p-4 rounded-lg border-2 transition-all ${
                selectedTheme === theme.id
                  ? 'border-purple-500 scale-105'
                  : 'border-white/20 hover:border-white/40'
              }`}
            >
              <div className={`w-full h-16 rounded mb-2 bg-gradient-to-br ${theme.gradient}`} />
              <div className="flex items-center gap-2 justify-center">
                <Icon className="w-4 h-4 text-white" />
                <span className="text-sm text-white font-medium">{theme.name}</span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}


