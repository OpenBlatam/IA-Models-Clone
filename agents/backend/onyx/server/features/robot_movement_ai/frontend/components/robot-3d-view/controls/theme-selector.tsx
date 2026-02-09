/**
 * Theme Selector Component
 * @module robot-3d-view/controls/theme-selector
 */

'use client';

import { memo } from 'react';
import { useTheme, type ThemeType } from '../hooks/use-theme';
import { notify } from '../utils/notifications';

/**
 * Theme Selector Component
 * 
 * Allows users to switch between different visual themes.
 * 
 * @returns Theme selector component
 */
export const ThemeSelector = memo(() => {
  const { theme, setTheme } = useTheme();

  const themes: Array<{ type: ThemeType; label: string; emoji: string }> = [
    { type: 'dark', label: 'Oscuro', emoji: '🌙' },
    { type: 'light', label: 'Claro', emoji: '☀️' },
    { type: 'industrial', label: 'Industrial', emoji: '🏭' },
    { type: 'futuristic', label: 'Futurista', emoji: '🚀' },
    { type: 'minimal', label: 'Minimal', emoji: '⚪' },
  ];

  const handleThemeChange = (newTheme: ThemeType) => {
    setTheme(newTheme);
    notify.success(`Tema "${themes.find((t) => t.type === newTheme)?.label}" aplicado`);
  };

  return (
    <div className="absolute bottom-4 right-4 z-40">
      <div className="bg-gray-800/95 backdrop-blur-md border border-gray-700/50 rounded-lg p-2 shadow-lg">
        <div className="text-[10px] text-gray-400 mb-2 px-2">Tema:</div>
        <div className="flex gap-1 flex-wrap">
          {themes.map(({ type, label, emoji }) => (
            <button
              key={type}
              onClick={() => handleThemeChange(type)}
              className={`
                px-2 py-1 text-[10px] rounded transition-all
                ${theme === type ? 'bg-blue-600' : 'bg-gray-700/50 hover:bg-gray-600'}
              `}
              title={label}
            >
              {emoji} {label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
});

ThemeSelector.displayName = 'ThemeSelector';



