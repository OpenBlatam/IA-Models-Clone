'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiPalette, FiX } from 'react-icons/fi';

interface Theme {
  name: string;
  primary: string;
  secondary: string;
  background: string;
  text: string;
}

const themes: Theme[] = [
  {
    name: 'Azul',
    primary: '#3B82F6',
    secondary: '#60A5FA',
    background: '#FFFFFF',
    text: '#1F2937',
  },
  {
    name: 'Verde',
    primary: '#10B981',
    secondary: '#34D399',
    background: '#FFFFFF',
    text: '#1F2937',
  },
  {
    name: 'Púrpura',
    primary: '#8B5CF6',
    secondary: '#A78BFA',
    background: '#FFFFFF',
    text: '#1F2937',
  },
  {
    name: 'Rojo',
    primary: '#EF4444',
    secondary: '#F87171',
    background: '#FFFFFF',
    text: '#1F2937',
  },
  {
    name: 'Naranja',
    primary: '#F97316',
    secondary: '#FB923C',
    background: '#FFFFFF',
    text: '#1F2937',
  },
];

interface ThemeCustomizerProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function ThemeCustomizer({ isOpen, onClose }: ThemeCustomizerProps) {
  const [selectedTheme, setSelectedTheme] = useState<Theme>(themes[0]);

  useEffect(() => {
    if (isOpen) {
      const stored = localStorage.getItem('bul_custom_theme');
      if (stored) {
        setSelectedTheme(JSON.parse(stored));
      }
    }
  }, [isOpen]);

  const applyTheme = (theme: Theme) => {
    setSelectedTheme(theme);
    const root = document.documentElement;
    root.style.setProperty('--color-primary', theme.primary);
    root.style.setProperty('--color-secondary', theme.secondary);
    localStorage.setItem('bul_custom_theme', JSON.stringify(theme));
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ x: 300 }}
        animate={{ x: 0 }}
        exit={{ x: 300 }}
        className="fixed right-0 top-0 h-full w-80 bg-white dark:bg-gray-800 border-l border-gray-200 dark:border-gray-700 shadow-xl z-40 flex flex-col"
      >
        <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FiPalette size={20} className="text-primary-600" />
            <h3 className="font-semibold text-gray-900 dark:text-white">Personalizar Tema</h3>
          </div>
          <button onClick={onClose} className="btn-icon">
            <FiX size={20} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {themes.map((theme) => (
            <button
              key={theme.name}
              onClick={() => applyTheme(theme)}
              className={`w-full p-4 rounded-lg border-2 transition-all ${
                selectedTheme.name === theme.name
                  ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                  : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
              }`}
            >
              <div className="flex items-center gap-3">
                <div
                  className="w-12 h-12 rounded-lg"
                  style={{ backgroundColor: theme.primary }}
                />
                <div className="flex-1 text-left">
                  <p className="font-medium text-gray-900 dark:text-white">{theme.name}</p>
                  <div className="flex gap-1 mt-1">
                    <div
                      className="w-4 h-4 rounded-full"
                      style={{ backgroundColor: theme.primary }}
                    />
                    <div
                      className="w-4 h-4 rounded-full"
                      style={{ backgroundColor: theme.secondary }}
                    />
                  </div>
                </div>
              </div>
            </button>
          ))}
        </div>
      </motion.div>
    </AnimatePresence>
  );
}


