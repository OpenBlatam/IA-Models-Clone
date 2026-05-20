'use client';

import { useAppStore } from '@/store/app-store';
import { apiClient } from '@/lib/api-client';
import { useEffect } from 'react';
import ThemeToggle from './ThemeToggle';
import NotificationCenter from './NotificationCenter';
import SettingsPanel from './SettingsPanel';
import ThemeCustomizer from './ThemeCustomizer';
import AccessibilityTools from './AccessibilityTools';
import HelpCenter from './HelpCenter';
import { useState } from 'react';
import { FiSettings, FiPalette, FiEye, FiHelpCircle } from 'react-icons/fi';

export default function Header() {
  const { isConnected, health, setHealth, setConnected } = useAppStore();
  const [showSettings, setShowSettings] = useState(false);
  const [showThemeCustomizer, setShowThemeCustomizer] = useState(false);
  const [showAccessibility, setShowAccessibility] = useState(false);
  const [showHelp, setShowHelp] = useState(false);

  const handleRefresh = async () => {
    try {
      const healthData = await apiClient.getHealth();
      setHealth(healthData);
      setConnected(true);
    } catch (error) {
      console.error('Refresh failed:', error);
      setConnected(false);
    }
  };

  return (
    <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 sticky top-0 z-50 transition-colors">
      <div className="px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">🚀 BUL</h1>
          <span className="text-sm text-gray-500 dark:text-gray-400">Generador de Documentos con IA</span>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${
                isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'
              }`}
            />
            <span className="text-sm text-gray-600 dark:text-gray-400">
              {isConnected ? 'Conectado' : 'Desconectado'}
            </span>
          </div>
          
          <button
            onClick={handleRefresh}
            className="btn-icon"
            title="Actualizar estado"
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
              <path d="M21 3v5h-5" />
              <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
              <path d="M3 21v-5h5" />
            </svg>
          </button>
          <NotificationCenter />
          <button
            onClick={() => setShowHelp(true)}
            className="btn-icon"
            title="Centro de Ayuda"
          >
            <FiHelpCircle size={20} />
          </button>
          <button
            onClick={() => setShowAccessibility(true)}
            className="btn-icon"
            title="Herramientas de Accesibilidad"
          >
            <FiEye size={20} />
          </button>
          <button
            onClick={() => setShowThemeCustomizer(true)}
            className="btn-icon"
            title="Personalizar Tema"
          >
            <FiPalette size={20} />
          </button>
          <button
            onClick={() => setShowSettings(true)}
            className="btn-icon"
            title="Configuración"
          >
            <FiSettings size={20} />
          </button>
          <ThemeToggle />
        </div>
      </div>
      <SettingsPanel isOpen={showSettings} onClose={() => setShowSettings(false)} />
      <ThemeCustomizer isOpen={showThemeCustomizer} onClose={() => setShowThemeCustomizer(false)} />
      <AccessibilityTools isOpen={showAccessibility} onClose={() => setShowAccessibility(false)} />
      <HelpCenter isOpen={showHelp} onClose={() => setShowHelp(false)} />
    </header>
  );
}

