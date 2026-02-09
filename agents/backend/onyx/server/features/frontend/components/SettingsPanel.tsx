'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiSettings, FiX, FiSave, FiShield, FiDatabase, FiInfo, FiFlag } from 'react-icons/fi';
import SecuritySettings from './SecuritySettings';
import DataExport from './DataExport';
import SystemInfo from './SystemInfo';
import FeatureFlags from './FeatureFlags';

interface Settings {
  autoSave: boolean;
  notifications: boolean;
  darkMode: boolean;
  language: string;
  itemsPerPage: number;
}

const defaultSettings: Settings = {
  autoSave: true,
  notifications: true,
  darkMode: false,
  language: 'es',
  itemsPerPage: 20,
};

interface SettingsPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function SettingsPanel({ isOpen, onClose }: SettingsPanelProps) {
  const [settings, setSettings] = useState<Settings>(defaultSettings);
  const [hasChanges, setHasChanges] = useState(false);
  const [showSecurity, setShowSecurity] = useState(false);
  const [showDataExport, setShowDataExport] = useState(false);
  const [showSystemInfo, setShowSystemInfo] = useState(false);
  const [showFeatureFlags, setShowFeatureFlags] = useState(false);

  useEffect(() => {
    if (isOpen) {
      const stored = localStorage.getItem('bul_settings');
      if (stored) {
        setSettings({ ...defaultSettings, ...JSON.parse(stored) });
      }
    }
  }, [isOpen]);

  const handleChange = (key: keyof Settings, value: any) => {
    setSettings({ ...settings, [key]: value });
    setHasChanges(true);
  };

  const handleSave = () => {
    localStorage.setItem('bul_settings', JSON.stringify(settings));
    setHasChanges(false);
    
    // Apply settings
    if (settings.darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    
    onClose();
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ x: 300 }}
        animate={{ x: 0 }}
        exit={{ x: 300 }}
        className="fixed right-0 top-0 h-full w-96 bg-white dark:bg-gray-800 border-l border-gray-200 dark:border-gray-700 shadow-xl z-40 flex flex-col"
      >
        <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FiSettings size={20} className="text-primary-600" />
            <h3 className="font-semibold text-gray-900 dark:text-white">Configuración</h3>
          </div>
          <button onClick={onClose} className="btn-icon">
            <FiX size={20} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          <div>
            <label className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Guardado Automático
              </span>
              <input
                type="checkbox"
                checked={settings.autoSave}
                onChange={(e) => handleChange('autoSave', e.target.checked)}
                className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
              />
            </label>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Guardar cambios automáticamente
            </p>
          </div>

          <div>
            <label className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Notificaciones
              </span>
              <input
                type="checkbox"
                checked={settings.notifications}
                onChange={(e) => handleChange('notifications', e.target.checked)}
                className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
              />
            </label>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Recibir notificaciones del sistema
            </p>
          </div>

          <div>
            <label className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Sonidos de Notificación
              </span>
              <input
                type="checkbox"
                checked={localStorage.getItem('bul_notification_sounds') === 'true'}
                onChange={(e) => {
                  localStorage.setItem('bul_notification_sounds', e.target.checked ? 'true' : 'false');
                }}
                className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
              />
            </label>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Reproducir sonidos al recibir notificaciones
            </p>
          </div>

          <div>
            <label className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Modo Oscuro
              </span>
              <input
                type="checkbox"
                checked={settings.darkMode}
                onChange={(e) => handleChange('darkMode', e.target.checked)}
                className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
              />
            </label>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Activar tema oscuro
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Idioma
            </label>
            <select
              value={settings.language}
              onChange={(e) => handleChange('language', e.target.value)}
              className="select"
            >
              <option value="es">Español</option>
              <option value="en">English</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Items por Página
            </label>
            <select
              value={settings.itemsPerPage}
              onChange={(e) => handleChange('itemsPerPage', parseInt(e.target.value))}
              className="select"
            >
              <option value="10">10</option>
              <option value="20">20</option>
              <option value="50">50</option>
              <option value="100">100</option>
            </select>
          </div>

          <div className="pt-4 border-t border-gray-200 dark:border-gray-700 space-y-2">
            <button
              onClick={() => setShowSecurity(true)}
              className="w-full flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
            >
              <div className="flex items-center gap-2">
                <FiShield size={18} className="text-primary-600" />
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Configuración de Seguridad
                </span>
              </div>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M9 18l6-6-6-6" />
              </svg>
            </button>
            <button
              onClick={() => setShowDataExport(true)}
              className="w-full flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
            >
              <div className="flex items-center gap-2">
                <FiDatabase size={18} className="text-primary-600" />
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Exportar Datos
                </span>
              </div>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M9 18l6-6-6-6" />
              </svg>
            </button>
            <button
              onClick={() => setShowSystemInfo(true)}
              className="w-full flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
            >
              <div className="flex items-center gap-2">
                <FiInfo size={18} className="text-primary-600" />
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Información del Sistema
                </span>
              </div>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M9 18l6-6-6-6" />
              </svg>
            </button>
            <button
              onClick={() => setShowFeatureFlags(true)}
              className="w-full flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
            >
              <div className="flex items-center gap-2">
                <FiFlag size={18} className="text-primary-600" />
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Feature Flags
                </span>
              </div>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M9 18l6-6-6-6" />
              </svg>
            </button>
          </div>
        </div>

        <div className="p-4 border-t border-gray-200 dark:border-gray-700">
          <button
            onClick={handleSave}
            className="btn btn-primary w-full"
            disabled={!hasChanges}
          >
            <FiSave size={18} />
            Guardar Cambios
          </button>
        </div>
      </motion.div>
      <SecuritySettings isOpen={showSecurity} onClose={() => setShowSecurity(false)} />
      <DataExport isOpen={showDataExport} onClose={() => setShowDataExport(false)} />
      <SystemInfo isOpen={showSystemInfo} onClose={() => setShowSystemInfo(false)} />
      <FeatureFlags isOpen={showFeatureFlags} onClose={() => setShowFeatureFlags(false)} />
    </AnimatePresence>
  );
}

