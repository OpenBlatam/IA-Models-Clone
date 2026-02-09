'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api/client';
import { useThemeStore } from '@/lib/store/themeStore';
import { Settings, Moon, Sun, Save, RefreshCw, Download, Upload } from 'lucide-react';
import { toast } from '@/lib/utils/toast';
import { exportConfig, importConfig, getDefaultConfig } from '@/lib/utils/configExport';
import { Switch } from '@/components/ui/Switch';
import { useAsync } from '@/lib/hooks/useAsync';
import { useLocalStorageState } from '@/lib/hooks/useLocalStorageState';
import { logger } from '@/lib/utils/logger';

export default function SettingsPanel() {
  const { theme, toggleTheme, setTheme } = useThemeStore();
  const { execute: loadConfig, data: config, loading: isLoading } = useAsync(
    () => apiClient.getConfig()
  );
  const [pollingInterval, setPollingInterval] = useLocalStorageState('pollingInterval', 2000);
  const [autoReconnect, setAutoReconnect] = useLocalStorageState('autoReconnect', true);

  const handleExportConfig = () => {
    const appConfig = {
      theme,
      pollingInterval,
      autoReconnect,
      apiUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8010',
      preferences: {},
    };
    exportConfig(appConfig);
    toast.success('Configuración exportada');
  };

  const handleImportConfig = async () => {
    const importedConfig = await importConfig();
    if (importedConfig) {
      setTheme(importedConfig.theme);
      setPollingInterval(importedConfig.pollingInterval);
      setAutoReconnect(importedConfig.autoReconnect);
      toast.success('Configuración importada');
    } else {
      toast.error('Error al importar configuración');
    }
  };

  useEffect(() => {
    loadConfig();
  }, [loadConfig]);

  const handleSaveConfig = async (key: string, value: any) => {
    try {
      await apiClient.client.post(`/api/v1/system/config/${key}`, { value });
      toast.success('Configuración guardada');
      loadConfig();
    } catch (error: any) {
      toast.error(`Error: ${error.message || 'Failed to save config'}`);
    }
  };

  return (
    <div className="space-y-tesla-lg">
      {/* Theme Settings */}
      <div className="bg-white rounded-lg p-tesla-lg border border-gray-200 shadow-sm">
        <div className="flex items-center gap-tesla-sm mb-tesla-lg">
          <Settings className="w-5 h-5 text-tesla-blue" />
          <h3 className="text-lg font-semibold text-tesla-black">Configuración de Tema</h3>
        </div>
        <div className="flex items-center justify-between">
          <div>
            <p className="text-tesla-black font-semibold mb-tesla-xs">Modo Oscuro/Claro</p>
            <p className="text-sm text-tesla-gray-dark">Cambia entre tema oscuro y claro</p>
          </div>
          <button
            onClick={toggleTheme}
            className="flex items-center gap-tesla-sm px-tesla-lg py-tesla-sm bg-white border-2 border-gray-300 hover:border-gray-400 text-tesla-black rounded-md transition-all font-medium min-h-[44px]"
            aria-label={`Cambiar a tema ${theme === 'dark' ? 'claro' : 'oscuro'}`}
          >
            {theme === 'dark' ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
            {theme === 'dark' ? 'Oscuro' : 'Claro'}
          </button>
        </div>
      </div>

      {/* Frontend Settings */}
      <div className="bg-white rounded-lg p-tesla-lg border border-gray-200 shadow-sm">
        <h3 className="text-lg font-semibold text-tesla-black mb-tesla-lg">Configuración del Frontend</h3>
        <div className="space-y-tesla-lg">
          <div>
            <label className="block text-sm font-medium text-tesla-black mb-tesla-sm">
              Intervalo de Actualización (ms)
            </label>
            <input
              type="number"
              min="500"
              max="10000"
              step="500"
              value={pollingInterval}
              onChange={(e) => setPollingInterval(parseInt(e.target.value))}
              className="w-full px-tesla-md py-tesla-sm bg-white border border-gray-300 rounded-md text-tesla-black focus:outline-none focus:ring-2 focus:ring-tesla-blue focus:border-transparent transition-all"
              aria-label="Intervalo de actualización en milisegundos"
            />
            <p className="text-xs text-tesla-gray-dark mt-tesla-sm">
              Frecuencia con la que se actualiza el estado del robot
            </p>
          </div>
          <div className="flex items-center justify-between p-tesla-md bg-gray-50 rounded-md border border-gray-200">
            <div>
              <p className="text-tesla-black font-semibold mb-tesla-xs">Reconexión Automática</p>
              <p className="text-sm text-tesla-gray-dark">Reconectar automáticamente si se pierde la conexión</p>
            </div>
            <Switch
              checked={autoReconnect}
              onCheckedChange={setAutoReconnect}
              aria-label="Activar reconexión automática"
            />
          </div>
        </div>
      </div>

      {/* Backend Config */}
      {config && (
        <div className="bg-white rounded-lg p-tesla-lg border border-gray-200 shadow-sm">
          <div className="flex items-center justify-between mb-tesla-lg">
            <h3 className="text-lg font-semibold text-tesla-black">Configuración del Backend</h3>
            <button
              onClick={loadConfig}
              disabled={isLoading}
              className="flex items-center gap-tesla-sm px-tesla-md py-tesla-sm bg-white border-2 border-gray-300 hover:border-gray-400 text-tesla-black rounded-md transition-all disabled:opacity-50 font-medium min-h-[44px]"
              aria-label="Actualizar configuración"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
              Actualizar
            </button>
          </div>
          <div className="space-y-tesla-sm max-h-96 overflow-y-auto">
            {Object.entries(config).map(([key, value]: [string, any]) => (
              <div key={key} className="p-tesla-md bg-gray-50 rounded-md border border-gray-200">
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <p className="text-sm font-semibold text-tesla-black mb-tesla-xs">{key}</p>
                    <p className="text-xs text-tesla-gray-dark font-mono">
                      {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Keyboard Shortcuts */}
      <div className="bg-white rounded-lg p-tesla-lg border border-gray-200 shadow-sm">
        <h3 className="text-lg font-semibold text-tesla-black mb-tesla-lg">Atajos de Teclado</h3>
        <div className="space-y-tesla-sm">
          <div className="flex items-center justify-between p-tesla-md bg-gray-50 rounded-md border border-gray-200">
            <kbd className="px-tesla-sm py-tesla-sm bg-white border border-gray-300 rounded text-sm text-tesla-black font-medium">Ctrl + H</kbd>
            <span className="text-tesla-black font-medium">Ir a posición home</span>
          </div>
          <div className="flex items-center justify-between p-tesla-md bg-gray-50 rounded-md border border-gray-200">
            <kbd className="px-tesla-sm py-tesla-sm bg-white border border-gray-300 rounded text-sm text-tesla-black font-medium">Ctrl + S</kbd>
            <span className="text-tesla-black font-medium">Detener robot</span>
          </div>
          <div className="flex items-center justify-between p-tesla-md bg-gray-50 rounded-md border border-gray-200">
            <kbd className="px-tesla-sm py-tesla-sm bg-white border border-gray-300 rounded text-sm text-tesla-black font-medium">Ctrl + R</kbd>
            <span className="text-tesla-black font-medium">Iniciar/Detener grabación</span>
          </div>
        </div>
      </div>

      {/* Import/Export */}
      <div className="bg-white rounded-lg p-tesla-lg border border-gray-200 shadow-sm">
        <h3 className="text-lg font-semibold text-tesla-black mb-tesla-lg">Importar/Exportar Configuración</h3>
        <div className="flex gap-tesla-sm">
          <button
            onClick={handleExportConfig}
            className="flex-1 flex items-center justify-center gap-tesla-sm px-tesla-lg py-tesla-sm bg-tesla-blue hover:bg-opacity-90 text-white rounded-md transition-all font-medium min-h-[44px]"
            aria-label="Exportar configuración"
          >
            <Download className="w-4 h-4" />
            Exportar Configuración
          </button>
          <button
            onClick={handleImportConfig}
            className="flex-1 flex items-center justify-center gap-tesla-sm px-tesla-lg py-tesla-sm bg-white border-2 border-gray-300 hover:border-gray-400 text-tesla-black rounded-md transition-all font-medium min-h-[44px]"
            aria-label="Importar configuración"
          >
            <Upload className="w-4 h-4" />
            Importar Configuración
          </button>
        </div>
      </div>
    </div>
  );
}

