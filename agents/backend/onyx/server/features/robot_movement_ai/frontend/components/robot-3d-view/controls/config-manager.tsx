/**
 * Configuration Manager Component
 * @module robot-3d-view/controls/config-manager
 */

'use client';

import { memo, useRef } from 'react';
import { exportConfig, importConfig, copyConfigToClipboard, getConfigFromClipboard } from '../utils/config-export';
import { notify } from '../utils/notifications';
import type { SceneConfig } from '../types';

/**
 * Props for ConfigManager component
 */
interface ConfigManagerProps {
  config: SceneConfig;
  onConfigChange: (config: SceneConfig) => void;
}

/**
 * Configuration Manager Component
 * 
 * Provides UI for exporting and importing configurations.
 * 
 * @param props - Component props
 * @returns Configuration manager component
 */
export const ConfigManager = memo(({ config, onConfigChange }: ConfigManagerProps) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleExport = () => {
    try {
      exportConfig(config);
      notify.success('Configuración exportada correctamente');
    } catch (error) {
      notify.error('Error al exportar configuración');
    }
  };

  const handleImport = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const importedConfig = await importConfig(file);
      onConfigChange(importedConfig);
      notify.success('Configuración importada correctamente');
    } catch (error) {
      notify.error('Error al importar configuración');
    }

    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleCopy = async () => {
    try {
      await copyConfigToClipboard(config);
      notify.success('Configuración copiada al portapapeles');
    } catch (error) {
      notify.error('Error al copiar configuración');
    }
  };

  const handlePaste = async () => {
    try {
      const pastedConfig = await getConfigFromClipboard();
      onConfigChange(pastedConfig);
      notify.success('Configuración pegada desde portapapeles');
    } catch (error) {
      notify.error('Error al pegar configuración');
    }
  };

  return (
    <div className="absolute bottom-4 left-4 z-40">
      <div className="bg-gray-800/95 backdrop-blur-md border border-gray-700/50 rounded-lg p-2 shadow-lg">
        <div className="text-[10px] text-gray-400 mb-2 px-2">Config:</div>
        <div className="flex gap-1 flex-wrap">
          <button
            onClick={handleExport}
            className="px-2 py-1 text-[10px] rounded bg-gray-700/50 hover:bg-gray-600 transition-all"
            title="Exportar configuración"
          >
            📥 Exportar
          </button>
          <button
            onClick={() => fileInputRef.current?.click()}
            className="px-2 py-1 text-[10px] rounded bg-gray-700/50 hover:bg-gray-600 transition-all"
            title="Importar configuración"
          >
            📤 Importar
          </button>
          <button
            onClick={handleCopy}
            className="px-2 py-1 text-[10px] rounded bg-gray-700/50 hover:bg-gray-600 transition-all"
            title="Copiar configuración"
          >
            📋 Copiar
          </button>
          <button
            onClick={handlePaste}
            className="px-2 py-1 text-[10px] rounded bg-gray-700/50 hover:bg-gray-600 transition-all"
            title="Pegar configuración"
          >
            📄 Pegar
          </button>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept=".json"
          onChange={handleImport}
          className="hidden"
          aria-label="Importar archivo de configuración"
        />
      </div>
    </div>
  );
});

ConfigManager.displayName = 'ConfigManager';



