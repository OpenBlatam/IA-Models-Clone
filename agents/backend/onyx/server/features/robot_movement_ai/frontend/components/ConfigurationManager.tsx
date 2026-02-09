'use client';

import { useState } from 'react';
import { Settings, Save, RotateCcw, Download, Upload } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface ConfigSection {
  id: string;
  name: string;
  settings: { key: string; value: string; type: 'text' | 'number' | 'boolean' }[];
}

export default function ConfigurationManager() {
  const [sections] = useState<ConfigSection[]>([
    {
      id: '1',
      name: 'General',
      settings: [
        { key: 'robot_name', value: 'Robot-001', type: 'text' },
        { key: 'polling_interval', value: '1000', type: 'number' },
        { key: 'auto_reconnect', value: 'true', type: 'boolean' },
      ],
    },
    {
      id: '2',
      name: 'Seguridad',
      settings: [
        { key: 'max_speed', value: '1.0', type: 'number' },
        { key: 'enable_safety_limits', value: 'true', type: 'boolean' },
        { key: 'emergency_stop_enabled', value: 'true', type: 'boolean' },
      ],
    },
    {
      id: '3',
      name: 'Red',
      settings: [
        { key: 'api_timeout', value: '5000', type: 'number' },
        { key: 'retry_attempts', value: '3', type: 'number' },
        { key: 'enable_websocket', value: 'true', type: 'boolean' },
      ],
    },
  ]);

  const handleSave = () => {
    toast.success('Configuración guardada');
  };

  const handleReset = () => {
    toast.info('Configuración restaurada a valores por defecto');
  };

  const handleExport = () => {
    toast.success('Configuración exportada');
  };

  const handleImport = () => {
    toast.success('Configuración importada');
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Settings className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Gestor de Configuración</h3>
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleExport}
              className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors flex items-center gap-2 text-sm"
            >
              <Download className="w-4 h-4" />
              Exportar
            </button>
            <button
              onClick={handleImport}
              className="px-3 py-1 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors flex items-center gap-2 text-sm"
            >
              <Upload className="w-4 h-4" />
              Importar
            </button>
            <button
              onClick={handleReset}
              className="px-3 py-1 bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg transition-colors flex items-center gap-2 text-sm"
            >
              <RotateCcw className="w-4 h-4" />
              Restaurar
            </button>
            <button
              onClick={handleSave}
              className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors flex items-center gap-2"
            >
              <Save className="w-4 h-4" />
              Guardar
            </button>
          </div>
        </div>

        {/* Configuration Sections */}
        <div className="space-y-6">
          {sections.map((section) => (
            <div key={section.id} className="p-4 bg-gray-700/50 rounded-lg border border-gray-600">
              <h4 className="text-sm font-semibold text-white mb-4">{section.name}</h4>
              <div className="space-y-3">
                {section.settings.map((setting) => (
                  <div key={setting.key} className="flex items-center justify-between">
                    <label className="text-sm text-gray-300 capitalize">
                      {setting.key.replace(/_/g, ' ')}
                    </label>
                    {setting.type === 'boolean' ? (
                      <select
                        defaultValue={setting.value}
                        className="px-3 py-1 bg-gray-800 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                      >
                        <option value="true">Habilitado</option>
                        <option value="false">Deshabilitado</option>
                      </select>
                    ) : (
                      <input
                        type={setting.type}
                        defaultValue={setting.value}
                        className="px-3 py-1 bg-gray-800 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 w-32"
                      />
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}


