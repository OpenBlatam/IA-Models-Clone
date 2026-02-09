'use client';

import { useState } from 'react';
import { Settings, Save, RotateCcw } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface Setting {
  category: string;
  items: { key: string; label: string; value: string | number | boolean; type: string }[];
}

export default function SettingsAdvanced() {
  const [settings, setSettings] = useState<Setting[]>([
    {
      category: 'Rendimiento',
      items: [
        { key: 'cache_enabled', label: 'Habilitar Cache', value: true, type: 'boolean' },
        { key: 'cache_size', label: 'Tamaño de Cache (MB)', value: 100, type: 'number' },
        { key: 'polling_interval', label: 'Intervalo de Polling (ms)', value: 1000, type: 'number' },
      ],
    },
    {
      category: 'Seguridad',
      items: [
        { key: 'ssl_enabled', label: 'Habilitar SSL', value: true, type: 'boolean' },
        { key: 'session_timeout', label: 'Timeout de Sesión (min)', value: 30, type: 'number' },
        { key: 'max_login_attempts', label: 'Intentos de Login Máximos', value: 5, type: 'number' },
      ],
    },
    {
      category: 'Notificaciones',
      items: [
        { key: 'email_notifications', label: 'Notificaciones por Email', value: true, type: 'boolean' },
        { key: 'push_notifications', label: 'Notificaciones Push', value: false, type: 'boolean' },
        { key: 'notification_sound', label: 'Sonido de Notificaciones', value: true, type: 'boolean' },
      ],
    },
  ]);

  const handleSave = () => {
    toast.success('Configuración guardada');
  };

  const handleReset = () => {
    toast.info('Configuración restaurada a valores por defecto');
  };

  const updateSetting = (categoryIndex: number, itemIndex: number, value: any) => {
    setSettings((prev) => {
      const newSettings = [...prev];
      newSettings[categoryIndex].items[itemIndex].value = value;
      return newSettings;
    });
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Settings className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Configuración Avanzada</h3>
          </div>
          <div className="flex gap-2">
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

        {/* Settings by Category */}
        <div className="space-y-6">
          {settings.map((category, catIndex) => (
            <div key={catIndex} className="p-4 bg-gray-700/50 rounded-lg border border-gray-600">
              <h4 className="text-sm font-semibold text-white mb-4">{category.category}</h4>
              <div className="space-y-3">
                {category.items.map((item, itemIndex) => (
                  <div key={item.key} className="flex items-center justify-between">
                    <label className="text-sm text-gray-300">{item.label}</label>
                    {item.type === 'boolean' ? (
                      <select
                        value={String(item.value)}
                        onChange={(e) =>
                          updateSetting(catIndex, itemIndex, e.target.value === 'true')
                        }
                        className="px-3 py-1 bg-gray-800 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                      >
                        <option value="true">Habilitado</option>
                        <option value="false">Deshabilitado</option>
                      </select>
                    ) : (
                      <input
                        type={item.type}
                        value={item.value}
                        onChange={(e) =>
                          updateSetting(
                            catIndex,
                            itemIndex,
                            item.type === 'number' ? Number(e.target.value) : e.target.value
                          )
                        }
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


