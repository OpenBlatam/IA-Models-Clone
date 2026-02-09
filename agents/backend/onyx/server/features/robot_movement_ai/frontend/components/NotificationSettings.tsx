'use client';

import { useState } from 'react';
import { Bell, Save, ToggleLeft, ToggleRight } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface NotificationSetting {
  id: string;
  name: string;
  description: string;
  email: boolean;
  push: boolean;
  inApp: boolean;
}

export default function NotificationSettings() {
  const [settings, setSettings] = useState<NotificationSetting[]>([
    {
      id: '1',
      name: 'Movimientos del Robot',
      description: 'Notificaciones cuando el robot se mueve',
      email: true,
      push: true,
      inApp: true,
    },
    {
      id: '2',
      name: 'Alertas de Seguridad',
      description: 'Alertas críticas de seguridad',
      email: true,
      push: true,
      inApp: true,
    },
    {
      id: '3',
      name: 'Errores del Sistema',
      description: 'Notificaciones de errores',
      email: true,
      push: false,
      inApp: true,
    },
    {
      id: '4',
      name: 'Actualizaciones',
      description: 'Notificaciones de actualizaciones disponibles',
      email: false,
      push: false,
      inApp: true,
    },
  ]);

  const toggleSetting = (id: string, type: 'email' | 'push' | 'inApp') => {
    setSettings((prev) =>
      prev.map((s) =>
        s.id === id ? { ...s, [type]: !s[type] } : s
      )
    );
  };

  const handleSave = () => {
    localStorage.setItem('notification_settings', JSON.stringify(settings));
    toast.success('Configuración guardada');
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Bell className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Configuración de Notificaciones</h3>
          </div>
          <button
            onClick={handleSave}
            className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <Save className="w-4 h-4" />
            Guardar
          </button>
        </div>

        {/* Settings List */}
        <div className="space-y-4">
          {settings.map((setting) => (
            <div
              key={setting.id}
              className="p-4 bg-gray-700/50 rounded-lg border border-gray-600"
            >
              <div className="mb-3">
                <h4 className="font-semibold text-white mb-1">{setting.name}</h4>
                <p className="text-sm text-gray-300">{setting.description}</p>
              </div>
              <div className="grid grid-cols-3 gap-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">Email</span>
                  <button
                    onClick={() => toggleSetting(setting.id, 'email')}
                    className="text-primary-400"
                  >
                    {setting.email ? (
                      <ToggleRight className="w-6 h-6" />
                    ) : (
                      <ToggleLeft className="w-6 h-6 text-gray-500" />
                    )}
                  </button>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">Push</span>
                  <button
                    onClick={() => toggleSetting(setting.id, 'push')}
                    className="text-primary-400"
                  >
                    {setting.push ? (
                      <ToggleRight className="w-6 h-6" />
                    ) : (
                      <ToggleLeft className="w-6 h-6 text-gray-500" />
                    )}
                  </button>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">En App</span>
                  <button
                    onClick={() => toggleSetting(setting.id, 'inApp')}
                    className="text-primary-400"
                  >
                    {setting.inApp ? (
                      <ToggleRight className="w-6 h-6" />
                    ) : (
                      <ToggleLeft className="w-6 h-6 text-gray-500" />
                    )}
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}


