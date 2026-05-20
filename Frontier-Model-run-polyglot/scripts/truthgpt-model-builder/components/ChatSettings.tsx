/**
 * ChatSettings - Panel de Configuración
 * =======================================
 * 
 * Componente para configurar opciones del chat
 */

import React, { useState } from 'react';
import { 
  Settings, 
  X, 
  Save,
  Palette,
  MessageSquare,
  Zap,
  Shield,
  Bell,
  Globe
} from 'lucide-react';

interface ChatSettingsProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (settings: ChatSettingsData) => void;
  initialSettings?: ChatSettingsData;
  isDarkMode?: boolean;
}

export interface ChatSettingsData {
  theme: 'light' | 'dark' | 'auto';
  language: string;
  maxMessages: number;
  enableMarkdown: boolean;
  enableSound: boolean;
  enableNotifications: boolean;
  autoSave: boolean;
  fontSize: 'small' | 'medium' | 'large';
  compactMode: boolean;
  showTimestamps: boolean;
  showAvatars: boolean;
}

export const ChatSettings: React.FC<ChatSettingsProps> = ({
  isOpen,
  onClose,
  onSave,
  initialSettings,
  isDarkMode = false,
}) => {
  const [settings, setSettings] = useState<ChatSettingsData>(
    initialSettings || {
      theme: 'auto',
      language: 'es',
      maxMessages: 1000,
      enableMarkdown: true,
      enableSound: false,
      enableNotifications: true,
      autoSave: true,
      fontSize: 'medium',
      compactMode: false,
      showTimestamps: true,
      showAvatars: false,
    }
  );

  const handleSave = () => {
    onSave(settings);
    onClose();
  };

  const updateSetting = <K extends keyof ChatSettingsData>(
    key: K,
    value: ChatSettingsData[K]
  ) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Overlay */}
      <div
        className="absolute inset-0 bg-black/50"
        onClick={onClose}
      />

      {/* Modal */}
      <div
        className={`relative w-full max-w-2xl max-h-[90vh] rounded-lg shadow-xl ${
          isDarkMode ? 'bg-gray-900 text-gray-100' : 'bg-white text-gray-900'
        } overflow-hidden flex flex-col`}
      >
        {/* Header */}
        <div
          className={`flex items-center justify-between p-4 border-b ${
            isDarkMode ? 'border-gray-800' : 'border-gray-200'
          }`}
        >
          <div className="flex items-center gap-2">
            <Settings className="w-5 h-5" />
            <h2 className="text-xl font-semibold">Configuración</h2>
          </div>
          <button
            onClick={onClose}
            className={`p-2 rounded-lg transition-colors ${
              isDarkMode
                ? 'hover:bg-gray-800 text-gray-400'
                : 'hover:bg-gray-100 text-gray-600'
            }`}
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Apariencia */}
          <section>
            <div className="flex items-center gap-2 mb-4">
              <Palette className="w-5 h-5" />
              <h3 className="text-lg font-semibold">Apariencia</h3>
            </div>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Tema</label>
                <select
                  value={settings.theme}
                  onChange={(e) =>
                    updateSetting('theme', e.target.value as 'light' | 'dark' | 'auto')
                  }
                  className={`w-full p-2 rounded-lg ${
                    isDarkMode
                      ? 'bg-gray-800 text-gray-100 border-gray-700'
                      : 'bg-gray-50 text-gray-900 border-gray-300'
                  } border focus:outline-none focus:ring-2 ${
                    isDarkMode ? 'focus:ring-blue-500' : 'focus:ring-blue-400'
                  }`}
                >
                  <option value="light">Claro</option>
                  <option value="dark">Oscuro</option>
                  <option value="auto">Automático</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Tamaño de fuente</label>
                <select
                  value={settings.fontSize}
                  onChange={(e) =>
                    updateSetting('fontSize', e.target.value as 'small' | 'medium' | 'large')
                  }
                  className={`w-full p-2 rounded-lg ${
                    isDarkMode
                      ? 'bg-gray-800 text-gray-100 border-gray-700'
                      : 'bg-gray-50 text-gray-900 border-gray-300'
                  } border focus:outline-none`}
                >
                  <option value="small">Pequeño</option>
                  <option value="medium">Mediano</option>
                  <option value="large">Grande</option>
                </select>
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <label className="block text-sm font-medium">Modo compacto</label>
                  <p className="text-xs text-gray-500">Menos espacio entre mensajes</p>
                </div>
                <button
                  onClick={() => updateSetting('compactMode', !settings.compactMode)}
                  className={`relative w-12 h-6 rounded-full transition-colors ${
                    settings.compactMode
                      ? isDarkMode ? 'bg-blue-600' : 'bg-blue-500'
                      : isDarkMode ? 'bg-gray-700' : 'bg-gray-300'
                  }`}
                >
                  <div
                    className={`absolute w-5 h-5 bg-white rounded-full transition-transform ${
                      settings.compactMode ? 'translate-x-6' : 'translate-x-0.5'
                    } top-0.5`}
                  />
                </button>
              </div>
            </div>
          </section>

          {/* Mensajes */}
          <section>
            <div className="flex items-center gap-2 mb-4">
              <MessageSquare className="w-5 h-5" />
              <h3 className="text-lg font-semibold">Mensajes</h3>
            </div>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <label className="block text-sm font-medium">Soporte Markdown</label>
                  <p className="text-xs text-gray-500">Renderizar formato Markdown</p>
                </div>
                <button
                  onClick={() => updateSetting('enableMarkdown', !settings.enableMarkdown)}
                  className={`relative w-12 h-6 rounded-full transition-colors ${
                    settings.enableMarkdown
                      ? isDarkMode ? 'bg-blue-600' : 'bg-blue-500'
                      : isDarkMode ? 'bg-gray-700' : 'bg-gray-300'
                  }`}
                >
                  <div
                    className={`absolute w-5 h-5 bg-white rounded-full transition-transform ${
                      settings.enableMarkdown ? 'translate-x-6' : 'translate-x-0.5'
                    } top-0.5`}
                  />
                </button>
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <label className="block text-sm font-medium">Mostrar timestamps</label>
                  <p className="text-xs text-gray-500">Mostrar hora en cada mensaje</p>
                </div>
                <button
                  onClick={() => updateSetting('showTimestamps', !settings.showTimestamps)}
                  className={`relative w-12 h-6 rounded-full transition-colors ${
                    settings.showTimestamps
                      ? isDarkMode ? 'bg-blue-600' : 'bg-blue-500'
                      : isDarkMode ? 'bg-gray-700' : 'bg-gray-300'
                  }`}
                >
                  <div
                    className={`absolute w-5 h-5 bg-white rounded-full transition-transform ${
                      settings.showTimestamps ? 'translate-x-6' : 'translate-x-0.5'
                    } top-0.5`}
                  />
                </button>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">
                  Límite de mensajes: {settings.maxMessages}
                </label>
                <input
                  type="range"
                  min="100"
                  max="10000"
                  step="100"
                  value={settings.maxMessages}
                  onChange={(e) => updateSetting('maxMessages', parseInt(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>
          </section>

          {/* Notificaciones */}
          <section>
            <div className="flex items-center gap-2 mb-4">
              <Bell className="w-5 h-5" />
              <h3 className="text-lg font-semibold">Notificaciones</h3>
            </div>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <label className="block text-sm font-medium">Notificaciones</label>
                  <p className="text-xs text-gray-500">Recibir notificaciones de nuevos mensajes</p>
                </div>
                <button
                  onClick={() => updateSetting('enableNotifications', !settings.enableNotifications)}
                  className={`relative w-12 h-6 rounded-full transition-colors ${
                    settings.enableNotifications
                      ? isDarkMode ? 'bg-blue-600' : 'bg-blue-500'
                      : isDarkMode ? 'bg-gray-700' : 'bg-gray-300'
                  }`}
                >
                  <div
                    className={`absolute w-5 h-5 bg-white rounded-full transition-transform ${
                      settings.enableNotifications ? 'translate-x-6' : 'translate-x-0.5'
                    } top-0.5`}
                  />
                </button>
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <label className="block text-sm font-medium">Sonidos</label>
                  <p className="text-xs text-gray-500">Reproducir sonidos al recibir mensajes</p>
                </div>
                <button
                  onClick={() => updateSetting('enableSound', !settings.enableSound)}
                  className={`relative w-12 h-6 rounded-full transition-colors ${
                    settings.enableSound
                      ? isDarkMode ? 'bg-blue-600' : 'bg-blue-500'
                      : isDarkMode ? 'bg-gray-700' : 'bg-gray-300'
                  }`}
                >
                  <div
                    className={`absolute w-5 h-5 bg-white rounded-full transition-transform ${
                      settings.enableSound ? 'translate-x-6' : 'translate-x-0.5'
                    } top-0.5`}
                  />
                </button>
              </div>
            </div>
          </section>

          {/* General */}
          <section>
            <div className="flex items-center gap-2 mb-4">
              <Zap className="w-5 h-5" />
              <h3 className="text-lg font-semibold">General</h3>
            </div>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Idioma</label>
                <select
                  value={settings.language}
                  onChange={(e) => updateSetting('language', e.target.value)}
                  className={`w-full p-2 rounded-lg ${
                    isDarkMode
                      ? 'bg-gray-800 text-gray-100 border-gray-700'
                      : 'bg-gray-50 text-gray-900 border-gray-300'
                  } border focus:outline-none`}
                >
                  <option value="es">Español</option>
                  <option value="en">English</option>
                  <option value="fr">Français</option>
                  <option value="de">Deutsch</option>
                </select>
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <label className="block text-sm font-medium">Guardado automático</label>
                  <p className="text-xs text-gray-500">Guardar conversaciones automáticamente</p>
                </div>
                <button
                  onClick={() => updateSetting('autoSave', !settings.autoSave)}
                  className={`relative w-12 h-6 rounded-full transition-colors ${
                    settings.autoSave
                      ? isDarkMode ? 'bg-blue-600' : 'bg-blue-500'
                      : isDarkMode ? 'bg-gray-700' : 'bg-gray-300'
                  }`}
                >
                  <div
                    className={`absolute w-5 h-5 bg-white rounded-full transition-transform ${
                      settings.autoSave ? 'translate-x-6' : 'translate-x-0.5'
                    } top-0.5`}
                  />
                </button>
              </div>
            </div>
          </section>
        </div>

        {/* Footer */}
        <div
          className={`flex items-center justify-end gap-3 p-4 border-t ${
            isDarkMode ? 'border-gray-800' : 'border-gray-200'
          }`}
        >
          <button
            onClick={onClose}
            className={`px-4 py-2 rounded-lg transition-colors ${
              isDarkMode
                ? 'hover:bg-gray-800 text-gray-300'
                : 'hover:bg-gray-100 text-gray-600'
            }`}
          >
            Cancelar
          </button>
          <button
            onClick={handleSave}
            className={`px-4 py-2 rounded-lg transition-colors flex items-center gap-2 ${
              isDarkMode
                ? 'bg-blue-600 hover:bg-blue-700 text-white'
                : 'bg-blue-500 hover:bg-blue-600 text-white'
            }`}
          >
            <Save className="w-4 h-4" />
            Guardar
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatSettings;


