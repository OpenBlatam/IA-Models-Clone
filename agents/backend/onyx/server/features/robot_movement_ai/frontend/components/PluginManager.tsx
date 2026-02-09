'use client';

import { useState } from 'react';
import { Puzzle, Plus, Trash2, Power, Settings } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface Plugin {
  id: string;
  name: string;
  description: string;
  version: string;
  enabled: boolean;
  author: string;
}

export default function PluginManager() {
  const [plugins, setPlugins] = useState<Plugin[]>([
    {
      id: '1',
      name: 'Advanced Analytics',
      description: 'Análisis avanzado de datos del robot',
      version: '1.0.0',
      enabled: true,
      author: 'Robot AI Team',
    },
    {
      id: '2',
      name: 'Custom Themes',
      description: 'Temas personalizados adicionales',
      version: '0.5.0',
      enabled: false,
      author: 'Community',
    },
    {
      id: '3',
      name: 'Export Tools',
      description: 'Herramientas adicionales de exportación',
      version: '1.2.0',
      enabled: true,
      author: 'Robot AI Team',
    },
  ]);

  const handleTogglePlugin = (id: string) => {
    setPlugins((prev) =>
      prev.map((plugin) =>
        plugin.id === id
          ? { ...plugin, enabled: !plugin.enabled }
          : plugin
      )
    );
    const plugin = plugins.find((p) => p.id === id);
    toast.success(`${plugin?.name} ${plugin?.enabled ? 'desactivado' : 'activado'}`);
  };

  const handleInstallPlugin = () => {
    toast.info('Funcionalidad de instalación de plugins próximamente');
  };

  const handleUninstallPlugin = (id: string) => {
    setPlugins(plugins.filter((p) => p.id !== id));
    toast.success('Plugin desinstalado');
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Puzzle className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Gestión de Plugins</h3>
          </div>
          <button
            onClick={handleInstallPlugin}
            className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Instalar Plugin
          </button>
        </div>

        {/* Plugins List */}
        <div className="space-y-3">
          {plugins.length === 0 ? (
            <div className="text-center py-12 text-gray-400">
              <Puzzle className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>No hay plugins instalados</p>
            </div>
          ) : (
            plugins.map((plugin) => (
              <div
                key={plugin.id}
                className={`p-4 rounded-lg border ${
                  plugin.enabled
                    ? 'bg-gray-700/50 border-gray-600'
                    : 'bg-gray-800/50 border-gray-700 opacity-60'
                }`}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <h4 className="font-semibold text-white">{plugin.name}</h4>
                      <span className="px-2 py-0.5 bg-primary-500/20 text-primary-400 text-xs rounded">
                        v{plugin.version}
                      </span>
                      {plugin.enabled && (
                        <span className="px-2 py-0.5 bg-green-500/20 text-green-400 text-xs rounded">
                          Activo
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-gray-300 mb-2">{plugin.description}</p>
                    <p className="text-xs text-gray-400">Por: {plugin.author}</p>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleTogglePlugin(plugin.id)}
                      className={`p-2 rounded transition-colors ${
                        plugin.enabled
                          ? 'bg-yellow-600 hover:bg-yellow-700 text-white'
                          : 'bg-green-600 hover:bg-green-700 text-white'
                      }`}
                      title={plugin.enabled ? 'Desactivar' : 'Activar'}
                    >
                      <Power className="w-4 h-4" />
                    </button>
                    <button
                      className="p-2 bg-gray-600 hover:bg-gray-700 text-white rounded transition-colors"
                      title="Configurar"
                    >
                      <Settings className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => handleUninstallPlugin(plugin.id)}
                      className="p-2 bg-red-600 hover:bg-red-700 text-white rounded transition-colors"
                      title="Desinstalar"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>

        {/* Info */}
        <div className="mt-6 p-4 bg-blue-500/10 border border-blue-500/50 rounded-lg">
          <p className="text-sm text-blue-400">
            Los plugins extienden la funcionalidad de la aplicación. Instala plugins de fuentes
            confiables solamente.
          </p>
        </div>
      </div>
    </div>
  );
}


