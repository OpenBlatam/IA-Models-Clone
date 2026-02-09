'use client';

import { useState } from 'react';
import { GitBranch, GitCommit, History, Plus } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface Version {
  id: string;
  name: string;
  description: string;
  timestamp: Date;
  author: string;
  changes: string[];
}

export default function VersionControl() {
  const [versions, setVersions] = useState<Version[]>([
    {
      id: '1',
      name: 'v1.0.0',
      description: 'Versión inicial',
      timestamp: new Date('2024-01-01'),
      author: 'Admin',
      changes: ['Configuración inicial', 'Sistema básico'],
    },
    {
      id: '2',
      name: 'v1.1.0',
      description: 'Mejoras de rendimiento',
      timestamp: new Date('2024-01-15'),
      author: 'Admin',
      changes: ['Optimización de código', 'Mejora de UI'],
    },
  ]);
  const [newVersionName, setNewVersionName] = useState('');
  const [newVersionDesc, setNewVersionDesc] = useState('');

  const handleCreateVersion = () => {
    if (!newVersionName.trim()) {
      toast.error('El nombre de versión es requerido');
      return;
    }

    const newVersion: Version = {
      id: Date.now().toString(),
      name: newVersionName,
      description: newVersionDesc || 'Sin descripción',
      timestamp: new Date(),
      author: 'Usuario',
      changes: ['Cambios pendientes'],
    };

    setVersions([newVersion, ...versions]);
    setNewVersionName('');
    setNewVersionDesc('');
    toast.success(`Versión ${newVersionName} creada`);
  };

  const handleRestoreVersion = (version: Version) => {
    toast.success(`Restaurando versión ${version.name}...`);
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <GitBranch className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Control de Versiones</h3>
        </div>

        {/* Create New Version */}
        <div className="mb-6 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
          <h4 className="text-sm font-medium text-white mb-3 flex items-center gap-2">
            <Plus className="w-4 h-4" />
            Crear Nueva Versión
          </h4>
          <div className="space-y-3">
            <input
              type="text"
              value={newVersionName}
              onChange={(e) => setNewVersionName(e.target.value)}
              placeholder="Nombre de versión (ej: v1.2.0)"
              className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
            <textarea
              value={newVersionDesc}
              onChange={(e) => setNewVersionDesc(e.target.value)}
              placeholder="Descripción de cambios..."
              className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
              rows={3}
            />
            <button
              onClick={handleCreateVersion}
              className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
            >
              Crear Versión
            </button>
          </div>
        </div>

        {/* Versions List */}
        <div className="space-y-4">
          <h4 className="text-sm font-medium text-white flex items-center gap-2">
            <History className="w-4 h-4" />
            Historial de Versiones
          </h4>
          {versions.map((version) => (
            <div
              key={version.id}
              className="p-4 bg-gray-700/50 rounded-lg border border-gray-600 hover:border-primary-500/50 transition-colors"
            >
              <div className="flex items-start justify-between mb-2">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <GitCommit className="w-4 h-4 text-primary-400" />
                    <h5 className="font-semibold text-white">{version.name}</h5>
                  </div>
                  <p className="text-sm text-gray-300">{version.description}</p>
                </div>
                <button
                  onClick={() => handleRestoreVersion(version)}
                  className="px-3 py-1 bg-green-600 hover:bg-green-700 text-white text-sm rounded-lg transition-colors"
                >
                  Restaurar
                </button>
              </div>
              <div className="text-xs text-gray-400 mb-2">
                {version.timestamp.toLocaleString('es-ES')} • Por: {version.author}
              </div>
              <div className="mt-2">
                <p className="text-xs text-gray-400 mb-1">Cambios:</p>
                <ul className="text-xs text-gray-300 space-y-1">
                  {version.changes.map((change, i) => (
                    <li key={i} className="flex items-center gap-2">
                      <span className="w-1 h-1 bg-primary-400 rounded-full" />
                      {change}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}


