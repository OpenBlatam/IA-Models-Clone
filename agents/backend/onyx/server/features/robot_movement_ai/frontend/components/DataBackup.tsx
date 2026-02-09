'use client';

import { useState } from 'react';
import { HardDrive, Download, Upload, Trash2, Clock } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface Backup {
  id: string;
  name: string;
  size: string;
  date: Date;
  type: 'full' | 'incremental';
}

export default function DataBackup() {
  const [backups, setBackups] = useState<Backup[]>([
    {
      id: '1',
      name: 'Backup Completo - 2024-01-15',
      size: '2.5 GB',
      date: new Date(),
      type: 'full',
    },
    {
      id: '2',
      name: 'Backup Incremental - 2024-01-14',
      size: '450 MB',
      date: new Date(Date.now() - 86400000),
      type: 'incremental',
    },
    {
      id: '3',
      name: 'Backup Completo - 2024-01-10',
      size: '2.3 GB',
      date: new Date(Date.now() - 432000000),
      type: 'full',
    },
  ]);

  const handleCreateBackup = () => {
    const newBackup: Backup = {
      id: Date.now().toString(),
      name: `Backup Completo - ${new Date().toISOString().split('T')[0]}`,
      size: '2.5 GB',
      date: new Date(),
      type: 'full',
    };
    setBackups([newBackup, ...backups]);
    toast.success('Backup creado exitosamente');
  };

  const handleDownload = (id: string) => {
    toast.success('Descargando backup...');
  };

  const handleRestore = (id: string) => {
    toast.info('Restaurando backup...');
  };

  const handleDelete = (id: string) => {
    setBackups(backups.filter((b) => b.id !== id));
    toast.success('Backup eliminado');
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <HardDrive className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Respaldo de Datos</h3>
          </div>
          <button
            onClick={handleCreateBackup}
            className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <Upload className="w-4 h-4" />
            Crear Backup
          </button>
        </div>

        {/* Backups List */}
        <div className="space-y-3">
          {backups.map((backup) => (
            <div
              key={backup.id}
              className="p-4 bg-gray-700/50 rounded-lg border border-gray-600"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <h4 className="font-semibold text-white">{backup.name}</h4>
                    <span className={`px-2 py-0.5 rounded text-xs ${
                      backup.type === 'full'
                        ? 'bg-blue-500/20 text-blue-400'
                        : 'bg-green-500/20 text-green-400'
                    }`}>
                      {backup.type === 'full' ? 'Completo' : 'Incremental'}
                    </span>
                  </div>
                  <div className="flex items-center gap-4 text-sm text-gray-300">
                    <span className="flex items-center gap-1">
                      <HardDrive className="w-3 h-3" />
                      {backup.size}
                    </span>
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {backup.date.toLocaleString('es-ES')}
                    </span>
                  </div>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => handleDownload(backup.id)}
                    className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg transition-colors flex items-center gap-2"
                  >
                    <Download className="w-3 h-3" />
                    Descargar
                  </button>
                  <button
                    onClick={() => handleRestore(backup.id)}
                    className="px-3 py-1 bg-green-600 hover:bg-green-700 text-white text-sm rounded-lg transition-colors"
                  >
                    Restaurar
                  </button>
                  <button
                    onClick={() => handleDelete(backup.id)}
                    className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded-lg transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>

        {backups.length === 0 && (
          <div className="text-center py-12 text-gray-400">
            <HardDrive className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No hay backups disponibles</p>
          </div>
        )}
      </div>
    </div>
  );
}


