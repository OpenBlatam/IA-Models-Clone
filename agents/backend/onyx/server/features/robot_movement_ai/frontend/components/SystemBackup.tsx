'use client';

import { useState } from 'react';
import { Download, Upload, Database, Clock, CheckCircle } from 'lucide-react';
import { toast } from '@/lib/utils/toast';
import { format } from 'date-fns';

interface Backup {
  id: string;
  name: string;
  timestamp: Date;
  size: string;
  type: 'full' | 'config' | 'data';
}

export default function SystemBackup() {
  const [backups, setBackups] = useState<Backup[]>([
    {
      id: '1',
      name: 'Backup completo',
      timestamp: new Date(),
      size: '2.5 MB',
      type: 'full',
    },
  ]);
  const [isCreating, setIsCreating] = useState(false);

  const handleCreateBackup = async (type: 'full' | 'config' | 'data') => {
    setIsCreating(true);
    try {
      // Simulate backup creation
      await new Promise((resolve) => setTimeout(resolve, 2000));

      const backup: Backup = {
        id: Date.now().toString(),
        name: `Backup ${type === 'full' ? 'completo' : type === 'config' ? 'configuración' : 'datos'}`,
        timestamp: new Date(),
        size: `${(Math.random() * 5).toFixed(2)} MB`,
        type,
      };

      setBackups([backup, ...backups]);
      toast.success('Backup creado exitosamente');
    } catch (error: any) {
      toast.error(`Error: ${error.message || 'Failed to create backup'}`);
    } finally {
      setIsCreating(false);
    }
  };

  const handleDownloadBackup = (backup: Backup) => {
    // Would download backup file
    toast.success(`Descargando ${backup.name}`);
  };

  const handleRestoreBackup = (backup: Backup) => {
    // Would restore from backup
    toast.info(`Restaurando desde ${backup.name}`);
  };

  const handleUploadBackup = () => {
    // Would trigger file upload
    toast.info('Funcionalidad de carga próximamente');
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Database className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Sistema de Backup</h3>
        </div>

        {/* Create Backup */}
        <div className="mb-6">
          <h4 className="text-sm font-medium text-gray-300 mb-3">Crear Backup</h4>
          <div className="grid grid-cols-3 gap-3">
            <button
              onClick={() => handleCreateBackup('full')}
              disabled={isCreating}
              className="p-4 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors disabled:opacity-50 flex flex-col items-center gap-2"
            >
              <Database className="w-6 h-6 text-blue-400" />
              <span className="text-white text-sm">Completo</span>
            </button>
            <button
              onClick={() => handleCreateBackup('config')}
              disabled={isCreating}
              className="p-4 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors disabled:opacity-50 flex flex-col items-center gap-2"
            >
              <Database className="w-6 h-6 text-green-400" />
              <span className="text-white text-sm">Configuración</span>
            </button>
            <button
              onClick={() => handleCreateBackup('data')}
              disabled={isCreating}
              className="p-4 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors disabled:opacity-50 flex flex-col items-center gap-2"
            >
              <Database className="w-6 h-6 text-yellow-400" />
              <span className="text-white text-sm">Datos</span>
            </button>
          </div>
          {isCreating && (
            <div className="mt-4 text-center text-gray-400">
              <div className="inline-block animate-spin rounded-full h-6 w-6 border-b-2 border-primary-400"></div>
              <p className="mt-2">Creando backup...</p>
            </div>
          )}
        </div>

        {/* Upload Backup */}
        <div className="mb-6">
          <button
            onClick={handleUploadBackup}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
          >
            <Upload className="w-4 h-4" />
            Cargar Backup
          </button>
        </div>

        {/* Backups List */}
        <div>
          <h4 className="text-sm font-medium text-gray-300 mb-3">Backups Disponibles</h4>
          <div className="space-y-2">
            {backups.length === 0 ? (
              <div className="text-center py-8 text-gray-400">
                <Database className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>No hay backups disponibles</p>
              </div>
            ) : (
              backups.map((backup) => (
                <div
                  key={backup.id}
                  className="p-4 bg-gray-700/50 rounded-lg border border-gray-600"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <CheckCircle className="w-4 h-4 text-green-400" />
                        <h5 className="font-semibold text-white">{backup.name}</h5>
                        <span className="px-2 py-1 bg-primary-500/20 text-primary-400 text-xs rounded">
                          {backup.type}
                        </span>
                      </div>
                      <div className="flex items-center gap-4 text-xs text-gray-400">
                        <div className="flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          {format(backup.timestamp, 'dd/MM/yyyy HH:mm')}
                        </div>
                        <span>{backup.size}</span>
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <button
                        onClick={() => handleRestoreBackup(backup)}
                        className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded transition-colors"
                      >
                        Restaurar
                      </button>
                      <button
                        onClick={() => handleDownloadBackup(backup)}
                        className="px-3 py-1 bg-gray-600 hover:bg-gray-700 text-white text-sm rounded transition-colors flex items-center gap-1"
                      >
                        <Download className="w-3 h-3" />
                        Descargar
                      </button>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}


