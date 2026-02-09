'use client';

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Cloud, CloudOff, Download, Upload, Loader2 } from 'lucide-react';
import toast from 'react-hot-toast';

export function MusicBackup() {
  const [backupType, setBackupType] = useState<'local' | 'cloud'>('local');

  const backupMutation = useMutation({
    mutationFn: async (type: 'local' | 'cloud') => {
      // Simular backup
      await new Promise((resolve) => setTimeout(resolve, 2000));
      return { success: true };
    },
    onSuccess: () => {
      toast.success('Backup completado');
    },
    onError: () => {
      toast.error('Error en backup');
    },
  });

  const restoreMutation = useMutation({
    mutationFn: async (type: 'local' | 'cloud') => {
      // Simular restauración
      await new Promise((resolve) => setTimeout(resolve, 2000));
      return { success: true };
    },
    onSuccess: () => {
      toast.success('Restauración completada');
    },
    onError: () => {
      toast.error('Error en restauración');
    },
  });

  const handleBackup = () => {
    backupMutation.mutate(backupType);
  };

  const handleRestore = () => {
    restoreMutation.mutate(backupType);
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Cloud className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Backup y Restauración</h3>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-sm text-gray-400 mb-2">Tipo de Backup</label>
          <div className="flex gap-2">
            <button
              onClick={() => setBackupType('local')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                backupType === 'local'
                  ? 'bg-purple-600 text-white'
                  : 'bg-white/10 text-gray-300 hover:bg-white/20'
              }`}
            >
              <Download className="w-4 h-4" />
              Local
            </button>
            <button
              onClick={() => setBackupType('cloud')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                backupType === 'cloud'
                  ? 'bg-purple-600 text-white'
                  : 'bg-white/10 text-gray-300 hover:bg-white/20'
              }`}
            >
              <Cloud className="w-4 h-4" />
              Nube
            </button>
          </div>
        </div>

        <div className="flex gap-2">
          <button
            onClick={handleBackup}
            disabled={backupMutation.isPending}
            className="flex-1 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
          >
            {backupMutation.isPending ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Haciendo backup...
              </>
            ) : (
              <>
                <Upload className="w-4 h-4" />
                Hacer Backup
              </>
            )}
          </button>
          <button
            onClick={handleRestore}
            disabled={restoreMutation.isPending}
            className="flex-1 px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
          >
            {restoreMutation.isPending ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Restaurando...
              </>
            ) : (
              <>
                <Download className="w-4 h-4" />
                Restaurar
              </>
            )}
          </button>
        </div>

        <div className="p-3 bg-white/5 rounded-lg border border-white/10">
          <p className="text-xs text-gray-400">
            {backupType === 'local'
              ? 'El backup local se guarda en tu navegador'
              : 'El backup en la nube se sincroniza con tu cuenta'}
          </p>
        </div>
      </div>
    </div>
  );
}


