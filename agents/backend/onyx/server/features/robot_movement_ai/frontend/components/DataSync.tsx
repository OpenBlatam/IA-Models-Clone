'use client';

import { useState } from 'react';
import { RefreshCw, CheckCircle, XCircle, Clock } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface SyncStatus {
  id: string;
  name: string;
  status: 'synced' | 'syncing' | 'error' | 'pending';
  lastSync?: Date;
  nextSync?: Date;
}

export default function DataSync() {
  const [syncs, setSyncs] = useState<SyncStatus[]>([
    {
      id: '1',
      name: 'Sincronización Principal',
      status: 'synced',
      lastSync: new Date(),
      nextSync: new Date(Date.now() + 3600000),
    },
    {
      id: '2',
      name: 'Backup en la Nube',
      status: 'synced',
      lastSync: new Date(Date.now() - 7200000),
      nextSync: new Date(Date.now() + 3600000),
    },
    {
      id: '3',
      name: 'Sincronización Externa',
      status: 'error',
      lastSync: new Date(Date.now() - 86400000),
    },
  ]);

  const handleSync = async (id: string) => {
    setSyncs((prev) =>
      prev.map((s) => (s.id === id ? { ...s, status: 'syncing' as const } : s))
    );

    // Simulate sync
    await new Promise((resolve) => setTimeout(resolve, 2000));

    const success = Math.random() > 0.2;
    setSyncs((prev) =>
      prev.map((s) =>
        s.id === id
          ? {
              ...s,
              status: (success ? 'synced' : 'error') as const,
              lastSync: new Date(),
              nextSync: success ? new Date(Date.now() + 3600000) : s.nextSync,
            }
          : s
      )
    );

    toast[success ? 'success' : 'error'](
      success ? 'Sincronización exitosa' : 'Error en la sincronización'
    );
  };

  const handleSyncAll = async () => {
    toast.info('Sincronizando todos...');
    for (const sync of syncs) {
      await handleSync(sync.id);
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
    toast.success('Sincronización completa');
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'synced':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'syncing':
        return <RefreshCw className="w-5 h-5 text-blue-400 animate-spin" />;
      case 'error':
        return <XCircle className="w-5 h-5 text-red-400" />;
      default:
        return <Clock className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'synced':
        return 'bg-green-500/10 border-green-500/50';
      case 'syncing':
        return 'bg-blue-500/10 border-blue-500/50';
      case 'error':
        return 'bg-red-500/10 border-red-500/50';
      default:
        return 'bg-gray-700/50 border-gray-600';
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <RefreshCw className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Sincronización de Datos</h3>
          </div>
          <button
            onClick={handleSyncAll}
            className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <RefreshCw className="w-4 h-4" />
            Sincronizar Todo
          </button>
        </div>

        {/* Syncs List */}
        <div className="space-y-3">
          {syncs.map((sync) => (
            <div
              key={sync.id}
              className={`p-4 rounded-lg border ${getStatusColor(sync.status)}`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-3 flex-1">
                  {getStatusIcon(sync.status)}
                  <div className="flex-1">
                    <h4 className="font-semibold text-white mb-1">{sync.name}</h4>
                    {sync.lastSync && (
                      <p className="text-sm text-gray-300 mb-1">
                        Última sincronización: {sync.lastSync.toLocaleString('es-ES')}
                      </p>
                    )}
                    {sync.nextSync && sync.status === 'synced' && (
                      <p className="text-sm text-green-400">
                        Próxima sincronización: {sync.nextSync.toLocaleString('es-ES')}
                      </p>
                    )}
                  </div>
                </div>
                <button
                  onClick={() => handleSync(sync.id)}
                  disabled={sync.status === 'syncing'}
                  className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                >
                  <RefreshCw className="w-4 h-4" />
                  Sincronizar
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}


