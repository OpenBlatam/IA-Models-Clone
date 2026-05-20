'use client';

import { useState } from 'react';
import { RefreshCw, CheckCircle, XCircle, Loader2 } from 'lucide-react';
import toast from 'react-hot-toast';

export function MusicSync() {
  const [isSyncing, setIsSyncing] = useState(false);
  const [lastSync, setLastSync] = useState<Date | null>(null);
  const [syncStatus, setSyncStatus] = useState<'success' | 'error' | null>(null);

  const handleSync = async () => {
    setIsSyncing(true);
    setSyncStatus(null);

    try {
      // Simular sincronización
      await new Promise((resolve) => setTimeout(resolve, 2000));
      
      setLastSync(new Date());
      setSyncStatus('success');
      toast.success('Sincronización completada');
    } catch (error) {
      setSyncStatus('error');
      toast.error('Error en sincronización');
    } finally {
      setIsSyncing(false);
    }
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {syncStatus === 'success' ? (
            <CheckCircle className="w-4 h-4 text-green-400" />
          ) : syncStatus === 'error' ? (
            <XCircle className="w-4 h-4 text-red-400" />
          ) : (
            <RefreshCw className="w-4 h-4 text-purple-300" />
          )}
          <div>
            <p className="text-sm text-white font-medium">Sincronización</p>
            {lastSync && (
              <p className="text-xs text-gray-400">
                Última sync: {lastSync.toLocaleTimeString()}
              </p>
            )}
          </div>
        </div>
        <button
          onClick={handleSync}
          disabled={isSyncing}
          className="p-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50"
        >
          {isSyncing ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <RefreshCw className="w-4 h-4" />
          )}
        </button>
      </div>
    </div>
  );
}


