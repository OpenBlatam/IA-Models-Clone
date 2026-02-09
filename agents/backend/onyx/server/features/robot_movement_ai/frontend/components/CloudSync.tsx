'use client';

import { useState } from 'react';
import { Cloud, CloudOff, Upload, Download, RefreshCw, CheckCircle } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

export default function CloudSync() {
  const [isConnected, setIsConnected] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [lastSync, setLastSync] = useState<Date | null>(null);
  const [syncStatus, setSyncStatus] = useState<'idle' | 'syncing' | 'success' | 'error'>('idle');

  const handleConnect = async () => {
    setIsSyncing(true);
    try {
      await new Promise((resolve) => setTimeout(resolve, 2000));
      setIsConnected(true);
      toast.success('Conectado a la nube');
    } catch (error) {
      toast.error('Error al conectar');
    } finally {
      setIsSyncing(false);
    }
  };

  const handleDisconnect = () => {
    setIsConnected(false);
    toast.info('Desconectado de la nube');
  };

  const handleSync = async (direction: 'upload' | 'download') => {
    setIsSyncing(true);
    setSyncStatus('syncing');
    try {
      await new Promise((resolve) => setTimeout(resolve, 2000));
      setLastSync(new Date());
      setSyncStatus('success');
      toast.success(
        direction === 'upload' ? 'Datos subidos exitosamente' : 'Datos descargados exitosamente'
      );
    } catch (error) {
      setSyncStatus('error');
      toast.error('Error en la sincronización');
    } finally {
      setIsSyncing(false);
      setTimeout(() => setSyncStatus('idle'), 3000);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Cloud className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Sincronización en la Nube</h3>
        </div>

        {/* Connection Status */}
        <div className="mb-6 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              {isConnected ? (
                <CheckCircle className="w-5 h-5 text-green-400" />
              ) : (
                <CloudOff className="w-5 h-5 text-gray-400" />
              )}
              <span className={`font-semibold ${isConnected ? 'text-green-400' : 'text-gray-400'}`}>
                {isConnected ? 'Conectado' : 'Desconectado'}
              </span>
            </div>
            {isConnected ? (
              <button
                onClick={handleDisconnect}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
              >
                Desconectar
              </button>
            ) : (
              <button
                onClick={handleConnect}
                disabled={isSyncing}
                className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center gap-2"
              >
                {isSyncing && <RefreshCw className="w-4 h-4 animate-spin" />}
                Conectar
              </button>
            )}
          </div>
          {lastSync && (
            <p className="text-xs text-gray-400">
              Última sincronización: {lastSync.toLocaleString('es-ES')}
            </p>
          )}
        </div>

        {/* Sync Actions */}
        {isConnected && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <button
                onClick={() => handleSync('upload')}
                disabled={isSyncing}
                className="p-4 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors disabled:opacity-50 flex flex-col items-center gap-2"
              >
                <Upload className="w-6 h-6" />
                <span>Subir a la Nube</span>
              </button>
              <button
                onClick={() => handleSync('download')}
                disabled={isSyncing}
                className="p-4 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors disabled:opacity-50 flex flex-col items-center gap-2"
              >
                <Download className="w-6 h-6" />
                <span>Descargar de la Nube</span>
              </button>
            </div>

            {isSyncing && (
              <div className="text-center py-4">
                <RefreshCw className="w-8 h-8 mx-auto mb-2 text-primary-400 animate-spin" />
                <p className="text-gray-400">Sincronizando...</p>
              </div>
            )}

            {syncStatus === 'success' && (
              <div className="p-4 bg-green-500/10 border border-green-500/50 rounded-lg">
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-green-400" />
                  <span className="text-green-400">Sincronización exitosa</span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Info */}
        <div className="mt-6 p-4 bg-blue-500/10 border border-blue-500/50 rounded-lg">
          <p className="text-sm text-blue-400">
            La sincronización en la nube te permite respaldar y acceder a tus datos desde
            cualquier dispositivo.
          </p>
        </div>
      </div>
    </div>
  );
}


