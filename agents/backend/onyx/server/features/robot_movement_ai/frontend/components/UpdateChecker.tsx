'use client';

import { useState, useEffect } from 'react';
import { RefreshCw, Download, CheckCircle, AlertCircle } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface UpdateInfo {
  currentVersion: string;
  latestVersion: string;
  updateAvailable: boolean;
  releaseNotes?: string;
  downloadUrl?: string;
}

export default function UpdateChecker() {
  const [updateInfo, setUpdateInfo] = useState<UpdateInfo>({
    currentVersion: '1.0.0',
    latestVersion: '1.0.0',
    updateAvailable: false,
  });
  const [isChecking, setIsChecking] = useState(false);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);

  const checkForUpdates = async () => {
    setIsChecking(true);
    try {
      // Simulate update check
      await new Promise((resolve) => setTimeout(resolve, 2000));
      
      const mockUpdate: UpdateInfo = {
        currentVersion: '1.0.0',
        latestVersion: '1.1.0',
        updateAvailable: true,
        releaseNotes: 'Nuevas características:\n- Mejoras en rendimiento\n- Nuevos componentes\n- Corrección de bugs',
        downloadUrl: 'https://example.com/download',
      };
      
      setUpdateInfo(mockUpdate);
      setLastChecked(new Date());
      if (mockUpdate.updateAvailable) {
        toast.info('Actualización disponible');
      } else {
        toast.success('Estás usando la última versión');
      }
    } catch (error: any) {
      toast.error(`Error: ${error.message || 'Failed to check for updates'}`);
    } finally {
      setIsChecking(false);
    }
  };

  useEffect(() => {
    checkForUpdates();
  }, []);

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg p-6 border border-gray-200 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
          <div className="flex items-center gap-2">
            <RefreshCw className="w-5 h-5 text-tesla-blue" />
            <h3 className="text-lg font-semibold text-tesla-black">Verificador de Actualizaciones</h3>
          </div>
          <button
            onClick={checkForUpdates}
            disabled={isChecking}
            className="flex items-center gap-2 px-5 py-2 bg-tesla-blue hover:bg-opacity-90 text-white rounded-md transition-all disabled:opacity-50 font-medium min-h-[44px]"
            aria-label="Verificar actualizaciones"
          >
            <RefreshCw className={`w-4 h-4 ${isChecking ? 'animate-spin' : ''}`} />
            Verificar
          </button>
        </div>

        {/* Version Info */}
        <div className="mb-6 p-5 bg-gray-50 rounded-md border border-gray-200">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-tesla-gray-dark mb-2 font-medium">Versión Actual</p>
              <p className="text-lg font-bold text-tesla-black">{updateInfo.currentVersion}</p>
            </div>
            <div>
              <p className="text-sm text-tesla-gray-dark mb-2 font-medium">Última Versión</p>
              <p className="text-lg font-bold text-tesla-black">{updateInfo.latestVersion}</p>
            </div>
          </div>
          {lastChecked && (
            <p className="text-xs text-tesla-gray-dark mt-4 pt-4 border-t border-gray-200">
              Última verificación: {lastChecked.toLocaleString('es-ES')}
            </p>
          )}
        </div>

        {/* Update Status */}
        {updateInfo.updateAvailable ? (
          <div className="p-5 bg-yellow-50 border border-yellow-200 rounded-md">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-yellow-600 mt-0.5 flex-shrink-0" />
              <div className="flex-1">
                <h4 className="font-semibold text-yellow-700 mb-3">Actualización Disponible</h4>
                {updateInfo.releaseNotes && (
                  <div className="mb-4">
                    <p className="text-sm text-tesla-gray-dark whitespace-pre-line">
                      {updateInfo.releaseNotes}
                    </p>
                  </div>
                )}
                {updateInfo.downloadUrl && (
                  <a
                    href={updateInfo.downloadUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 px-5 py-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded-md transition-all font-medium min-h-[44px]"
                    aria-label="Descargar actualización"
                  >
                    <Download className="w-4 h-4" />
                    Descargar Actualización
                  </a>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="p-5 bg-green-50 border border-green-200 rounded-md">
            <div className="flex items-center gap-3">
              <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0" />
              <p className="text-green-700 font-semibold">
                Estás usando la última versión disponible
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}


