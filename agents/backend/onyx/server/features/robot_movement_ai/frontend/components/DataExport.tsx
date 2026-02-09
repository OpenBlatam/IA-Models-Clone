'use client';

import { useState } from 'react';
import { apiClient } from '@/lib/api/client';
import { Download, FileJson, FileSpreadsheet, FileText, Database } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

export default function DataExport() {
  const [isExporting, setIsExporting] = useState(false);
  const [exportType, setExportType] = useState<'all' | 'status' | 'metrics' | 'history' | 'config'>('all');

  const exportData = async (format: 'json' | 'csv' | 'txt') => {
    setIsExporting(true);
    try {
      let data: any = {};

      switch (exportType) {
        case 'all':
          const [status, metrics, history, config] = await Promise.all([
            apiClient.getStatus().catch(() => null),
            apiClient.getMetrics().catch(() => null),
            apiClient.getMovementHistory(100).catch(() => null),
            apiClient.getConfig().catch(() => null),
          ]);
          data = { status, metrics, history, config, exportedAt: new Date().toISOString() };
          break;
        case 'status':
          data = await apiClient.getStatus();
          break;
        case 'metrics':
          data = await apiClient.getMetrics();
          break;
        case 'history':
          data = await apiClient.getMovementHistory(100);
          break;
        case 'config':
          data = await apiClient.getConfig();
          break;
      }

      let content: string;
      let mimeType: string;
      let extension: string;
      let filename: string;

      if (format === 'json') {
        content = JSON.stringify(data, null, 2);
        mimeType = 'application/json';
        extension = 'json';
        filename = `robot-data-${exportType}-${Date.now()}.json`;
      } else if (format === 'csv') {
        // Simple CSV conversion
        const rows: string[][] = [];
        const flatten = (obj: any, prefix = ''): void => {
          for (const key in obj) {
            const value = obj[key];
            const newKey = prefix ? `${prefix}.${key}` : key;
            if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
              flatten(value, newKey);
            } else {
              rows.push([newKey, String(value)]);
            }
          }
        };
        flatten(data);
        content = ['Campo,Valor', ...rows.map((row) => row.join(','))].join('\n');
        mimeType = 'text/csv';
        extension = 'csv';
        filename = `robot-data-${exportType}-${Date.now()}.csv`;
      } else {
        content = JSON.stringify(data, null, 2);
        mimeType = 'text/plain';
        extension = 'txt';
        filename = `robot-data-${exportType}-${Date.now()}.txt`;
      }

      const blob = new Blob([content], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      link.click();
      URL.revokeObjectURL(url);

      toast.success(`Datos exportados como ${format.toUpperCase()}`);
    } catch (error: any) {
      toast.error(`Error: ${error.message || 'Failed to export data'}`);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Database className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Exportar Datos</h3>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Tipo de Datos
            </label>
            <select
              value={exportType}
              onChange={(e) => setExportType(e.target.value as any)}
              className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              <option value="all">Todos los Datos</option>
              <option value="status">Estado del Robot</option>
              <option value="metrics">Métricas</option>
              <option value="history">Historial de Movimientos</option>
              <option value="config">Configuración</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Formato de Exportación
            </label>
            <div className="grid grid-cols-3 gap-3">
              <button
                onClick={() => exportData('json')}
                disabled={isExporting}
                className="flex flex-col items-center gap-2 p-4 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors disabled:opacity-50"
              >
                <FileJson className="w-8 h-8 text-yellow-400" />
                <span className="text-white text-sm">JSON</span>
              </button>
              <button
                onClick={() => exportData('csv')}
                disabled={isExporting}
                className="flex flex-col items-center gap-2 p-4 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors disabled:opacity-50"
              >
                <FileSpreadsheet className="w-8 h-8 text-green-400" />
                <span className="text-white text-sm">CSV</span>
              </button>
              <button
                onClick={() => exportData('txt')}
                disabled={isExporting}
                className="flex flex-col items-center gap-2 p-4 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors disabled:opacity-50"
              >
                <FileText className="w-8 h-8 text-blue-400" />
                <span className="text-white text-sm">TXT</span>
              </button>
            </div>
          </div>

          {isExporting && (
            <div className="text-center py-4">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-primary-400"></div>
              <p className="mt-2 text-gray-400">Exportando datos...</p>
            </div>
          )}
        </div>
      </div>

      {/* Export History */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <h4 className="text-lg font-semibold text-white mb-4">Historial de Exportaciones</h4>
        <div className="text-center text-gray-400 py-8">
          <Database className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>No hay exportaciones recientes</p>
        </div>
      </div>
    </div>
  );
}


