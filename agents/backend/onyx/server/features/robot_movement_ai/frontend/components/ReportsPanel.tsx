'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api/client';
import { FileText, Download, Calendar, Filter, BarChart3 } from 'lucide-react';
import { toast } from '@/lib/utils/toast';
import { format, subDays } from 'date-fns';

interface Report {
  id: string;
  type: 'daily' | 'weekly' | 'monthly' | 'custom';
  title: string;
  generatedAt: string;
  data: any;
}

export default function ReportsPanel() {
  const [reports, setReports] = useState<Report[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [reportType, setReportType] = useState<'daily' | 'weekly' | 'monthly'>('daily');
  const [dateRange, setDateRange] = useState({
    start: format(subDays(new Date(), 7), 'yyyy-MM-dd'),
    end: format(new Date(), 'yyyy-MM-dd'),
  });

  const generateReport = async () => {
    setIsGenerating(true);
    try {
      // Simulate report generation
      await new Promise((resolve) => setTimeout(resolve, 2000));

      const report: Report = {
        id: `report-${Date.now()}`,
        type: reportType,
        title: `Reporte ${reportType} - ${format(new Date(), 'dd/MM/yyyy')}`,
        generatedAt: new Date().toISOString(),
        data: {
          movements: Math.floor(Math.random() * 100),
          errors: Math.floor(Math.random() * 10),
          uptime: Math.floor(Math.random() * 100),
          energy: Math.floor(Math.random() * 1000),
        },
      };

      setReports([report, ...reports]);
      toast.success('Reporte generado exitosamente');
    } catch (error: any) {
      toast.error(`Error: ${error.message || 'Failed to generate report'}`);
    } finally {
      setIsGenerating(false);
    }
  };

  const exportReport = (report: Report, format: 'json' | 'csv' = 'json') => {
    let content: string;
    let mimeType: string;
    let extension: string;

    if (format === 'json') {
      content = JSON.stringify(report, null, 2);
      mimeType = 'application/json';
      extension = 'json';
    } else {
      // Simple CSV conversion
      const rows = [
        ['Campo', 'Valor'],
        ['Tipo', report.type],
        ['Título', report.title],
        ['Generado', format(new Date(report.generatedAt), 'dd/MM/yyyy HH:mm:ss')],
        ['Movimientos', report.data.movements],
        ['Errores', report.data.errors],
        ['Uptime', report.data.uptime],
        ['Energía', report.data.energy],
      ];
      content = rows.map((row) => row.join(',')).join('\n');
      mimeType = 'text/csv';
      extension = 'csv';
    }

    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${report.title.replace(/\s+/g, '-')}.${extension}`;
    link.click();
    URL.revokeObjectURL(url);
    toast.success(`Reporte exportado como ${format.toUpperCase()}`);
  };

  return (
    <div className="space-y-6">
      {/* Generate Report */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-4">
          <FileText className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Generar Reporte</h3>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Tipo de Reporte
            </label>
            <select
              value={reportType}
              onChange={(e) => setReportType(e.target.value as any)}
              className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              <option value="daily">Diario</option>
              <option value="weekly">Semanal</option>
              <option value="monthly">Mensual</option>
            </select>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Fecha Inicio
              </label>
              <input
                type="date"
                value={dateRange.start}
                onChange={(e) => setDateRange({ ...dateRange, start: e.target.value })}
                className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Fecha Fin
              </label>
              <input
                type="date"
                value={dateRange.end}
                onChange={(e) => setDateRange({ ...dateRange, end: e.target.value })}
                className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
            </div>
          </div>

          <button
            onClick={generateReport}
            disabled={isGenerating}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <BarChart3 className="w-4 h-4" />
            {isGenerating ? 'Generando...' : 'Generar Reporte'}
          </button>
        </div>
      </div>

      {/* Reports List */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-white">Reportes Generados</h3>
        {reports.length === 0 ? (
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-8 border border-gray-700 text-center text-gray-400">
            No hay reportes generados. Genera uno para comenzar.
          </div>
        ) : (
          reports.map((report) => (
            <div
              key={report.id}
              className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700"
            >
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h4 className="font-semibold text-white mb-1">{report.title}</h4>
                  <p className="text-sm text-gray-400">
                    Generado: {format(new Date(report.generatedAt), 'dd/MM/yyyy HH:mm:ss')}
                  </p>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => exportReport(report, 'json')}
                    className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm transition-colors"
                  >
                    <Download className="w-4 h-4 inline mr-1" />
                    JSON
                  </button>
                  <button
                    onClick={() => exportReport(report, 'csv')}
                    className="px-3 py-1 bg-green-600 hover:bg-green-700 text-white rounded text-sm transition-colors"
                  >
                    <Download className="w-4 h-4 inline mr-1" />
                    CSV
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-4 gap-4">
                <div className="p-3 bg-gray-700/50 rounded">
                  <p className="text-xs text-gray-400 mb-1">Movimientos</p>
                  <p className="text-xl font-bold text-white">{report.data.movements}</p>
                </div>
                <div className="p-3 bg-gray-700/50 rounded">
                  <p className="text-xs text-gray-400 mb-1">Errores</p>
                  <p className="text-xl font-bold text-red-400">{report.data.errors}</p>
                </div>
                <div className="p-3 bg-gray-700/50 rounded">
                  <p className="text-xs text-gray-400 mb-1">Uptime</p>
                  <p className="text-xl font-bold text-green-400">{report.data.uptime}%</p>
                </div>
                <div className="p-3 bg-gray-700/50 rounded">
                  <p className="text-xs text-gray-400 mb-1">Energía</p>
                  <p className="text-xl font-bold text-yellow-400">{report.data.energy}W</p>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

