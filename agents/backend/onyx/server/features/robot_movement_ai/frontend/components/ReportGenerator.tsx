'use client';

import { useState } from 'react';
import { FileText, Download, Calendar, Filter } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface Report {
  id: string;
  name: string;
  type: 'daily' | 'weekly' | 'monthly' | 'custom';
  format: 'pdf' | 'csv' | 'json';
  dateRange: { start: Date; end: Date };
  status: 'pending' | 'generating' | 'completed' | 'error';
}

export default function ReportGenerator() {
  const [reports, setReports] = useState<Report[]>([
    {
      id: '1',
      name: 'Reporte Diario',
      type: 'daily',
      format: 'pdf',
      dateRange: {
        start: new Date(Date.now() - 86400000),
        end: new Date(),
      },
      status: 'completed',
    },
    {
      id: '2',
      name: 'Reporte Semanal',
      type: 'weekly',
      format: 'csv',
      dateRange: {
        start: new Date(Date.now() - 604800000),
        end: new Date(),
      },
      status: 'completed',
    },
  ]);
  const [newReportName, setNewReportName] = useState('');
  const [newReportType, setNewReportType] = useState<'daily' | 'weekly' | 'monthly' | 'custom'>('daily');
  const [newReportFormat, setNewReportFormat] = useState<'pdf' | 'csv' | 'json'>('pdf');

  const handleGenerate = () => {
    if (!newReportName.trim()) {
      toast.error('El nombre del reporte es requerido');
      return;
    }

    const newReport: Report = {
      id: Date.now().toString(),
      name: newReportName,
      type: newReportType,
      format: newReportFormat,
      dateRange: {
        start: new Date(Date.now() - 86400000),
        end: new Date(),
      },
      status: 'generating',
    };

    setReports([newReport, ...reports]);
    setNewReportName('');

    // Simulate generation
    setTimeout(() => {
      setReports((prev) =>
        prev.map((r) =>
          r.id === newReport.id ? { ...r, status: 'completed' as const } : r
        )
      );
      toast.success('Reporte generado exitosamente');
    }, 2000);
  };

  const handleDownload = (id: string) => {
    toast.success('Descargando reporte...');
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-500/10 border-green-500/50';
      case 'generating':
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
        <div className="flex items-center gap-2 mb-6">
          <FileText className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Generador de Reportes</h3>
        </div>

        {/* Generate Form */}
        <div className="mb-6 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
          <h4 className="text-sm font-medium text-white mb-3">Generar Nuevo Reporte</h4>
          <div className="space-y-3">
            <input
              type="text"
              value={newReportName}
              onChange={(e) => setNewReportName(e.target.value)}
              placeholder="Nombre del reporte"
              className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
            <div className="grid grid-cols-2 gap-3">
              <select
                value={newReportType}
                onChange={(e) => setNewReportType(e.target.value as any)}
                className="px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
              >
                <option value="daily">Diario</option>
                <option value="weekly">Semanal</option>
                <option value="monthly">Mensual</option>
                <option value="custom">Personalizado</option>
              </select>
              <select
                value={newReportFormat}
                onChange={(e) => setNewReportFormat(e.target.value as any)}
                className="px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
              >
                <option value="pdf">PDF</option>
                <option value="csv">CSV</option>
                <option value="json">JSON</option>
              </select>
            </div>
            <button
              onClick={handleGenerate}
              className="w-full px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
            >
              Generar Reporte
            </button>
          </div>
        </div>

        {/* Reports List */}
        <div className="space-y-3">
          {reports.map((report) => (
            <div
              key={report.id}
              className={`p-4 rounded-lg border ${getStatusColor(report.status)}`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <h4 className="font-semibold text-white">{report.name}</h4>
                    <span className="px-2 py-0.5 bg-gray-700 text-gray-300 text-xs rounded capitalize">
                      {report.type}
                    </span>
                    <span className="px-2 py-0.5 bg-gray-700 text-gray-300 text-xs rounded uppercase">
                      {report.format}
                    </span>
                  </div>
                  <p className="text-sm text-gray-300">
                    {report.dateRange.start.toLocaleDateString('es-ES')} -{' '}
                    {report.dateRange.end.toLocaleDateString('es-ES')}
                  </p>
                </div>
                <div className="flex gap-2">
                  {report.status === 'completed' && (
                    <button
                      onClick={() => handleDownload(report.id)}
                      className="px-3 py-1 bg-green-600 hover:bg-green-700 text-white text-sm rounded-lg transition-colors flex items-center gap-2"
                    >
                      <Download className="w-3 h-3" />
                      Descargar
                    </button>
                  )}
                  <span className={`px-2 py-1 rounded text-xs capitalize ${
                    report.status === 'completed'
                      ? 'bg-green-500/20 text-green-400'
                      : report.status === 'generating'
                      ? 'bg-blue-500/20 text-blue-400'
                      : 'bg-red-500/20 text-red-400'
                  }`}>
                    {report.status}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}


