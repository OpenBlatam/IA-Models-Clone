'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiFileText, FiDownload, FiX, FiCalendar, FiFilter } from 'react-icons/fi';
import { format } from 'date-fns';
import { apiClient } from '@/lib/api-client';
import { showToast } from '@/lib/toast';
import type { TaskListItem } from '@/types/api';

interface ReportConfig {
  type: 'summary' | 'detailed' | 'analytics';
  dateRange: {
    start: Date;
    end: Date;
  };
  status?: string[];
  businessArea?: string[];
  includeMetadata: boolean;
  format: 'json' | 'csv' | 'pdf';
}

export default function ReportGenerator() {
  const [isOpen, setIsOpen] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [config, setConfig] = useState<ReportConfig>({
    type: 'summary',
    dateRange: {
      start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // 30 days ago
      end: new Date(),
    },
    includeMetadata: true,
    format: 'json',
  });

  const generateReport = async () => {
    setIsGenerating(true);
    try {
      // Load all tasks
      const response = await apiClient.listTasks({ limit: 10000 });
      let filteredTasks = response.tasks;

      // Filter by date range
      filteredTasks = filteredTasks.filter((task) => {
        const taskDate = new Date(task.created_at);
        return (
          taskDate >= config.dateRange.start && taskDate <= config.dateRange.end
        );
      });

      // Filter by status
      if (config.status && config.status.length > 0) {
        filteredTasks = filteredTasks.filter((task) =>
          config.status!.includes(task.status)
        );
      }

      // Generate report based on type
      let reportData: any;

      if (config.type === 'summary') {
        reportData = {
          reportType: 'summary',
          generatedAt: new Date().toISOString(),
          dateRange: {
            start: config.dateRange.start.toISOString(),
            end: config.dateRange.end.toISOString(),
          },
          totalTasks: filteredTasks.length,
          byStatus: filteredTasks.reduce((acc, task) => {
            acc[task.status] = (acc[task.status] || 0) + 1;
            return acc;
          }, {} as Record<string, number>),
          byBusinessArea: filteredTasks.reduce((acc, task) => {
            const area = (task as any).business_area || 'unknown';
            acc[area] = (acc[area] || 0) + 1;
            return acc;
          }, {} as Record<string, number>),
        };
      } else if (config.type === 'detailed') {
        reportData = {
          reportType: 'detailed',
          generatedAt: new Date().toISOString(),
          dateRange: {
            start: config.dateRange.start.toISOString(),
            end: config.dateRange.end.toISOString(),
          },
          tasks: filteredTasks.map((task) => ({
            task_id: task.task_id,
            status: task.status,
            query_preview: task.query_preview,
            created_at: task.created_at,
            ...(config.includeMetadata && {
              metadata: (task as any).metadata,
            }),
          })),
        };
      } else {
        // Analytics
        reportData = {
          reportType: 'analytics',
          generatedAt: new Date().toISOString(),
          dateRange: {
            start: config.dateRange.start.toISOString(),
            end: config.dateRange.end.toISOString(),
          },
          totalTasks: filteredTasks.length,
          completed: filteredTasks.filter((t) => t.status === 'completed').length,
          failed: filteredTasks.filter((t) => t.status === 'failed').length,
          processing: filteredTasks.filter((t) => t.status === 'processing').length,
          successRate:
            filteredTasks.length > 0
              ? (
                  (filteredTasks.filter((t) => t.status === 'completed').length /
                    filteredTasks.length) *
                  100
                ).toFixed(2)
              : '0',
          tasksByDay: filteredTasks.reduce((acc, task) => {
            const date = format(new Date(task.created_at), 'yyyy-MM-dd');
            acc[date] = (acc[date] || 0) + 1;
            return acc;
          }, {} as Record<string, number>),
        };
      }

      // Export based on format
      if (config.format === 'json') {
        const blob = new Blob([JSON.stringify(reportData, null, 2)], {
          type: 'application/json',
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `report-${config.type}-${format(new Date(), 'yyyy-MM-dd')}.json`;
        a.click();
        URL.revokeObjectURL(url);
      } else if (config.format === 'csv') {
        // Convert to CSV
        let csv = '';
        if (config.type === 'summary') {
          csv = 'Metric,Value\n';
          csv += `Total Tasks,${reportData.totalTasks}\n`;
          Object.entries(reportData.byStatus || {}).forEach(([status, count]) => {
            csv += `${status},${count}\n`;
          });
        } else if (config.type === 'detailed') {
          csv = 'Task ID,Status,Query Preview,Created At\n';
          reportData.tasks.forEach((task: any) => {
            csv += `"${task.task_id}","${task.status}","${task.query_preview}","${task.created_at}"\n`;
          });
        }

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `report-${config.type}-${format(new Date(), 'yyyy-MM-dd')}.csv`;
        a.click();
        URL.revokeObjectURL(url);
      }

      showToast('Reporte generado exitosamente', 'success');
      setIsOpen(false);
    } catch (error: any) {
      showToast(error.message || 'Error al generar reporte', 'error');
    } finally {
      setIsGenerating(false);
    }
  };

  if (!isOpen) {
    return (
      <button onClick={() => setIsOpen(true)} className="btn btn-secondary">
        <FiFileText size={18} className="mr-2" />
        Generar Reporte
      </button>
    );
  }

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4"
        onClick={() => setIsOpen(false)}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          className="bg-white dark:bg-gray-800 rounded-xl shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="p-6 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white">
              Generar Reporte
            </h3>
            <button onClick={() => setIsOpen(false)} className="btn-icon">
              <FiX size={20} />
            </button>
          </div>

          <div className="p-6 space-y-6">
            {/* Report Type */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Tipo de Reporte
              </label>
              <div className="grid grid-cols-3 gap-3">
                {(['summary', 'detailed', 'analytics'] as const).map((type) => (
                  <button
                    key={type}
                    onClick={() => setConfig({ ...config, type })}
                    className={`p-3 border rounded-lg text-sm font-medium transition-colors ${
                      config.type === type
                        ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-400'
                        : 'border-gray-200 dark:border-gray-700 hover:border-primary-300'
                    }`}
                  >
                    {type === 'summary' && 'Resumen'}
                    {type === 'detailed' && 'Detallado'}
                    {type === 'analytics' && 'Analíticas'}
                  </button>
                ))}
              </div>
            </div>

            {/* Date Range */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Rango de Fechas
              </label>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-gray-500 dark:text-gray-400 mb-1 block">
                    Desde
                  </label>
                  <input
                    type="date"
                    value={format(config.dateRange.start, 'yyyy-MM-dd')}
                    onChange={(e) =>
                      setConfig({
                        ...config,
                        dateRange: {
                          ...config.dateRange,
                          start: new Date(e.target.value),
                        },
                      })
                    }
                    className="input w-full"
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-500 dark:text-gray-400 mb-1 block">
                    Hasta
                  </label>
                  <input
                    type="date"
                    value={format(config.dateRange.end, 'yyyy-MM-dd')}
                    onChange={(e) =>
                      setConfig({
                        ...config,
                        dateRange: {
                          ...config.dateRange,
                          end: new Date(e.target.value),
                        },
                      })
                    }
                    className="input w-full"
                  />
                </div>
              </div>
            </div>

            {/* Format */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Formato de Exportación
              </label>
              <div className="flex gap-3">
                {(['json', 'csv', 'pdf'] as const).map((format) => (
                  <button
                    key={format}
                    onClick={() => setConfig({ ...config, format })}
                    className={`px-4 py-2 border rounded-lg text-sm font-medium transition-colors ${
                      config.format === format
                        ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-400'
                        : 'border-gray-200 dark:border-gray-700 hover:border-primary-300'
                    }`}
                  >
                    {format.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>

            {/* Options */}
            {config.type === 'detailed' && (
              <div>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={config.includeMetadata}
                    onChange={(e) =>
                      setConfig({ ...config, includeMetadata: e.target.checked })
                    }
                    className="rounded"
                  />
                  <span className="text-sm text-gray-700 dark:text-gray-300">
                    Incluir metadatos
                  </span>
                </label>
              </div>
            )}

            <div className="flex gap-3 pt-4">
              <button
                onClick={generateReport}
                disabled={isGenerating}
                className="btn btn-primary flex-1"
              >
                {isGenerating ? (
                  <>
                    <span className="animate-spin h-5 w-5 mr-3 rounded-full border-b-2 border-white"></span>
                    Generando...
                  </>
                ) : (
                  <>
                    <FiDownload size={18} className="mr-2" />
                    Generar Reporte
                  </>
                )}
              </button>
              <button
                onClick={() => setIsOpen(false)}
                className="btn btn-secondary"
              >
                Cancelar
              </button>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}


