'use client';

import { useCallback } from 'react';
import { toast } from 'sonner';

interface ExportOptions {
  filename?: string;
  headers?: string[];
  data: any[];
}

export const useExport = () => {
  const exportToCSV = useCallback(({ filename = 'export', headers, data }: ExportOptions) => {
    try {
      if (!data || data.length === 0) {
        toast.error('No hay datos para exportar');
        return;
      }

      const csvHeaders = headers || Object.keys(data[0]);
      const csvRows = [
        csvHeaders.join(','),
        ...data.map((row) =>
          csvHeaders.map((header) => {
            const value = row[header];
            return typeof value === 'string' && value.includes(',')
              ? `"${value}"`
              : value;
          }).join(',')
        ),
      ];

      const csvContent = csvRows.join('\n');
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const link = document.createElement('a');
      const url = URL.createObjectURL(blob);

      link.setAttribute('href', url);
      link.setAttribute('download', `${filename}.csv`);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      toast.success('Archivo CSV exportado exitosamente');
    } catch (error) {
      toast.error('Error al exportar CSV');
      console.error('Export error:', error);
    }
  }, []);

  const exportToJSON = useCallback(({ filename = 'export', data }: ExportOptions) => {
    try {
      if (!data || data.length === 0) {
        toast.error('No hay datos para exportar');
        return;
      }

      const jsonContent = JSON.stringify(data, null, 2);
      const blob = new Blob([jsonContent], { type: 'application/json' });
      const link = document.createElement('a');
      const url = URL.createObjectURL(blob);

      link.setAttribute('href', url);
      link.setAttribute('download', `${filename}.json`);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      toast.success('Archivo JSON exportado exitosamente');
    } catch (error) {
      toast.error('Error al exportar JSON');
      console.error('Export error:', error);
    }
  }, []);

  return {
    exportToCSV,
    exportToJSON,
  };
};



