/**
 * Export button component for reports
 */

'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui';
import { reportsApi } from '@/lib/api/reports';
import { Download, FileText, FileJson, FileCode } from 'lucide-react';
import toast from 'react-hot-toast';

export interface ExportButtonProps {
  validationId: string;
  fileName?: string;
}

type ExportFormat = 'json' | 'pdf' | 'html';

const EXPORT_OPTIONS = [
  { value: 'json', label: 'JSON', icon: <FileJson className="h-4 w-4" /> },
  { value: 'pdf', label: 'PDF', icon: <FileText className="h-4 w-4" /> },
  { value: 'html', label: 'HTML', icon: <FileCode className="h-4 w-4" /> },
];

export const ExportButton: React.FC<ExportButtonProps> = ({
  validationId,
  fileName = 'reporte',
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  const handleExport = async (format: ExportFormat) => {
    if (!validationId) {
      toast.error('ID de validación no válido');
      return;
    }

    setIsExporting(true);
    setIsOpen(false);

    try {
      const blob = await reportsApi.export(validationId, format);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${fileName}.${format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      toast.success(`Reporte exportado como ${format.toUpperCase()}`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Error al exportar';
      toast.error(`Error al exportar: ${errorMessage}`);
    } finally {
      setIsExporting(false);
    }
  };

  const handleToggle = () => {
    if (isExporting) {
      return;
    }
    setIsOpen(!isOpen);
  };

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Escape') {
      setIsOpen(false);
    }
  };

  return (
    <div ref={dropdownRef} className="relative">
      <Button
        variant="outline"
        size="sm"
        onClick={handleToggle}
        onKeyDown={handleKeyDown}
        isLoading={isExporting}
        aria-label="Exportar reporte"
        aria-haspopup="true"
        aria-expanded={isOpen}
        tabIndex={0}
      >
        <Download className="h-4 w-4 mr-2" aria-hidden="true" />
        Exportar
      </Button>
      {isOpen && (
        <div
          className="absolute right-0 mt-2 w-48 bg-background border border-input rounded-md shadow-lg z-50"
          role="menu"
          aria-label="Opciones de exportación"
        >
          {EXPORT_OPTIONS.map((option) => (
            <button
              key={option.value}
              type="button"
              onClick={() => handleExport(option.value as ExportFormat)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  handleExport(option.value as ExportFormat);
                }
              }}
              disabled={isExporting}
              className="w-full flex items-center gap-2 px-4 py-2 text-sm text-left hover:bg-accent hover:text-accent-foreground transition-colors disabled:opacity-50 disabled:cursor-not-allowed focus-visible:outline-none focus-visible:bg-accent"
              role="menuitem"
              aria-label={`Exportar como ${option.label}`}
              tabIndex={0}
            >
              {option.icon}
              <span>{option.label}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};
