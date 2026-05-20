'use client';

import { Download, FileText, FileSpreadsheet } from 'lucide-react';
import { Button } from './Button';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import { cn } from '@/lib/utils';

interface ExportButtonProps {
  onExportCSV?: () => void;
  onExportPDF?: () => void;
  onExportExcel?: () => void;
  className?: string;
  disabled?: boolean;
}

export const ExportButton = ({
  onExportCSV,
  onExportPDF,
  onExportExcel,
  className,
  disabled = false,
}: ExportButtonProps) => {
  const hasExports = onExportCSV || onExportPDF || onExportExcel;

  if (!hasExports) return null;

  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <Button
          variant="secondary"
          size="sm"
          className={cn('flex items-center gap-2', className)}
          disabled={disabled}
          aria-label="Exportar datos"
        >
          <Download className="h-4 w-4" />
          Exportar
        </Button>
      </DropdownMenu.Trigger>

      <DropdownMenu.Portal>
        <DropdownMenu.Content
          className={cn(
            'z-50 min-w-[180px] rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-1 shadow-lg',
            'animate-in fade-in-0 zoom-in-95'
          )}
          sideOffset={5}
          align="end"
        >
          {onExportCSV && (
            <DropdownMenu.Item asChild>
              <button
                type="button"
                onClick={onExportCSV}
                className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors outline-none"
              >
                <FileText className="h-4 w-4" />
                Exportar CSV
              </button>
            </DropdownMenu.Item>
          )}
          {onExportExcel && (
            <DropdownMenu.Item asChild>
              <button
                type="button"
                onClick={onExportExcel}
                className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors outline-none"
              >
                <FileSpreadsheet className="h-4 w-4" />
                Exportar Excel
              </button>
            </DropdownMenu.Item>
          )}
          {onExportPDF && (
            <DropdownMenu.Item asChild>
              <button
                type="button"
                onClick={onExportPDF}
                className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors outline-none"
              >
                <FileText className="h-4 w-4" />
                Exportar PDF
              </button>
            </DropdownMenu.Item>
          )}
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
};



