'use client';

import { Button } from '../ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu';
import { Download, MoreVertical } from 'lucide-react';
import { apiClient } from '@/lib/api/client';
import { showErrorToast, showSuccessToast } from '@/lib/utils/error-handler';
import { downloadBlob } from '@/lib/utils/download';
import { MESSAGES } from '@/lib/constants';
import type { ExportMenuProps } from '@/lib/types/components';

export const ExportMenu = ({ manualId }: ExportMenuProps): JSX.Element => {
  const handleExport = async (format: 'markdown' | 'text' | 'json'): Promise<void> => {
    try {
      let blob: Blob;
      let filename: string;

      if (format === 'markdown') {
        blob = await apiClient.exportToMarkdown(manualId);
        filename = `manual_${manualId}.md`;
      } else if (format === 'text') {
        blob = await apiClient.exportToText(manualId);
        filename = `manual_${manualId}.txt`;
      } else {
        blob = await apiClient.exportToJson(manualId);
        filename = `manual_${manualId}.json`;
      }

      downloadBlob(blob, filename);
      showSuccessToast(MESSAGES.EXPORT.SUCCESS(format));
    } catch (error) {
      showErrorToast(error);
    }
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="icon" aria-label="Opciones de exportación">
          <MoreVertical className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem onClick={() => handleExport('markdown')}>
          <Download className="h-4 w-4 mr-2" />
          Exportar como Markdown
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => handleExport('text')}>
          <Download className="h-4 w-4 mr-2" />
          Exportar como Texto
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => handleExport('json')}>
          <Download className="h-4 w-4 mr-2" />
          Exportar como JSON
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

