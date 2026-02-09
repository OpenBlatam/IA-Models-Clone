/**
 * Validation popover component for quick actions
 */

'use client';

import React from 'react';
import { Popover, Button } from '@/components/ui';
import { MoreVertical, Eye, Download, Share2, Trash2 } from 'lucide-react';
import Link from 'next/link';
import type { ValidationRead } from '@/lib/types';

export interface ValidationPopoverProps {
  validation: ValidationRead;
  onView?: () => void;
  onDownload?: () => void;
  onShare?: () => void;
  onDelete?: () => void;
}

export const ValidationPopover: React.FC<ValidationPopoverProps> = ({
  validation,
  onView,
  onDownload,
  onShare,
  onDelete,
}) => {
  return (
    <Popover
      trigger={
        <Button variant="ghost" size="sm" aria-label="Más opciones">
          <MoreVertical className="h-4 w-4" aria-hidden="true" />
        </Button>
      }
      content={
        <div className="flex flex-col gap-1 min-w-[150px]">
          {onView && (
            <Link href={`/validations/${validation.id}`}>
              <Button
                variant="ghost"
                size="sm"
                className="w-full justify-start"
                onClick={onView}
                aria-label="Ver validación"
                tabIndex={0}
              >
                <Eye className="h-4 w-4 mr-2" aria-hidden="true" />
                Ver
              </Button>
            </Link>
          )}
          {onDownload && validation.has_report && (
            <Button
              variant="ghost"
              size="sm"
              className="w-full justify-start"
              onClick={onDownload}
              aria-label="Descargar reporte"
              tabIndex={0}
            >
              <Download className="h-4 w-4 mr-2" aria-hidden="true" />
              Descargar
            </Button>
          )}
          {onShare && (
            <Button
              variant="ghost"
              size="sm"
              className="w-full justify-start"
              onClick={onShare}
              aria-label="Compartir validación"
              tabIndex={0}
            >
              <Share2 className="h-4 w-4 mr-2" aria-hidden="true" />
              Compartir
            </Button>
          )}
          {onDelete && (
            <Button
              variant="ghost"
              size="sm"
              className="w-full justify-start text-destructive hover:text-destructive"
              onClick={onDelete}
              aria-label="Eliminar validación"
              tabIndex={0}
            >
              <Trash2 className="h-4 w-4 mr-2" aria-hidden="true" />
              Eliminar
            </Button>
          )}
        </div>
      }
      position="bottom"
      align="end"
    />
  );
};



