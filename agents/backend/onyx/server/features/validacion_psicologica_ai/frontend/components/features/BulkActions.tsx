/**
 * Bulk actions component for multiple selections
 */

'use client';

import React from 'react';
import { Button, Badge } from '@/components/ui';
import { Trash2, Download, Archive } from 'lucide-react';

export interface BulkActionsProps {
  selectedCount: number;
  onDelete?: () => void;
  onDownload?: () => void;
  onArchive?: () => void;
  onClearSelection?: () => void;
}

export const BulkActions: React.FC<BulkActionsProps> = ({
  selectedCount,
  onDelete,
  onDownload,
  onArchive,
  onClearSelection,
}) => {
  if (selectedCount === 0) {
    return null;
  }

  const handleKeyDown = (event: React.KeyboardEvent, action: () => void) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      action();
    }
  };

  return (
    <div className="flex items-center gap-3 p-4 bg-accent border rounded-lg">
      <div className="flex items-center gap-2">
        <Badge variant="primary">{selectedCount} seleccionado(s)</Badge>
        {onClearSelection && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onClearSelection}
            onKeyDown={(e) => handleKeyDown(e, onClearSelection)}
            aria-label="Limpiar selección"
            tabIndex={0}
          >
            Limpiar
          </Button>
        )}
      </div>
      <div className="flex items-center gap-2 ml-auto">
        {onDownload && (
          <Button
            variant="outline"
            size="sm"
            onClick={onDownload}
            onKeyDown={(e) => handleKeyDown(e, onDownload)}
            aria-label="Descargar seleccionados"
            tabIndex={0}
          >
            <Download className="h-4 w-4 mr-2" aria-hidden="true" />
            Descargar
          </Button>
        )}
        {onArchive && (
          <Button
            variant="outline"
            size="sm"
            onClick={onArchive}
            onKeyDown={(e) => handleKeyDown(e, onArchive)}
            aria-label="Archivar seleccionados"
            tabIndex={0}
          >
            <Archive className="h-4 w-4 mr-2" aria-hidden="true" />
            Archivar
          </Button>
        )}
        {onDelete && (
          <Button
            variant="destructive"
            size="sm"
            onClick={onDelete}
            onKeyDown={(e) => handleKeyDown(e, onDelete)}
            aria-label="Eliminar seleccionados"
            tabIndex={0}
          >
            <Trash2 className="h-4 w-4 mr-2" aria-hidden="true" />
            Eliminar
          </Button>
        )}
      </div>
    </div>
  );
};

export { BulkActions };




