/**
 * Action buttons for validation detail page
 */

'use client';

import React from 'react';
import { Button } from '@/components/ui';
import { ExportButton } from './ExportButton';
import { ShareButton } from './ShareButton';
import { CopyButton } from '@/components/ui';
import { MoreVertical, Trash2, Edit } from 'lucide-react';
import type { ValidationRead } from '@/lib/types';

export interface ValidationActionsProps {
  validation: ValidationRead;
  onDelete?: () => void;
  onEdit?: () => void;
}

export const ValidationActions: React.FC<ValidationActionsProps> = ({
  validation,
  onDelete,
  onEdit,
}) => {
  const [isMenuOpen, setIsMenuOpen] = React.useState(false);

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Escape') {
      setIsMenuOpen(false);
    }
  };

  return (
    <div className="flex items-center gap-2">
      {validation.has_report && (
        <ExportButton validationId={validation.id} fileName={`validacion-${validation.id.slice(0, 8)}`} />
      )}
      <ShareButton validationId={validation.id} title={`Validación ${validation.id.slice(0, 8)}`} />
      <CopyButton text={validation.id} label="ID" />
      
      {(onDelete || onEdit) && (
        <div className="relative">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsMenuOpen(!isMenuOpen)}
            onKeyDown={handleKeyDown}
            aria-label="Más opciones"
            aria-haspopup="true"
            aria-expanded={isMenuOpen}
            tabIndex={0}
          >
            <MoreVertical className="h-4 w-4" aria-hidden="true" />
          </Button>
          {isMenuOpen && (
            <div
              className="absolute right-0 mt-2 w-48 bg-background border border-input rounded-md shadow-lg z-50"
              role="menu"
              aria-label="Opciones adicionales"
            >
              {onEdit && (
                <button
                  type="button"
                  onClick={() => {
                    onEdit();
                    setIsMenuOpen(false);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      onEdit();
                      setIsMenuOpen(false);
                    }
                  }}
                  className="w-full flex items-center gap-2 px-4 py-2 text-sm text-left hover:bg-accent hover:text-accent-foreground transition-colors"
                  role="menuitem"
                  aria-label="Editar validación"
                  tabIndex={0}
                >
                  <Edit className="h-4 w-4" aria-hidden="true" />
                  Editar
                </button>
              )}
              {onDelete && (
                <button
                  type="button"
                  onClick={() => {
                    onDelete();
                    setIsMenuOpen(false);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      onDelete();
                      setIsMenuOpen(false);
                    }
                  }}
                  className="w-full flex items-center gap-2 px-4 py-2 text-sm text-left text-destructive hover:bg-destructive/10 transition-colors"
                  role="menuitem"
                  aria-label="Eliminar validación"
                  tabIndex={0}
                >
                  <Trash2 className="h-4 w-4" aria-hidden="true" />
                  Eliminar
                </button>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};




