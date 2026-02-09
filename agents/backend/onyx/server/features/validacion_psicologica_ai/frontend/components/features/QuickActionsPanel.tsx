/**
 * Quick actions panel component
 */

'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, Button } from '@/components/ui';
import { Plus, Upload, Download, Settings, Filter } from 'lucide-react';
import Link from 'next/link';

export interface QuickActionsPanelProps {
  onCreateValidation?: () => void;
  onImport?: () => void;
  onExport?: () => void;
  onSettings?: () => void;
  onFilters?: () => void;
  className?: string;
}

export const QuickActionsPanel: React.FC<QuickActionsPanelProps> = ({
  onCreateValidation,
  onImport,
  onExport,
  onSettings,
  onFilters,
  className,
}) => {
  const actions = [
    {
      id: 'create',
      label: 'Nueva Validación',
      icon: <Plus className="h-4 w-4" aria-hidden="true" />,
      onClick: onCreateValidation,
      variant: 'primary' as const,
    },
    {
      id: 'import',
      label: 'Importar',
      icon: <Upload className="h-4 w-4" aria-hidden="true" />,
      onClick: onImport,
      variant: 'outline' as const,
    },
    {
      id: 'export',
      label: 'Exportar',
      icon: <Download className="h-4 w-4" aria-hidden="true" />,
      onClick: onExport,
      variant: 'outline' as const,
    },
    {
      id: 'filters',
      label: 'Filtros',
      icon: <Filter className="h-4 w-4" aria-hidden="true" />,
      onClick: onFilters,
      variant: 'outline' as const,
    },
    {
      id: 'settings',
      label: 'Configuración',
      icon: <Settings className="h-4 w-4" aria-hidden="true" />,
      onClick: onSettings,
      variant: 'ghost' as const,
    },
  ];

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>Acciones Rápidas</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-2">
          {actions.map((action) => (
            <Button
              key={action.id}
              variant={action.variant}
              onClick={action.onClick}
              onKeyDown={(e) => {
                if (action.onClick && (e.key === 'Enter' || e.key === ' ')) {
                  e.preventDefault();
                  action.onClick();
                }
              }}
              className="w-full justify-start"
              aria-label={action.label}
              tabIndex={0}
            >
              {action.icon}
              <span className="ml-2">{action.label}</span>
            </Button>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};



