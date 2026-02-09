/**
 * Validation tabs component for organizing validation details
 */

'use client';

import React from 'react';
import { Tabs } from '@/components/ui';
import { ProfileDisplay } from './ProfileDisplay';
import { ReportViewer } from './ReportViewer';
import { FileText, User, BarChart3 } from 'lucide-react';
import type { ValidationRead } from '@/lib/types';

export interface ValidationTabsProps {
  validation: ValidationRead;
}

export const ValidationTabs: React.FC<ValidationTabsProps> = ({ validation }) => {
  const tabs = [
    {
      id: 'profile',
      label: 'Perfil Psicológico',
      icon: <User className="h-4 w-4" aria-hidden="true" />,
      content: validation.has_profile ? (
        <ProfileDisplay validationId={validation.id} />
      ) : (
        <div className="text-center py-12 text-muted-foreground">
          No hay perfil psicológico disponible para esta validación
        </div>
      ),
      disabled: !validation.has_profile,
    },
    {
      id: 'report',
      label: 'Reporte',
      icon: <FileText className="h-4 w-4" aria-hidden="true" />,
      content: validation.has_report ? (
        <ReportViewer validationId={validation.id} />
      ) : (
        <div className="text-center py-12 text-muted-foreground">
          No hay reporte disponible para esta validación
        </div>
      ),
      disabled: !validation.has_report,
    },
    {
      id: 'analytics',
      label: 'Analíticas',
      icon: <BarChart3 className="h-4 w-4" aria-hidden="true" />,
      content: (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="p-4 border rounded-lg">
              <div className="text-sm text-muted-foreground">Estado</div>
              <div className="text-lg font-semibold mt-1">{validation.status}</div>
            </div>
            <div className="p-4 border rounded-lg">
              <div className="text-sm text-muted-foreground">Plataformas</div>
              <div className="text-lg font-semibold mt-1">
                {validation.connected_platforms.length}
              </div>
            </div>
          </div>
          <div className="p-4 border rounded-lg">
            <div className="text-sm text-muted-foreground mb-2">Plataformas Conectadas</div>
            <div className="flex flex-wrap gap-2">
              {validation.connected_platforms.map((platform) => (
                <span
                  key={platform}
                  className="px-2 py-1 text-xs bg-muted rounded"
                >
                  {platform}
                </span>
              ))}
            </div>
          </div>
        </div>
      ),
    },
  ];

  return <Tabs items={tabs} defaultActiveId="profile" variant="underline" />;
};



