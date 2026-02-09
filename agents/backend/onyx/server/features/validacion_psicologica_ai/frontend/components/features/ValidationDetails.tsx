/**
 * Validation details component with collapsible sections
 */

'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, Collapsible, Badge } from '@/components/ui';
import { format } from 'date-fns';
import { Calendar, Clock, CheckCircle2, XCircle, Play, AlertCircle } from 'lucide-react';
import type { ValidationRead } from '@/lib/types';

export interface ValidationDetailsProps {
  validation: ValidationRead;
}

const getStatusIcon = (status: ValidationRead['status']) => {
  switch (status) {
    case 'completed':
      return <CheckCircle2 className="h-4 w-4 text-green-600" aria-hidden="true" />;
    case 'failed':
      return <XCircle className="h-4 w-4 text-red-600" aria-hidden="true" />;
    case 'running':
      return <Play className="h-4 w-4 text-blue-600" aria-hidden="true" />;
    default:
      return <AlertCircle className="h-4 w-4 text-gray-600" aria-hidden="true" />;
  }
};

const getStatusVariant = (status: ValidationRead['status']): 'success' | 'destructive' | 'warning' | 'info' | 'default' => {
  switch (status) {
    case 'completed':
      return 'success';
    case 'failed':
      return 'destructive';
    case 'running':
      return 'info';
    case 'cancelled':
      return 'warning';
    default:
      return 'default';
  }
};

export const ValidationDetails: React.FC<ValidationDetailsProps> = ({ validation }) => {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Detalles de la Validación</CardTitle>
          <Badge variant={getStatusVariant(validation.status)}>
            {validation.status}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <Collapsible
          title="Información General"
          defaultOpen
          icon={<Calendar className="h-4 w-4" aria-hidden="true" />}
        >
          <div className="space-y-2 text-sm">
            <div>
              <span className="font-medium">ID:</span> {validation.id}
            </div>
            <div>
              <span className="font-medium">Creada:</span>{' '}
              <time dateTime={validation.created_at}>
                {format(new Date(validation.created_at), 'PPp')}
              </time>
            </div>
            {validation.completed_at && (
              <div>
                <span className="font-medium">Completada:</span>{' '}
                <time dateTime={validation.completed_at}>
                  {format(new Date(validation.completed_at), 'PPp')}
                </time>
              </div>
            )}
            <div>
              <span className="font-medium">Actualizada:</span>{' '}
              <time dateTime={validation.updated_at}>
                {format(new Date(validation.updated_at), 'PPp')}
              </time>
            </div>
          </div>
        </Collapsible>

        <Collapsible
          title="Plataformas Conectadas"
          defaultOpen
          icon={<CheckCircle2 className="h-4 w-4" aria-hidden="true" />}
        >
          {validation.connected_platforms.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              {validation.connected_platforms.map((platform) => (
                <Badge key={platform} variant="secondary">
                  {platform}
                </Badge>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No hay plataformas conectadas</p>
          )}
        </Collapsible>

        <Collapsible
          title="Estado del Proceso"
          defaultOpen={validation.status === 'running'}
          icon={getStatusIcon(validation.status)}
        >
          <div className="space-y-2 text-sm">
            <div className="flex items-center gap-2">
              <span className="font-medium">Estado:</span>
              <Badge variant={getStatusVariant(validation.status)}>
                {validation.status}
              </Badge>
            </div>
            <div className="flex items-center gap-2">
              <span className="font-medium">Tiene Perfil:</span>
              <Badge variant={validation.has_profile ? 'success' : 'default'}>
                {validation.has_profile ? 'Sí' : 'No'}
              </Badge>
            </div>
            <div className="flex items-center gap-2">
              <span className="font-medium">Tiene Reporte:</span>
              <Badge variant={validation.has_report ? 'success' : 'default'}>
                {validation.has_report ? 'Sí' : 'No'}
              </Badge>
            </div>
          </div>
        </Collapsible>
      </CardContent>
    </Card>
  );
};



