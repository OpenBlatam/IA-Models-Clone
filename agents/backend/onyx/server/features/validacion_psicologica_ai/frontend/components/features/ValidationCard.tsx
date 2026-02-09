/**
 * Card component for individual validation with improved design
 */

'use client';

import React from 'react';
import Link from 'next/link';
import { Card, CardHeader, CardTitle, CardContent, Badge, Progress } from '@/components/ui';
import { format } from 'date-fns';
import type { ValidationRead } from '@/lib/types';
import { CheckCircle2, XCircle, Clock, Play, ExternalLink } from 'lucide-react';
import { cn } from '@/lib/utils/cn';

export interface ValidationCardProps {
  validation: ValidationRead;
}

const getStatusConfig = (status: ValidationRead['status']) => {
  switch (status) {
    case 'completed':
      return {
        icon: CheckCircle2,
        variant: 'success' as const,
        label: 'Completada',
        progress: 100,
      };
    case 'failed':
      return {
        icon: XCircle,
        variant: 'destructive' as const,
        label: 'Fallida',
        progress: 0,
      };
    case 'running':
      return {
        icon: Play,
        variant: 'info' as const,
        label: 'En Proceso',
        progress: 50,
      };
    case 'cancelled':
      return {
        icon: XCircle,
        variant: 'warning' as const,
        label: 'Cancelada',
        progress: 0,
      };
    default:
      return {
        icon: Clock,
        variant: 'default' as const,
        label: 'Pendiente',
        progress: 0,
      };
  }
};

export const ValidationCard: React.FC<ValidationCardProps> = ({ validation }) => {
  const statusConfig = getStatusConfig(validation.status);
  const Icon = statusConfig.icon;

  const handleKeyDown = (event: React.KeyboardEvent<HTMLAnchorElement>) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      window.location.href = `/validations/${validation.id}`;
    }
  };

  return (
    <Link
      href={`/validations/${validation.id}`}
      className="block"
      aria-label={`Ver validación ${validation.id.slice(0, 8)}`}
      onKeyDown={handleKeyDown}
      tabIndex={0}
    >
      <Card
        className={cn(
          'hover:shadow-lg transition-all duration-200 cursor-pointer',
          validation.status === 'completed' && 'border-green-200',
          validation.status === 'failed' && 'border-red-200',
          validation.status === 'running' && 'border-blue-200 animate-pulse'
        )}
      >
        <CardHeader>
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <CardTitle className="text-lg mb-2">
                Validación #{validation.id.slice(0, 8)}
              </CardTitle>
              <div className="flex items-center gap-2">
                <Icon
                  className={cn(
                    'h-4 w-4',
                    validation.status === 'completed' && 'text-green-600',
                    validation.status === 'failed' && 'text-red-600',
                    validation.status === 'running' && 'text-blue-600 animate-pulse'
                  )}
                  aria-hidden="true"
                />
                <Badge variant={statusConfig.variant}>{statusConfig.label}</Badge>
              </div>
            </div>
            <ExternalLink className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {validation.status === 'running' && (
              <Progress value={statusConfig.progress} label="Progreso" showValue />
            )}

            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2 text-muted-foreground">
                <time dateTime={validation.created_at}>
                  Creada: {format(new Date(validation.created_at), 'PPp')}
                </time>
              </div>
              {validation.completed_at && (
                <div className="flex items-center gap-2 text-muted-foreground">
                  <time dateTime={validation.completed_at}>
                    Completada: {format(new Date(validation.completed_at), 'PPp')}
                  </time>
                </div>
              )}
            </div>

            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-muted-foreground">Plataformas:</span>
              <div className="flex gap-1 flex-wrap" role="list" aria-label="Plataformas conectadas">
                {validation.connected_platforms.slice(0, 3).map((platform) => (
                  <Badge key={platform} variant="secondary" className="text-xs" role="listitem">
                    {platform}
                  </Badge>
                ))}
                {validation.connected_platforms.length > 3 && (
                  <Badge variant="secondary" className="text-xs">
                    +{validation.connected_platforms.length - 3}
                  </Badge>
                )}
              </div>
            </div>

            <div className="flex items-center gap-4 text-xs text-muted-foreground">
              {validation.has_profile && (
                <span className="flex items-center gap-1">
                  <CheckCircle2 className="h-3 w-3" aria-hidden="true" />
                  Perfil
                </span>
              )}
              {validation.has_report && (
                <span className="flex items-center gap-1">
                  <CheckCircle2 className="h-3 w-3" aria-hidden="true" />
                  Reporte
                </span>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </Link>
  );
};




