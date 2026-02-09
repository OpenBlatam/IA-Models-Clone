/**
 * Validation status badge with indicator
 */

import React from 'react';
import { Badge, StatusIndicator } from '@/components/ui';
import type { ValidationStatus } from '@/lib/types';

export interface ValidationStatusBadgeProps {
  status: ValidationStatus;
  showIndicator?: boolean;
  className?: string;
}

const statusConfig: Record<
  ValidationStatus,
  { label: string; variant: 'success' | 'destructive' | 'warning' | 'info' | 'default'; indicator: 'online' | 'offline' | 'away' | 'busy' }
> = {
  completed: {
    label: 'Completada',
    variant: 'success',
    indicator: 'online',
  },
  failed: {
    label: 'Fallida',
    variant: 'destructive',
    indicator: 'busy',
  },
  running: {
    label: 'En Proceso',
    variant: 'info',
    indicator: 'away',
  },
  pending: {
    label: 'Pendiente',
    variant: 'default',
    indicator: 'offline',
  },
  cancelled: {
    label: 'Cancelada',
    variant: 'warning',
    indicator: 'offline',
  },
};

export const ValidationStatusBadge: React.FC<ValidationStatusBadgeProps> = ({
  status,
  showIndicator = true,
  className,
}) => {
  const config = statusConfig[status];

  return (
    <Badge variant={config.variant} className={`flex items-center gap-2 ${className || ''}`}>
      {showIndicator && (
        <StatusIndicator
          status={config.indicator}
          size="sm"
          pulse={status === 'running'}
        />
      )}
      <span>{config.label}</span>
    </Badge>
  );
};



