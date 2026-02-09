/**
 * Connection status badge component
 */

import React from 'react';
import { Badge } from '@/components/ui';
import type { ConnectionStatus } from '@/lib/types';
import { CheckCircle2, XCircle, Clock, AlertCircle } from 'lucide-react';

export interface ConnectionStatusBadgeProps {
  status: ConnectionStatus;
  showIcon?: boolean;
}

const getStatusConfig = (status: ConnectionStatus) => {
  switch (status) {
    case 'connected':
      return {
        variant: 'success' as const,
        label: 'Conectado',
        icon: CheckCircle2,
      };
    case 'disconnected':
      return {
        variant: 'default' as const,
        label: 'Desconectado',
        icon: XCircle,
      };
    case 'expired':
      return {
        variant: 'warning' as const,
        label: 'Expirado',
        icon: Clock,
      };
    case 'error':
      return {
        variant: 'destructive' as const,
        label: 'Error',
        icon: AlertCircle,
      };
    default:
      return {
        variant: 'default' as const,
        label: status,
        icon: AlertCircle,
      };
  }
};

export const ConnectionStatusBadge: React.FC<ConnectionStatusBadgeProps> = ({
  status,
  showIcon = true,
}) => {
  const config = getStatusConfig(status);
  const Icon = config.icon;

  return (
    <Badge variant={config.variant} className="flex items-center gap-1">
      {showIcon && <Icon className="h-3 w-3" aria-hidden="true" />}
      <span>{config.label}</span>
    </Badge>
  );
};



