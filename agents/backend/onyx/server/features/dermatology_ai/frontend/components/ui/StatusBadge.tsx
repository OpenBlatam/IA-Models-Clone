'use client';

import React, { memo } from 'react';
import { Badge, BadgeVariant } from './Badge';
import { CheckCircle, XCircle, Clock, AlertCircle } from 'lucide-react';

export type StatusType = 'success' | 'error' | 'pending' | 'warning' | 'info';

interface StatusBadgeProps {
  status: StatusType;
  label?: string;
  showIcon?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

const getStatusIcon = (status: StatusType) => {
  switch (status) {
    case 'success':
      return <CheckCircle className="h-3 w-3" />;
    case 'error':
      return <XCircle className="h-3 w-3" />;
    case 'pending':
      return <Clock className="h-3 w-3" />;
    case 'warning':
    case 'info':
      return <AlertCircle className="h-3 w-3" />;
  }
};

const statusConfig: Record<StatusType, { variant: BadgeVariant; defaultLabel: string }> = {
  success: {
    variant: 'success',
    defaultLabel: 'Success',
  },
  error: {
    variant: 'danger',
    defaultLabel: 'Error',
  },
  pending: {
    variant: 'warning',
    defaultLabel: 'Pending',
  },
  warning: {
    variant: 'warning',
    defaultLabel: 'Warning',
  },
  info: {
    variant: 'info',
    defaultLabel: 'Info',
  },
};

export const StatusBadge: React.FC<StatusBadgeProps> = memo(({
  status,
  label,
  showIcon = true,
  size = 'sm',
}) => {
  const config = statusConfig[status];
  const icon = showIcon ? getStatusIcon(status) : null;
  
  return (
    <Badge variant={config.variant} size={size}>
      {icon && <span className="mr-1">{icon}</span>}
      {label || config.defaultLabel}
    </Badge>
  );
});

StatusBadge.displayName = 'StatusBadge';

