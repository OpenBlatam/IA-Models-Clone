import { memo, useMemo } from 'react';
import { cn } from '@/lib/utils';

type StatusVariant = 'success' | 'error' | 'warning' | 'info' | 'pending';

interface StatusBadgeProps {
  status: string;
  variant?: StatusVariant;
  className?: string;
}

const VARIANT_CLASSES: Record<StatusVariant, string> = {
  success: 'bg-green-100 text-green-700',
  error: 'bg-red-100 text-red-700',
  warning: 'bg-yellow-100 text-yellow-700',
  info: 'bg-blue-100 text-blue-700',
  pending: 'bg-gray-100 text-gray-700',
};

const getVariantFromStatus = (status: string): StatusVariant => {
  const normalizedStatus = status.toLowerCase();
  
  if (normalizedStatus.includes('success') || normalizedStatus.includes('completed')) {
    return 'success';
  }
  if (normalizedStatus.includes('error') || normalizedStatus.includes('failed')) {
    return 'error';
  }
  if (normalizedStatus.includes('warning')) {
    return 'warning';
  }
  if (normalizedStatus.includes('pending') || normalizedStatus.includes('processing')) {
    return 'pending';
  }
  
  return 'info';
};

const StatusBadge = memo(({ status, variant, className = '' }: StatusBadgeProps): JSX.Element => {
  const badgeVariant = useMemo(() => variant || getVariantFromStatus(status), [status, variant]);
  const variantClass = VARIANT_CLASSES[badgeVariant];

  return (
    <span
      className={cn('px-3 py-1 rounded-full text-sm font-medium', variantClass, className)}
      aria-label={`Status: ${status}`}
    >
      {status}
    </span>
  );
});

StatusBadge.displayName = 'StatusBadge';

export default StatusBadge;
