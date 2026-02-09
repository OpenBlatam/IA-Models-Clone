'use client';

import { memo, useMemo } from 'react';
import { CheckCircle, XCircle, AlertTriangle } from 'lucide-react';
import { getStatusLabel } from '@/lib/utils';
import { cn } from '@/lib/utils';

interface StatusBadgeProps {
  status: string;
  recommendation: string;
}

const StatusBadge = memo(({ status, recommendation }: StatusBadgeProps): JSX.Element => {
  const config = useMemo(() => {
    switch (status) {
      case 'excellent':
      case 'good':
        return {
          bg: 'bg-green-50',
          border: 'border-green-500',
          icon: <CheckCircle className="w-5 h-5 text-green-600 mt-0.5" aria-hidden="true" />,
          ariaLabel: 'Status: Good quality',
        };
      case 'acceptable':
        return {
          bg: 'bg-yellow-50',
          border: 'border-yellow-500',
          icon: <AlertTriangle className="w-5 h-5 text-yellow-600 mt-0.5" aria-hidden="true" />,
          ariaLabel: 'Status: Acceptable quality',
        };
      default:
        return {
          bg: 'bg-red-50',
          border: 'border-red-500',
          icon: <XCircle className="w-5 h-5 text-red-600 mt-0.5" aria-hidden="true" />,
          ariaLabel: 'Status: Poor quality',
        };
    }
  }, [status]);

  return (
    <div
      className={cn('p-4 rounded-lg border-l-4', config.bg, config.border)}
      role="status"
      aria-label={config.ariaLabel}
    >
      <div className="flex items-start space-x-3">
        {config.icon}
        <div className="flex-1">
          <h3 className="font-semibold text-gray-900 mb-1">Recommendation</h3>
          <p className="text-gray-700">{recommendation}</p>
        </div>
      </div>
    </div>
  );
});

StatusBadge.displayName = 'StatusBadge';

export default StatusBadge;

