'use client';

import { memo, useCallback } from 'react';
import { AlertTriangle, Info, XCircle, AlertCircle } from 'lucide-react';
import { getAlertLevelColor, formatDate } from '@/lib/utils';
import type { Alert } from '../types';

interface AlertItemProps {
  alert: Alert;
}

const AlertItem = memo(({ alert }: AlertItemProps): JSX.Element => {
  const getAlertIcon = useCallback((level: string) => {
    switch (level) {
      case 'critical':
        return <XCircle className="w-5 h-5 text-red-600" />;
      case 'error':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-yellow-600" />;
      default:
        return <Info className="w-5 h-5 text-blue-600" />;
    }
  }, []);

  return (
    <div className={`p-3 rounded-lg border ${getAlertLevelColor(alert.level)}`}>
      <div className="flex items-start space-x-3">
        {getAlertIcon(alert.level)}
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-medium text-gray-600">
              {alert.level.toUpperCase()}
            </span>
            <span className="text-xs text-gray-500">
              {formatDate(alert.timestamp)}
            </span>
          </div>
          <p className="text-sm text-gray-900">{alert.message}</p>
          {alert.source && (
            <p className="text-xs text-gray-500 mt-1">Source: {alert.source}</p>
          )}
        </div>
      </div>
    </div>
  );
});

AlertItem.displayName = 'AlertItem';

export default AlertItem;

