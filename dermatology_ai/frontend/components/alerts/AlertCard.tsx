'use client';

import React, { memo } from 'react';
import { Card, CardContent } from '../ui/Card';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import { Alert } from '@/lib/types/api';
import { AlertCircle, XCircle, Info, Bell, CheckCircle } from 'lucide-react';
import { formatAlertDate } from '@/lib/utils/dateUtils';
import { clsx } from 'clsx';

interface AlertCardProps {
  alert: Alert;
  onAcknowledge?: (alertId: string) => void;
  isAcknowledging?: boolean;
}

const getAlertIcon = (type: string): React.ReactNode => {
  switch (type) {
    case 'condition_detected':
      return <AlertCircle className="h-5 w-5 text-red-600 dark:text-red-400" />;
    case 'score_drop':
      return <XCircle className="h-5 w-5 text-yellow-600 dark:text-yellow-400" />;
    case 'recommendation':
      return <Info className="h-5 w-5 text-blue-600 dark:text-blue-400" />;
    case 'reminder':
      return <Bell className="h-5 w-5 text-purple-600 dark:text-purple-400" />;
    default:
      return <Info className="h-5 w-5 text-gray-600 dark:text-gray-400" />;
  }
};

const getSeverityVariant = (severity: string): 'danger' | 'warning' | 'info' | 'default' => {
  switch (severity) {
    case 'critical':
    case 'high':
      return 'danger';
    case 'medium':
      return 'warning';
    default:
      return 'info';
  }
};

const getTypeLabel = (type: string) => {
  switch (type) {
    case 'condition_detected':
      return 'Condition';
    case 'score_drop':
      return 'Score Drop';
    case 'recommendation':
      return 'Recommendation';
    case 'reminder':
      return 'Reminder';
    default:
      return type;
  }
};

export const AlertCard: React.FC<AlertCardProps> = memo(({
  alert,
  onAcknowledge,
  isAcknowledging = false,
}) => {
  const alertIcon = getAlertIcon(alert.type);
  const handleAcknowledge = onAcknowledge
    ? () => onAcknowledge(alert.alert_id)
    : undefined;

  return (
    <Card
      className={clsx(
        'transition-all',
        !alert.acknowledged
          ? 'border-l-4 border-primary-500 bg-primary-50 dark:bg-primary-900/20'
          : 'opacity-75'
      )}
    >
      <CardContent className="p-6">
        <div className="flex items-start justify-between">
          <div className="flex items-start space-x-4 flex-1">
            <div className="mt-1">{alertIcon}</div>
            <div className="flex-1">
              <div className="flex items-center space-x-2 mb-2">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  {alert.title}
                </h3>
                <Badge variant={getSeverityVariant(alert.severity)} size="sm">
                  {alert.severity}
                </Badge>
                <Badge variant="default" size="sm">
                  {getTypeLabel(alert.type)}
                </Badge>
              </div>
              <p className="text-gray-700 dark:text-gray-300 mb-3">
                {alert.message}
              </p>
              {alert.action_required && (
                <div className="mb-3 p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded">
                  <p className="text-sm font-medium text-yellow-800 dark:text-yellow-200">
                    Action:
                  </p>
                  <p className="text-sm text-yellow-700 dark:text-yellow-300 mt-1">
                    {alert.action_required}
                  </p>
                </div>
              )}
              <p className="text-sm text-gray-500 dark:text-gray-400">
                {formatAlertDate(alert.timestamp)}
              </p>
            </div>
          </div>
          {!alert.acknowledged && handleAcknowledge && (
            <Button
              size="sm"
              variant="outline"
              onClick={handleAcknowledge}
              disabled={isAcknowledging}
              className="ml-4"
            >
              Mark as read
            </Button>
          )}
          {alert.acknowledged && (
            <div className="ml-4 flex items-center text-green-600 dark:text-green-400">
              <CheckCircle className="h-5 w-5 mr-1" />
              <span className="text-sm">Read</span>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
});

AlertCard.displayName = 'AlertCard';

