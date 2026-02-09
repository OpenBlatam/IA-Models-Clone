'use client';

import { AlertTriangle, Info, XCircle, AlertCircle } from 'lucide-react';
import { useQualityControlStore } from '@/lib/store';
import { getAlertLevelColor, formatDate } from '@/lib/utils';

const AlertsPanel = (): JSX.Element => {
  const { alerts } = useQualityControlStore();

  const getAlertIcon = (level: string) => {
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
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">Alerts</h2>
      {alerts.length === 0 ? (
        <div className="text-center py-8 text-gray-500">
          <Info className="w-12 h-12 mx-auto mb-4 text-gray-400" />
          <p>No alerts</p>
        </div>
      ) : (
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {alerts.slice(0, 10).map((alert, index) => (
            <div
              key={index}
              className={`p-3 rounded-lg border ${getAlertLevelColor(
                alert.level
              )}`}
            >
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
                    <p className="text-xs text-gray-500 mt-1">
                      Source: {alert.source}
                    </p>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default AlertsPanel;

