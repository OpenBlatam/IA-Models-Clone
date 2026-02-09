'use client';

import { useQuery, useMutation, useQueryClient } from 'react-query';
import { apiClient } from '@/lib/api/client';
import { AlertSeverity, type AlertsResponse } from '@/types';
import PageLayout from '@/components/Layout/PageLayout';
import Card from '@/components/UI/Card';
import Button from '@/components/UI/Button';
import LoadingSpinner from '@/components/UI/LoadingSpinner';
import EmptyState from '@/components/UI/EmptyState';
import { formatDate, cn } from '@/lib/utils';

const SEVERITY_COLORS: Record<AlertSeverity, string> = {
  [AlertSeverity.CRITICAL]: 'bg-red-100 text-red-700 border-red-300',
  [AlertSeverity.ERROR]: 'bg-orange-100 text-orange-700 border-orange-300',
  [AlertSeverity.WARNING]: 'bg-yellow-100 text-yellow-700 border-yellow-300',
  [AlertSeverity.INFO]: 'bg-blue-100 text-blue-700 border-blue-300',
};

const AlertsPage = (): JSX.Element => {
  const queryClient = useQueryClient();
  const { data: alertsData, isLoading } = useQuery<AlertsResponse>('alerts', () =>
    apiClient.getAlerts()
  );

  const acknowledgeMutation = useMutation((alertId: string) => apiClient.acknowledgeAlert(alertId), {
    onSuccess: () => {
      queryClient.invalidateQueries('alerts');
    },
  });

  const resolveMutation = useMutation((alertId: string) => apiClient.resolveAlert(alertId), {
    onSuccess: () => {
      queryClient.invalidateQueries('alerts');
    },
  });

  const getSeverityColor = (severity: AlertSeverity): string => {
    return SEVERITY_COLORS[severity] || SEVERITY_COLORS[AlertSeverity.INFO];
  };

  const handleAcknowledge = (alertId: string): void => {
    acknowledgeMutation.mutate(alertId);
  };

  const handleResolve = (alertId: string): void => {
    resolveMutation.mutate(alertId);
  };

  if (isLoading) {
    return (
      <PageLayout>
        <LoadingSpinner />
      </PageLayout>
    );
  }

  return (
    <PageLayout>
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold">Alerts</h1>
          {alertsData && (
            <div className="flex gap-4">
              <div className="text-right">
                <p className="text-sm text-gray-600">Total</p>
                <p className="text-2xl font-bold">{alertsData.count}</p>
              </div>
              <div className="text-right">
                <p className="text-sm text-gray-600">Critical</p>
                <p className="text-2xl font-bold text-red-600">{alertsData.critical_count}</p>
              </div>
            </div>
          )}
        </div>

        {!alertsData || alertsData.alerts.length === 0 ? (
          <Card>
            <EmptyState title="No alerts" description="All systems are running smoothly" />
          </Card>
        ) : (
          <div className="space-y-4">
            {alertsData.alerts.map((alert) => {
              const severityColor = getSeverityColor(alert.severity);
              const opacityClass = alert.acknowledged ? 'opacity-75' : 'opacity-100';

              return (
                <Card
                  key={alert.alert_id}
                  className={cn('border-2', severityColor, opacityClass)}
                  role="alert"
                  aria-live="polite"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="font-semibold capitalize" aria-label={`Severity: ${alert.severity}`}>
                          {alert.severity}
                        </span>
                        <span className="text-sm text-gray-600" aria-label={`Type: ${alert.type}`}>
                          - {alert.type}
                        </span>
                        {alert.acknowledged && (
                          <span
                            className="px-2 py-1 bg-gray-200 text-gray-700 rounded text-xs"
                            aria-label="Alert acknowledged"
                          >
                            Acknowledged
                          </span>
                        )}
                      </div>
                      <p className="mb-2">{alert.message}</p>
                      <p className="text-xs text-gray-600" aria-label={`Created: ${formatDate(alert.created_at)}`}>
                        {formatDate(alert.created_at)}
                      </p>
                    </div>
                    <div className="flex gap-2 ml-4">
                      {!alert.acknowledged && (
                        <Button
                          variant="secondary"
                          onClick={() => handleAcknowledge(alert.alert_id)}
                          className="text-sm"
                          aria-label={`Acknowledge alert ${alert.alert_id}`}
                        >
                          Acknowledge
                        </Button>
                      )}
                      <Button
                        variant="secondary"
                        onClick={() => handleResolve(alert.alert_id)}
                        className="text-sm"
                        aria-label={`Resolve alert ${alert.alert_id}`}
                      >
                        Resolve
                      </Button>
                    </div>
                  </div>
                </Card>
              );
            })}
          </div>
        )}
      </div>
    </PageLayout>
  );
};

export default AlertsPage;

