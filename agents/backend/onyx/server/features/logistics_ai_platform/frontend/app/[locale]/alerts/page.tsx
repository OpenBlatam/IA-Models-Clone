'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useTranslations } from 'next-intl';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import Navbar from '@/components/layout/navbar';
import { alertsApi } from '@/lib/api/alerts';
import { formatDateTime } from '@/lib/utils';
import { getErrorMessage } from '@/lib/error-handler';
import Loading from '@/components/ui/loading';
import ErrorMessage from '@/components/ui/error-message';
import EmptyState from '@/components/ui/empty-state';

const AlertsPage = () => {
  const t = useTranslations('alerts');
  const queryClient = useQueryClient();

  const { data: alerts = [], isLoading, error } = useQuery({
    queryKey: ['alerts'],
    queryFn: () => alertsApi.getAll(),
  });

  const markAsReadMutation = useMutation({
    mutationFn: (alertId: string) => alertsApi.markAsRead(alertId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['alerts'] });
    },
  });

  const handleMarkAsRead = (alertId: string) => {
    markAsReadMutation.mutate(alertId);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <div className="mb-8">
            <h1 className="text-3xl font-bold">{t('title')}</h1>
            <p className="text-muted-foreground">Manage your alerts and notifications</p>
          </div>
          <Loading text="Loading alerts..." />
        </main>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <div className="mb-8">
            <h1 className="text-3xl font-bold">{t('title')}</h1>
            <p className="text-muted-foreground">Manage your alerts and notifications</p>
          </div>
          <ErrorMessage message={getErrorMessage(error)} />
        </main>
      </div>
    );
  }

  const unreadAlerts = alerts.filter((a) => !a.is_read);
  const readAlerts = alerts.filter((a) => a.is_read);

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold">{t('title')}</h1>
          <p className="text-muted-foreground">Manage your alerts and notifications</p>
        </div>

        {alerts.length === 0 ? (
          <EmptyState message={t('noAlerts')} />
        ) : (
          <div className="space-y-6">
            {unreadAlerts.length > 0 && (
              <div>
                <h2 className="mb-4 text-xl font-semibold">Unread Alerts ({unreadAlerts.length})</h2>
                <div className="space-y-4">
                  {unreadAlerts.map((alert) => {
                    const priorityClass =
                      alert.priority === 'critical'
                        ? 'bg-red-100 text-red-800'
                        : alert.priority === 'high'
                          ? 'bg-orange-100 text-orange-800'
                          : 'bg-blue-100 text-blue-800';

                    return (
                      <Card key={alert.alert_id} className="border-l-4 border-l-primary">
                        <CardHeader>
                          <div className="flex items-center justify-between">
                            <div>
                              <CardTitle>{alert.alert_type}</CardTitle>
                              <CardDescription>{formatDateTime(alert.created_at)}</CardDescription>
                            </div>
                            <span className={`rounded-full px-3 py-1 text-sm font-medium ${priorityClass}`}>
                              {alert.priority}
                            </span>
                          </div>
                        </CardHeader>
                        <CardContent>
                          <p className="mb-4">{alert.message}</p>
                          <Button
                            onClick={() => handleMarkAsRead(alert.alert_id)}
                            variant="outline"
                            disabled={markAsReadMutation.isPending}
                            aria-label={`${t('markAsRead')} ${alert.alert_id}`}
                          >
                            {markAsReadMutation.isPending ? t('common.loading') : t('markAsRead')}
                          </Button>
                        </CardContent>
                      </Card>
                    );
                  })}
                </div>
              </div>
            )}

            {readAlerts.length > 0 && (
              <div>
                <h2 className="mb-4 text-xl font-semibold">Read Alerts ({readAlerts.length})</h2>
                <div className="space-y-4">
                  {readAlerts.map((alert) => (
                    <Card key={alert.alert_id} className="opacity-75">
                      <CardHeader>
                        <div className="flex items-center justify-between">
                          <div>
                            <CardTitle>{alert.alert_type}</CardTitle>
                            <CardDescription>{formatDateTime(alert.created_at)}</CardDescription>
                          </div>
                          <span className="text-sm text-muted-foreground">Read</span>
                        </div>
                      </CardHeader>
                      <CardContent>
                        <p>{alert.message}</p>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
};

export default AlertsPage;
