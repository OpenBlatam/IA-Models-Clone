'use client';

import React from 'react';
import { apiClient } from '@/lib/api/client';
import { Alert, AlertsResponse } from '@/lib/types/api';
import { Loading } from '@/components/ui/Loading';
import { Bell, CheckCircle } from 'lucide-react';
import { useAuth } from '@/lib/contexts/AuthContext';
import { AlertCard } from '@/components/alerts/AlertCard';
import { ProtectedPage } from '@/components/auth/ProtectedPage';
import { EmptyState } from '@/lib/utils/emptyStates';
import { useAsyncData } from '@/lib/hooks/useAsyncData';
import { useMutation } from '@/lib/hooks/useMutation';
import { PageLayout } from '@/components/layout/PageLayout';
import { PageHeader } from '@/components/layout/PageHeader';

interface AlertsData {
  alerts: Alert[];
  unreadCount: number;
}

export default function AlertsPage() {
  const { user } = useAuth();

  const { data, isLoading, refetch } = useAsyncData<AlertsData>({
    fetchFn: async () => {
      if (!user) throw new Error('User not authenticated');
      const [alertsResponse, summaryResponse] = await Promise.all([
        apiClient.getAlerts(user.id),
        apiClient.getAlertsSummary(user.id),
      ]);
      const unread = summaryResponse.unread_count || 0;
      return {
        alerts: alertsResponse.alerts || [],
        unreadCount: unread,
      };
    },
    enabled: !!user,
    errorMessage: 'Failed to load alerts. Please try again.',
  });

  const acknowledgeMutation = useMutation({
    mutationFn: async (alertId: string) => {
      if (!user) throw new Error('User not authenticated');
      return await apiClient.acknowledgeAlert(user.id, alertId);
    },
    onSuccess: () => {
      refetch();
    },
    successMessage: 'Alert marked as read',
    errorMessage: 'Failed to mark alert. Please try again.',
  });

  const handleAcknowledge = (alertId: string) => {
    acknowledgeMutation.mutate(alertId);
  };

  const alerts = data?.alerts || [];

  return (
    <ProtectedPage message="Sign in to view your alerts">
      <PageLayout>
        <PageHeader
          title="Alerts"
          description="Stay informed about your skin health"
          icon={Bell}
          badge={
            data?.unreadCount && data.unreadCount > 0 ? (
              <div className="px-3 py-1 bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 rounded-full text-sm font-medium">
                {data.unreadCount} unread
              </div>
            ) : undefined
          }
        />

        {isLoading ? (
          <Loading fullScreen text="Loading alerts..." />
        ) : alerts.length === 0 ? (
          <EmptyState
            icon={<CheckCircle className="h-16 w-16 text-green-500" />}
            title="No alerts"
            description="All clear. No pending alerts."
          />
        ) : (
          <div className="space-y-4">
            {alerts.map((alert) => (
              <AlertCard
                key={alert.alert_id}
                alert={alert}
                onAcknowledge={handleAcknowledge}
                isAcknowledging={acknowledgeMutation.isLoading}
              />
            ))}
          </div>
        )}
      </PageLayout>
    </ProtectedPage>
  );
}

