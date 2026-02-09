'use client';

import { useQuery, useMutation, useQueryClient } from 'react-query';
import { apiClient } from '@/lib/api/client';
import PageLayout from '@/components/Layout/PageLayout';
import Card from '@/components/UI/Card';
import Button from '@/components/UI/Button';
import LoadingSpinner from '@/components/UI/LoadingSpinner';
import EmptyState from '@/components/UI/EmptyState';
import { formatDate } from '@/lib/utils';

interface Notification {
  notification_id: string;
  type: string;
  message: string;
  read: boolean;
  created_at: string;
}

const NotificationsPage = (): JSX.Element => {
  const queryClient = useQueryClient();

  const { data: notifications, isLoading } = useQuery<Notification[]>('notifications', () =>
    apiClient.getNotifications()
  );

  const markReadMutation = useMutation((id: string) => apiClient.markNotificationRead(id), {
    onSuccess: () => {
      queryClient.invalidateQueries('notifications');
    },
  });

  const markAllReadMutation = useMutation(() => apiClient.markAllNotificationsRead(), {
    onSuccess: () => {
      queryClient.invalidateQueries('notifications');
    },
  });

  const handleMarkRead = (id: string): void => {
    markReadMutation.mutate(id);
  };

  const handleMarkAllRead = (): void => {
    markAllReadMutation.mutate();
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
          <h1 className="text-3xl font-bold">Notifications</h1>
          {notifications && notifications.length > 0 && (
            <Button onClick={handleMarkAllRead} isLoading={markAllReadMutation.isLoading}>
              Mark All Read
            </Button>
          )}
        </div>

        {!notifications || notifications.length === 0 ? (
          <Card>
            <EmptyState title="No notifications" description="You're all caught up!" />
          </Card>
        ) : (
          <div className="space-y-4">
            {notifications.map((notification) => (
              <Card
                key={notification.notification_id}
                className={notification.read ? 'opacity-75' : ''}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="font-semibold">{notification.type}</span>
                      {!notification.read && (
                        <span className="px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs">
                          New
                        </span>
                      )}
                    </div>
                    <p className="mb-2">{notification.message}</p>
                    <p className="text-xs text-gray-500">{formatDate(notification.created_at)}</p>
                  </div>
                  {!notification.read && (
                    <Button
                      variant="secondary"
                      onClick={() => handleMarkRead(notification.notification_id)}
                      className="text-sm"
                    >
                      Mark Read
                    </Button>
                  )}
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>
    </PageLayout>
  );
};

export default NotificationsPage;



