import { memo, useState, useEffect } from 'react';
import { useEventListener } from '@/lib/hooks';
import Card from './Card';
import Button from './Button';
import { X, Bell } from 'lucide-react';
import { cn } from '@/lib/utils';

interface Notification {
  id: string;
  title: string;
  message: string;
  type?: 'info' | 'success' | 'warning' | 'error';
  duration?: number;
  action?: {
    label: string;
    onClick: () => void;
  };
}

interface NotificationCenterProps {
  className?: string;
  maxNotifications?: number;
}

const NotificationCenter = memo(({
  className = '',
  maxNotifications = 5,
}: NotificationCenterProps): JSX.Element => {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [isOpen, setIsOpen] = useState(false);

  const addNotification = (notification: Omit<Notification, 'id'>): void => {
    const id = Math.random().toString(36).substr(2, 9);
    const newNotification: Notification = {
      id,
      duration: 5000,
      ...notification,
    };

    setNotifications((prev) => {
      const updated = [newNotification, ...prev];
      return updated.slice(0, maxNotifications);
    });

    if (newNotification.duration && newNotification.duration > 0) {
      setTimeout(() => {
        removeNotification(id);
      }, newNotification.duration);
    }
  };

  const removeNotification = (id: string): void => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  };

  const clearAll = (): void => {
    setNotifications([]);
  };

  useEffect(() => {
    if (typeof window !== 'undefined') {
      (window as any).addNotification = addNotification;
    }
  }, []);

  if (notifications.length === 0) {
    return null;
  }

  return (
    <div className={cn('fixed top-4 right-4 z-50 space-y-2', className)}>
      <div className="flex items-center justify-between mb-2">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="p-2 rounded-full hover:bg-gray-100"
          aria-label="Toggle notifications"
        >
          <Bell className="w-5 h-5" />
          {notifications.length > 0 && (
            <span className="absolute top-0 right-0 w-4 h-4 bg-red-500 text-white text-xs rounded-full flex items-center justify-center">
              {notifications.length}
            </span>
          )}
        </button>
        {isOpen && (
          <Button onClick={clearAll} variant="secondary" size="sm">
            Clear All
          </Button>
        )}
      </div>

      {isOpen && (
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {notifications.map((notification) => (
            <Card
              key={notification.id}
              className={cn(
                'min-w-80 p-4',
                notification.type === 'error' && 'border-red-500',
                notification.type === 'warning' && 'border-yellow-500',
                notification.type === 'success' && 'border-green-500',
                notification.type === 'info' && 'border-blue-500'
              )}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1">
                  <h4 className="font-semibold">{notification.title}</h4>
                  <p className="text-sm text-gray-600 mt-1">{notification.message}</p>
                  {notification.action && (
                    <Button
                      onClick={notification.action.onClick}
                      variant="secondary"
                      size="sm"
                      className="mt-2"
                    >
                      {notification.action.label}
                    </Button>
                  )}
                </div>
                <button
                  onClick={() => removeNotification(notification.id)}
                  className="p-1 hover:bg-gray-100 rounded"
                  aria-label="Close notification"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
});

NotificationCenter.displayName = 'NotificationCenter';

export default NotificationCenter;



