/**
 * Notification Toast Component
 * @module robot-3d-view/components/notification-toast
 */

'use client';

import { memo } from 'react';
import { useNotifications } from '../hooks/use-notifications';
import type { Notification } from '../utils/notifications';

/**
 * Notification Toast Component
 * 
 * Displays notifications in a toast-style overlay.
 * 
 * @returns Notification toast component
 */
export const NotificationToast = memo(() => {
  const { notifications, remove } = useNotifications();

  if (notifications.length === 0) {
    return null;
  }

  const getNotificationStyles = (type: Notification['type']) => {
    const styles = {
      info: 'bg-blue-500/90 border-blue-400',
      success: 'bg-green-500/90 border-green-400',
      warning: 'bg-yellow-500/90 border-yellow-400',
      error: 'bg-red-500/90 border-red-400',
    };
    return styles[type] || styles.info;
  };

  const getIcon = (type: Notification['type']) => {
    const icons = {
      info: 'ℹ️',
      success: '✅',
      warning: '⚠️',
      error: '❌',
    };
    return icons[type] || icons.info;
  };

  return (
    <div className="fixed top-4 left-1/2 -translate-x-1/2 z-50 flex flex-col gap-2 max-w-md w-full px-4">
      {notifications.map((notification) => (
        <div
          key={notification.id}
          className={`
            ${getNotificationStyles(notification.type)}
            backdrop-blur-md border rounded-lg p-3 shadow-lg
            text-white text-sm animate-in slide-in-from-top
            flex items-start gap-2
          `}
          role="alert"
          aria-live="polite"
        >
          <span className="text-lg flex-shrink-0">{getIcon(notification.type)}</span>
          <div className="flex-1 min-w-0">
            {notification.title && (
              <div className="font-semibold mb-1">{notification.title}</div>
            )}
            <div>{notification.message}</div>
          </div>
          <button
            onClick={() => remove(notification.id)}
            className="flex-shrink-0 text-white/80 hover:text-white transition-colors"
            aria-label="Cerrar notificación"
          >
            ✕
          </button>
        </div>
      ))}
    </div>
  );
});

NotificationToast.displayName = 'NotificationToast';



