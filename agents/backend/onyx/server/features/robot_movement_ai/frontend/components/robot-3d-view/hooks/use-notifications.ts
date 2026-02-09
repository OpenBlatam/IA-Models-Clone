/**
 * Hook for notification management
 * @module robot-3d-view/hooks/use-notifications
 */

import { useState, useEffect } from 'react';
import { notificationManager, type Notification } from '../utils/notifications';

/**
 * Hook for managing notifications
 * 
 * @returns Notifications state and actions
 * 
 * @example
 * ```tsx
 * const { notifications, remove, clear } = useNotifications();
 * ```
 */
export function useNotifications() {
  const [notifications, setNotifications] = useState<Notification[]>(() =>
    notificationManager.getAll()
  );

  useEffect(() => {
    const unsubscribe = notificationManager.subscribe((newNotifications) => {
      setNotifications(newNotifications);
    });

    return unsubscribe;
  }, []);

  const remove = (id: string) => {
    notificationManager.remove(id);
  };

  const clear = () => {
    notificationManager.clear();
  };

  return {
    notifications,
    remove,
    clear,
  };
}



