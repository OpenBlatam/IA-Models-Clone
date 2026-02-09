/**
 * Hook para notificaciones del sistema.
 */

import { useState, useCallback, useEffect } from 'react';
import { toast } from 'sonner';

export interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message?: string;
  duration?: number;
  timestamp: Date;
  read: boolean;
}

interface UseNotificationsOptions {
  maxNotifications?: number;
  autoRemove?: boolean;
  autoRemoveDelay?: number;
}

/**
 * Hook para gestionar notificaciones.
 */
export function useNotifications(options: UseNotificationsOptions = {}) {
  const {
    maxNotifications = 50,
    autoRemove = true,
    autoRemoveDelay = 5000
  } = options;

  const [notifications, setNotifications] = useState<Notification[]>([]);

  const addNotification = useCallback((
    type: Notification['type'],
    title: string,
    message?: string,
    duration?: number
  ) => {
    const notification: Notification = {
      id: `notif-${Date.now()}-${Math.random()}`,
      type,
      title,
      message,
      duration,
      timestamp: new Date(),
      read: false
    };

    setNotifications(prev => {
      const updated = [notification, ...prev];
      if (updated.length > maxNotifications) {
        updated.pop();
      }
      return updated;
    });

    // Mostrar toast
    switch (type) {
      case 'success':
        toast.success(title, { description: message, duration });
        break;
      case 'error':
        toast.error(title, { description: message, duration });
        break;
      case 'warning':
        toast.warning(title, { description: message, duration });
        break;
      case 'info':
        toast.info(title, { description: message, duration });
        break;
    }

    // Auto-remover si está habilitado
    if (autoRemove) {
      setTimeout(() => {
        removeNotification(notification.id);
      }, autoRemoveDelay);
    }
  }, [maxNotifications, autoRemove, autoRemoveDelay]);

  const removeNotification = useCallback((id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  }, []);

  const markAsRead = useCallback((id: string) => {
    setNotifications(prev =>
      prev.map(n => n.id === id ? { ...n, read: true } : n)
    );
  }, []);

  const markAllAsRead = useCallback(() => {
    setNotifications(prev => prev.map(n => ({ ...n, read: true })));
  }, []);

  const clearAll = useCallback(() => {
    setNotifications([]);
  }, []);

  const getUnreadCount = useCallback(() => {
    return notifications.filter(n => !n.read).length;
  }, [notifications]);

  return {
    notifications,
    addNotification,
    removeNotification,
    markAsRead,
    markAllAsRead,
    clearAll,
    getUnreadCount,
    unreadCount: getUnreadCount()
  };
}



