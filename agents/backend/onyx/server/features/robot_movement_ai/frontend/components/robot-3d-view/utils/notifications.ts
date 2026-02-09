/**
 * Notification utilities for Robot 3D View
 * @module robot-3d-view/utils/notifications
 */

/**
 * Notification types
 */
export type NotificationType = 'info' | 'success' | 'warning' | 'error';

/**
 * Notification data
 */
export interface Notification {
  id: string;
  type: NotificationType;
  message: string;
  title?: string;
  duration?: number;
  timestamp: number;
}

/**
 * Notification manager class
 */
class NotificationManager {
  private notifications: Notification[] = [];
  private listeners: Set<(notifications: Notification[]) => void> = new Set();
  private defaultDuration = 3000;

  /**
   * Adds a notification
   */
  add(notification: Omit<Notification, 'id' | 'timestamp'>): string {
    const id = `notification-${Date.now()}-${Math.random()}`;
    const fullNotification: Notification = {
      ...notification,
      id,
      timestamp: Date.now(),
      duration: notification.duration ?? this.defaultDuration,
    };

    this.notifications.push(fullNotification);
    this.notifyListeners();

    // Auto-remove after duration
    if (fullNotification.duration > 0) {
      setTimeout(() => {
        this.remove(id);
      }, fullNotification.duration);
    }

    return id;
  }

  /**
   * Removes a notification
   */
  remove(id: string): void {
    this.notifications = this.notifications.filter((n) => n.id !== id);
    this.notifyListeners();
  }

  /**
   * Clears all notifications
   */
  clear(): void {
    this.notifications = [];
    this.notifyListeners();
  }

  /**
   * Gets all notifications
   */
  getAll(): Notification[] {
    return [...this.notifications];
  }

  /**
   * Subscribes to notification changes
   */
  subscribe(listener: (notifications: Notification[]) => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  /**
   * Notifies all listeners
   */
  private notifyListeners(): void {
    this.listeners.forEach((listener) => {
      listener([...this.notifications]);
    });
  }
}

/**
 * Global notification manager instance
 */
export const notificationManager = new NotificationManager();

/**
 * Helper functions for creating notifications
 */
export const notify = {
  info: (message: string, title?: string) =>
    notificationManager.add({ type: 'info', message, title }),
  success: (message: string, title?: string) =>
    notificationManager.add({ type: 'success', message, title }),
  warning: (message: string, title?: string) =>
    notificationManager.add({ type: 'warning', message, title }),
  error: (message: string, title?: string) =>
    notificationManager.add({ type: 'error', message, title }),
};



