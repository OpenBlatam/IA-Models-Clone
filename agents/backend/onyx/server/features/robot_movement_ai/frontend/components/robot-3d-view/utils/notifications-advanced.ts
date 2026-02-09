/**
 * Advanced notifications system
 * @module robot-3d-view/utils/notifications-advanced
 */

/**
 * Notification type
 */
export type NotificationType = 'info' | 'success' | 'warning' | 'error' | 'loading';

/**
 * Notification position
 */
export type NotificationPosition =
  | 'top-left'
  | 'top-center'
  | 'top-right'
  | 'bottom-left'
  | 'bottom-center'
  | 'bottom-right';

/**
 * Notification action
 */
export interface NotificationAction {
  label: string;
  onClick: () => void;
  variant?: 'primary' | 'secondary';
}

/**
 * Notification options
 */
export interface NotificationOptions {
  id?: string;
  title?: string;
  message: string;
  type?: NotificationType;
  duration?: number;
  position?: NotificationPosition;
  actions?: NotificationAction[];
  dismissible?: boolean;
  onClose?: () => void;
  persistent?: boolean;
}

/**
 * Notification
 */
export interface Notification extends NotificationOptions {
  id: string;
  timestamp: number;
}

/**
 * Advanced Notifications Manager class
 */
export class AdvancedNotificationsManager {
  private notifications: Map<string, Notification> = new Map();
  private maxNotifications = 10;
  private defaultDuration = 5000;
  private defaultPosition: NotificationPosition = 'top-right';
  private listeners: Set<(notifications: Notification[]) => void> = new Set();

  /**
   * Shows a notification
   */
  show(options: NotificationOptions): string {
    const id = options.id || `notification-${Date.now()}-${Math.random()}`;
    const notification: Notification = {
      ...options,
      id,
      timestamp: Date.now(),
      type: options.type || 'info',
      duration: options.duration ?? this.defaultDuration,
      position: options.position || this.defaultPosition,
      dismissible: options.dismissible !== false,
      persistent: options.persistent || false,
    };

    this.notifications.set(id, notification);

    // Limit notifications
    if (this.notifications.size > this.maxNotifications) {
      const oldest = Array.from(this.notifications.values())
        .sort((a, b) => a.timestamp - b.timestamp)[0];
      this.notifications.delete(oldest.id);
    }

    // Auto-dismiss
    if (!notification.persistent && notification.duration > 0) {
      setTimeout(() => {
        this.dismiss(id);
      }, notification.duration);
    }

    this.notifyListeners();
    return id;
  }

  /**
   * Dismisses a notification
   */
  dismiss(id: string): void {
    const notification = this.notifications.get(id);
    if (notification) {
      notification.onClose?.();
      this.notifications.delete(id);
      this.notifyListeners();
    }
  }

  /**
   * Dismisses all notifications
   */
  dismissAll(): void {
    this.notifications.forEach((notification) => {
      notification.onClose?.();
    });
    this.notifications.clear();
    this.notifyListeners();
  }

  /**
   * Updates a notification
   */
  update(id: string, updates: Partial<NotificationOptions>): boolean {
    const notification = this.notifications.get(id);
    if (!notification) return false;

    Object.assign(notification, updates);
    this.notifyListeners();
    return true;
  }

  /**
   * Gets all notifications
   */
  getAll(): Notification[] {
    return Array.from(this.notifications.values());
  }

  /**
   * Gets notifications by position
   */
  getByPosition(position: NotificationPosition): Notification[] {
    return this.getAll().filter((n) => n.position === position);
  }

  /**
   * Gets notifications by type
   */
  getByType(type: NotificationType): Notification[] {
    return this.getAll().filter((n) => n.type === type);
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
    const notifications = this.getAll();
    this.listeners.forEach((listener) => {
      try {
        listener(notifications);
      } catch (error) {
        console.error('Notification listener error:', error);
      }
    });
  }
}

/**
 * Global notifications manager instance
 */
export const advancedNotificationsManager = new AdvancedNotificationsManager();

/**
 * Helper functions
 */
export const notify = {
  info: (message: string, options?: Omit<NotificationOptions, 'message' | 'type'>) => {
    return advancedNotificationsManager.show({ ...options, message, type: 'info' });
  },
  success: (message: string, options?: Omit<NotificationOptions, 'message' | 'type'>) => {
    return advancedNotificationsManager.show({ ...options, message, type: 'success' });
  },
  warning: (message: string, options?: Omit<NotificationOptions, 'message' | 'type'>) => {
    return advancedNotificationsManager.show({ ...options, message, type: 'warning' });
  },
  error: (message: string, options?: Omit<NotificationOptions, 'message' | 'type'>) => {
    return advancedNotificationsManager.show({ ...options, message, type: 'error' });
  },
  loading: (message: string, options?: Omit<NotificationOptions, 'message' | 'type'>) => {
    return advancedNotificationsManager.show({ ...options, message, type: 'loading', persistent: true });
  },
};



