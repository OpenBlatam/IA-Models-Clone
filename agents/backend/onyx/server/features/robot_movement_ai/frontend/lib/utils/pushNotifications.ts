// Push notifications utility

export interface NotificationPermission {
  granted: boolean;
  denied: boolean;
  default: boolean;
}

class PushNotificationService {
  private permission: NotificationPermission = {
    granted: false,
    denied: false,
    default: true,
  };

  async requestPermission(): Promise<boolean> {
    if (!('Notification' in window)) {
      console.warn('This browser does not support notifications');
      return false;
    }

    if (Notification.permission === 'granted') {
      this.permission = { granted: true, denied: false, default: false };
      return true;
    }

    if (Notification.permission === 'denied') {
      this.permission = { granted: false, denied: true, default: false };
      return false;
    }

    const permission = await Notification.requestPermission();
    if (permission === 'granted') {
      this.permission = { granted: true, denied: false, default: false };
      return true;
    }

    this.permission = { granted: false, denied: true, default: false };
    return false;
  }

  showNotification(
    title: string,
    options?: NotificationOptions
  ): Notification | null {
    if (!('Notification' in window)) {
      return null;
    }

    if (Notification.permission !== 'granted') {
      console.warn('Notification permission not granted');
      return null;
    }

    const notification = new Notification(title, {
      icon: '/favicon.ico',
      badge: '/favicon.ico',
      ...options,
    });

    notification.onclick = () => {
      window.focus();
      notification.close();
    };

    return notification;
  }

  getPermission(): NotificationPermission {
    if ('Notification' in window) {
      if (Notification.permission === 'granted') {
        return { granted: true, denied: false, default: false };
      }
      if (Notification.permission === 'denied') {
        return { granted: false, denied: true, default: false };
      }
    }
    return { granted: false, denied: false, default: true };
  }
}

export const pushNotificationService = new PushNotificationService();

