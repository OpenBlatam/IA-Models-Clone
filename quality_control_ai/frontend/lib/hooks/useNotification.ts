import { useCallback, useState } from 'react';

interface NotificationOptions {
  title: string;
  body?: string;
  icon?: string;
  badge?: string;
  image?: string;
  tag?: string;
  requireInteraction?: boolean;
  silent?: boolean;
  timestamp?: number;
  vibrate?: number[];
  data?: unknown;
  actions?: NotificationAction[];
  dir?: 'auto' | 'ltr' | 'rtl';
  lang?: string;
  renotify?: boolean;
}

export const useNotification = () => {
  const [permission, setPermission] = useState<NotificationPermission>(
    typeof Notification !== 'undefined' ? Notification.permission : 'default'
  );
  const [isSupported, setIsSupported] = useState(
    typeof Notification !== 'undefined' && 'Notification' in window
  );

  const requestPermission = useCallback(async (): Promise<NotificationPermission> => {
    if (!isSupported) {
      return 'denied';
    }

    const result = await Notification.requestPermission();
    setPermission(result);
    return result;
  }, [isSupported]);

  const showNotification = useCallback(
    async (options: NotificationOptions): Promise<Notification | null> => {
      if (!isSupported) {
        return null;
      }

      if (permission !== 'granted') {
        const newPermission = await requestPermission();
        if (newPermission !== 'granted') {
          return null;
        }
      }

      const notification = new Notification(options.title, {
        body: options.body,
        icon: options.icon,
        badge: options.badge,
        image: options.image,
        tag: options.tag,
        requireInteraction: options.requireInteraction,
        silent: options.silent,
        timestamp: options.timestamp,
        vibrate: options.vibrate,
        data: options.data,
        actions: options.actions,
        dir: options.dir,
        lang: options.lang,
        renotify: options.renotify,
      });

      return notification;
    },
    [isSupported, permission, requestPermission]
  );

  return {
    showNotification,
    requestPermission,
    permission,
    isSupported,
  };
};

