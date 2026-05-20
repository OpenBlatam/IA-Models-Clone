import { useCallback } from 'react';
import { toast } from '@/lib/utils/toast';

export type NotificationType = 'success' | 'error' | 'warning' | 'info';

export interface UseNotificationReturn {
  notify: (message: string, type?: NotificationType) => void;
  success: (message: string) => void;
  error: (message: string) => void;
  warning: (message: string) => void;
  info: (message: string) => void;
}

/**
 * Hook for notifications/toasts
 */
export function useNotification(): UseNotificationReturn {
  const notify = useCallback((message: string, type: NotificationType = 'info') => {
    switch (type) {
      case 'success':
        toast.success(message);
        break;
      case 'error':
        toast.error(message);
        break;
      case 'warning':
        toast.warning?.(message) || toast.error(message);
        break;
      case 'info':
      default:
        toast.info?.(message) || toast(message);
        break;
    }
  }, []);

  const success = useCallback((message: string) => {
    notify(message, 'success');
  }, [notify]);

  const error = useCallback((message: string) => {
    notify(message, 'error');
  }, [notify]);

  const warning = useCallback((message: string) => {
    notify(message, 'warning');
  }, [notify]);

  const info = useCallback((message: string) => {
    notify(message, 'info');
  }, [notify]);

  return {
    notify,
    success,
    error,
    warning,
    info,
  };
}



