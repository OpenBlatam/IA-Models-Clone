/**
 * Toast hook for managing toast notifications
 */

import { useState, useCallback } from 'react';
import type { ToastVariant } from '@/components/ui/Toast';
import type { ToastData } from '@/components/ui/ToastContainer';

export const useToast = () => {
  const [toasts, setToasts] = useState<ToastData[]>([]);

  const showToast = useCallback((
    message: string,
    variant: ToastVariant = 'info',
    duration?: number,
    title?: string
  ) => {
    const id = `toast-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const newToast: ToastData = {
      id,
      message,
      variant,
      duration,
      title,
    };

    setToasts((prev) => [...prev, newToast]);
    return id;
  }, []);

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((toast) => toast.id !== id));
  }, []);

  const success = useCallback((message: string, title?: string) => {
    return showToast(message, 'success', 5000, title);
  }, [showToast]);

  const error = useCallback((message: string, title?: string) => {
    return showToast(message, 'error', 7000, title);
  }, [showToast]);

  const warning = useCallback((message: string, title?: string) => {
    return showToast(message, 'warning', 6000, title);
  }, [showToast]);

  const info = useCallback((message: string, title?: string) => {
    return showToast(message, 'info', 5000, title);
  }, [showToast]);

  return {
    toasts,
    showToast,
    removeToast,
    success,
    error,
    warning,
    info,
  };
};



