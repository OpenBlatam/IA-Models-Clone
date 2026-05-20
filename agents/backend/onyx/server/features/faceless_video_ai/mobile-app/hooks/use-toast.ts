import { useState, useCallback } from 'react';
import { Toast, type ToastOptions } from '@/components/ui/toast';

export function useToast() {
  const [toasts, setToasts] = useState<Array<ToastOptions & { id: string }>>([]);

  const showToast = useCallback((options: ToastOptions) => {
    const id = Math.random().toString(36).substring(7);
    const newToast = { ...options, id };

    setToasts((prev) => [...prev, newToast]);

    return id;
  }, []);

  const hideToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((toast) => toast.id !== id));
  }, []);

  const showSuccess = useCallback(
    (message: string, duration?: number) => {
      return showToast({ message, type: 'success', duration });
    },
    [showToast]
  );

  const showError = useCallback(
    (message: string, duration?: number) => {
      return showToast({ message, type: 'error', duration });
    },
    [showToast]
  );

  const showWarning = useCallback(
    (message: string, duration?: number) => {
      return showToast({ message, type: 'warning', duration });
    },
    [showToast]
  );

  const showInfo = useCallback(
    (message: string, duration?: number) => {
      return showToast({ message, type: 'info', duration });
    },
    [showToast]
  );

  const ToastContainer = () => (
    <>
      {toasts.map((toast) => (
        <Toast
          key={toast.id}
          {...toast}
          onHide={() => hideToast(toast.id)}
        />
      ))}
    </>
  );

  return {
    showToast,
    showSuccess,
    showError,
    showWarning,
    showInfo,
    hideToast,
    ToastContainer,
  };
}


