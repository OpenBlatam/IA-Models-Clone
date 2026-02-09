import { useState, useCallback } from 'react';

type ToastVariant = 'success' | 'error' | 'warning' | 'info';

interface ToastState {
  message: string;
  variant: ToastVariant;
  isVisible: boolean;
}

const INITIAL_STATE: ToastState = {
  message: '',
  variant: 'info',
  isVisible: false,
};

export const useToast = () => {
  const [toast, setToast] = useState<ToastState>(INITIAL_STATE);

  const showToast = useCallback((message: string, variant: ToastVariant = 'info') => {
    setToast({
      message,
      variant,
      isVisible: true,
    });
  }, []);

  const hideToast = useCallback(() => {
    setToast((prev) => ({ ...prev, isVisible: false }));
  }, []);

  const showSuccess = useCallback(
    (message: string) => {
      showToast(message, 'success');
    },
    [showToast]
  );

  const showError = useCallback(
    (message: string) => {
      showToast(message, 'error');
    },
    [showToast]
  );

  const showWarning = useCallback(
    (message: string) => {
      showToast(message, 'warning');
    },
    [showToast]
  );

  const showInfo = useCallback(
    (message: string) => {
      showToast(message, 'info');
    },
    [showToast]
  );

  return {
    toast,
    showToast,
    hideToast,
    showSuccess,
    showError,
    showWarning,
    showInfo,
  };
};



