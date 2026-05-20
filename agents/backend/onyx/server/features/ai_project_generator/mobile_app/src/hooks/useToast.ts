import { useContext } from 'react';
import { ToastContext } from '../components/Toast';
import { ToastType } from '../components/Toast';

export const useToast = () => {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within ToastProvider');
  }
  return context;
};

export const useToastHelpers = () => {
  const { showToast } = useToast();

  return {
    showSuccess: (message: string, duration?: number) =>
      showToast(message, 'success', duration),
    showError: (message: string, duration?: number) =>
      showToast(message, 'error', duration),
    showWarning: (message: string, duration?: number) =>
      showToast(message, 'warning', duration),
    showInfo: (message: string, duration?: number) =>
      showToast(message, 'info', duration),
  };
};

