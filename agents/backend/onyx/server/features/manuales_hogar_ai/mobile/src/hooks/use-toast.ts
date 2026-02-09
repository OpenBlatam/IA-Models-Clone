/**
 * useToast Hook
 * =============
 * Hook for showing toast notifications
 */

import { useCallback } from 'react';
import { toastManager } from '@/lib/utils/toast-manager';

export function useToast() {
  const showSuccess = useCallback((message: string) => {
    toastManager.success(message);
  }, []);

  const showError = useCallback((message: string) => {
    toastManager.error(message);
  }, []);

  const showInfo = useCallback((message: string) => {
    toastManager.info(message);
  }, []);

  const showWarning = useCallback((message: string) => {
    toastManager.warning(message);
  }, []);

  return {
    showSuccess,
    showError,
    showInfo,
    showWarning,
  };
}



