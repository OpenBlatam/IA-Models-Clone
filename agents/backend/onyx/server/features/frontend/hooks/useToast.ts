'use client';

import { toastManager } from '@/lib/toast-manager';

export function useToast() {
  return {
    success: toastManager.success.bind(toastManager),
    error: toastManager.error.bind(toastManager),
    warning: toastManager.warning.bind(toastManager),
    info: toastManager.info.bind(toastManager),
    show: toastManager.show.bind(toastManager),
  };
}

