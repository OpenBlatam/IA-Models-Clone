'use client';

import { toast as sonnerToast } from 'sonner';

export type ToastType = 'success' | 'error' | 'warning' | 'info';

export const toast = {
  success: (message: string, duration?: number) => {
    sonnerToast.success(message, {
      duration: duration || 3000,
      style: {
        background: '#f0fdf4',
        border: '1px solid #bbf7d0',
        color: '#166534',
      },
    });
  },
  error: (message: string, duration?: number) => {
    sonnerToast.error(message, {
      duration: duration || 4000,
      style: {
        background: '#fef2f2',
        border: '1px solid #fecaca',
        color: '#991b1b',
      },
    });
  },
  warning: (message: string, duration?: number) => {
    sonnerToast.warning(message, {
      duration: duration || 3000,
      style: {
        background: '#fffbeb',
        border: '1px solid #fde68a',
        color: '#92400e',
      },
    });
  },
  info: (message: string, duration?: number) => {
    sonnerToast.info(message, {
      duration: duration || 3000,
      style: {
        background: '#eff6ff',
        border: '1px solid #bfdbfe',
        color: '#1e40af',
      },
    });
  },
};



