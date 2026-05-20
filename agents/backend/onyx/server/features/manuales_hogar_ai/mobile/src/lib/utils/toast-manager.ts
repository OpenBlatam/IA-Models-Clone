/**
 * Toast Manager
 * =============
 * Global toast notification manager
 */

import { EventEmitter } from 'events';

interface ToastOptions {
  message: string;
  type?: 'success' | 'error' | 'info' | 'warning';
  duration?: number;
}

class ToastManager extends EventEmitter {
  show(options: ToastOptions) {
    this.emit('show', options);
  }

  success(message: string, duration = 3000) {
    this.show({ message, type: 'success', duration });
  }

  error(message: string, duration = 4000) {
    this.show({ message, type: 'error', duration });
  }

  info(message: string, duration = 3000) {
    this.show({ message, type: 'info', duration });
  }

  warning(message: string, duration = 3000) {
    this.show({ message, type: 'warning', duration });
  }
}

export const toastManager = new ToastManager();



