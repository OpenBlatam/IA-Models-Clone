type ToastType = 'success' | 'error' | 'warning' | 'info';

interface Toast {
  id: string;
  message: string;
  type: ToastType;
}

let toastId = 0;
const toasts: Toast[] = [];
let listeners: Array<() => void> = [];

const notify = () => {
  listeners.forEach((listener) => listener());
};

export const showToast = (message: string, type: ToastType = 'info') => {
  const id = `toast-${toastId++}`;
  toasts.push({ id, message, type });
  notify();

  setTimeout(() => {
    const index = toasts.findIndex((t) => t.id === id);
    if (index > -1) {
      toasts.splice(index, 1);
      notify();
    }
  }, 5000);
};

export const getToasts = () => toasts;

export const subscribe = (listener: () => void) => {
  listeners.push(listener);
  return () => {
    listeners = listeners.filter((l) => l !== listener);
  };
};


