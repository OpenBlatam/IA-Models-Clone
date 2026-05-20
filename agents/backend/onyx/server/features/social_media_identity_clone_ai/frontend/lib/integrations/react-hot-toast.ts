import toast, { Toaster, ToastOptions } from 'react-hot-toast';

export const showToast = {
  success: (message: string, options?: ToastOptions): string => {
    return toast.success(message, {
      duration: 3000,
      position: 'top-right',
      ...options,
    });
  },
  error: (message: string, options?: ToastOptions): string => {
    return toast.error(message, {
      duration: 4000,
      position: 'top-right',
      ...options,
    });
  },
  loading: (message: string, options?: ToastOptions): string => {
    return toast.loading(message, {
      position: 'top-right',
      ...options,
    });
  },
  promise: <T,>(
    promise: Promise<T>,
    messages: {
      loading: string;
      success: string;
      error: string;
    },
    options?: ToastOptions
  ): Promise<T> => {
    return toast.promise(promise, messages, {
      position: 'top-right',
      ...options,
    });
  },
};

export { Toaster, toast };



