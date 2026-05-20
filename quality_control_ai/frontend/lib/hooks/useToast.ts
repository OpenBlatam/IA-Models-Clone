import { toast as sonnerToast } from 'sonner';

export const useToast = () => {
  const toast = (message: string, type: 'success' | 'error' | 'warning' | 'info' = 'info'): void => {
    switch (type) {
      case 'success':
        sonnerToast.success(message);
        break;
      case 'error':
        sonnerToast.error(message);
        break;
      case 'warning':
        sonnerToast.warning(message);
        break;
      case 'info':
      default:
        sonnerToast.info(message);
        break;
    }
  };

  return {
    toast,
    success: (message: string) => sonnerToast.success(message),
    error: (message: string) => sonnerToast.error(message),
    warning: (message: string) => sonnerToast.warning(message),
    info: (message: string) => sonnerToast.info(message),
    promise: <T,>(promise: Promise<T>, messages: { loading: string; success: string; error: string }) => {
      return sonnerToast.promise(promise, messages);
    },
  };
};
