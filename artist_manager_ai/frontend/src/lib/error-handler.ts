import { toast } from './toast';

export const handleApiError = (error: unknown, defaultMessage = 'Ha ocurrido un error') => {
  if (error instanceof Error) {
    toast.error(error.message || defaultMessage);
    return error.message || defaultMessage;
  }
  
  if (typeof error === 'string') {
    toast.error(error);
    return error;
  }
  
  toast.error(defaultMessage);
  return defaultMessage;
};

export const handleApiSuccess = (message: string) => {
  toast.success(message);
};

