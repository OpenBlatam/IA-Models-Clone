import toast from 'react-hot-toast';
import { AxiosError } from 'axios';

interface ApiError {
  detail?: string;
  message?: string;
  error?: string;
}

export const handleApiError = (error: unknown): string => {
  if (error instanceof AxiosError) {
    const apiError = error.response?.data as ApiError | undefined;
    
    if (apiError?.detail) {
      return apiError.detail;
    }
    
    if (apiError?.message) {
      return apiError.message;
    }
    
    if (apiError?.error) {
      return apiError.error;
    }
    
    if (error.response?.status === 404) {
      return 'Recurso no encontrado';
    }
    
    if (error.response?.status === 403) {
      return 'No tienes permisos para realizar esta acción';
    }
    
    if (error.response?.status === 401) {
      return 'No autorizado. Por favor, inicia sesión';
    }
    
    if (error.response?.status >= 500) {
      return 'Error del servidor. Por favor, intenta más tarde';
    }
    
    if (error.message) {
      return error.message;
    }
  }
  
  if (error instanceof Error) {
    return error.message;
  }
  
  return 'Ocurrió un error inesperado';
};

export const showErrorToast = (error: unknown): void => {
  const message = handleApiError(error);
  toast.error(message);
};

export const showSuccessToast = (message: string): void => {
  toast.success(message);
};

