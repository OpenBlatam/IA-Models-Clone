/**
 * Hook para manejo centralizado de errores.
 */

import { useCallback, useState } from 'react';
import { toast } from 'sonner';
import { useNotifications } from './useNotifications';

export interface ErrorInfo {
  message: string;
  code?: string;
  details?: any;
  timestamp: Date;
  stack?: string;
}

interface UseErrorHandlerOptions {
  showToast?: boolean;
  logToConsole?: boolean;
  onError?: (error: ErrorInfo) => void;
}

/**
 * Hook para manejo centralizado de errores.
 */
export function useErrorHandler(options: UseErrorHandlerOptions = {}) {
  const {
    showToast = true,
    logToConsole = true,
    onError
  } = options;

  const { addNotification } = useNotifications();
  const [errors, setErrors] = useState<ErrorInfo[]>([]);
  const [lastError, setLastError] = useState<ErrorInfo | null>(null);

  const handleError = useCallback((
    error: Error | string | unknown,
    context?: string
  ) => {
    let errorInfo: ErrorInfo;

    if (error instanceof Error) {
      errorInfo = {
        message: error.message,
        code: (error as any).code,
        details: (error as any).details,
        timestamp: new Date(),
        stack: error.stack
      };
    } else if (typeof error === 'string') {
      errorInfo = {
        message: error,
        timestamp: new Date()
      };
    } else {
      errorInfo = {
        message: 'Error desconocido',
        details: error,
        timestamp: new Date()
      };
    }

    // Agregar contexto si se proporciona
    if (context) {
      errorInfo.message = `[${context}] ${errorInfo.message}`;
    }

    // Log a consola
    if (logToConsole) {
      console.error('Error handled:', errorInfo);
      if (errorInfo.stack) {
        console.error('Stack:', errorInfo.stack);
      }
    }

    // Mostrar toast
    if (showToast) {
      toast.error(errorInfo.message, {
        description: errorInfo.code ? `Código: ${errorInfo.code}` : undefined,
        duration: 5000
      });
    }

    // Agregar notificación
    addNotification('error', 'Error', errorInfo.message);

    // Guardar en estado
    setErrors(prev => [errorInfo, ...prev].slice(0, 50)); // Mantener últimos 50
    setLastError(errorInfo);

    // Callback personalizado
    onError?.(errorInfo);
  }, [showToast, logToConsole, addNotification, onError]);

  const handleAsyncError = useCallback(async <T,>(
    asyncFn: () => Promise<T>,
    context?: string
  ): Promise<T | null> => {
    try {
      return await asyncFn();
    } catch (error) {
      handleError(error, context);
      return null;
    }
  }, [handleError]);

  const clearErrors = useCallback(() => {
    setErrors([]);
    setLastError(null);
  }, []);

  const clearLastError = useCallback(() => {
    setLastError(null);
  }, []);

  return {
    handleError,
    handleAsyncError,
    errors,
    lastError,
    clearErrors,
    clearLastError,
    hasErrors: errors.length > 0
  };
}



