/**
 * Centralized error handling
 */

export class AppError extends Error {
  constructor(
    message: string,
    public code: string,
    public statusCode?: number,
    public originalError?: Error
  ) {
    super(message);
    this.name = 'AppError';
  }
}

export const handleError = (error: any): string => {
  if (error instanceof AppError) {
    return error.message;
  }

  if (error.response) {
    // API Error
    const status = error.response.status;
    const data = error.response.data;

    switch (status) {
      case 400:
        return data.message || 'Solicitud inválida';
      case 401:
        return 'No autorizado. Por favor, inicia sesión.';
      case 403:
        return 'Acceso denegado';
      case 404:
        return 'Recurso no encontrado';
      case 500:
        return 'Error del servidor. Intenta más tarde.';
      default:
        return data.message || 'Error de conexión';
    }
  }

  if (error.message) {
    return error.message;
  }

  return 'Ocurrió un error inesperado';
};

export const logError = (error: any, context?: string) => {
  if (__DEV__) {
    console.error(`[Error${context ? ` - ${context}` : ''}]:`, error);
  }
  // In production, you could send to error tracking service
  // e.g., Sentry, Crashlytics, etc.
};

