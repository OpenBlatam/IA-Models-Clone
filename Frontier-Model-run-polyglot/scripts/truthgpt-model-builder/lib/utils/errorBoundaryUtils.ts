/**
 * Utilidades de Error Boundary
 * =============================
 * 
 * Funciones para manejar errores en boundaries
 */

export interface ErrorInfo {
  error: Error
  errorInfo?: {
    componentStack?: string
  }
  timestamp: number
  context?: Record<string, any>
}

/**
 * Clase para manejar errores de boundaries
 */
export class ErrorBoundaryHandler {
  private errors: ErrorInfo[] = []
  private maxErrors: number = 100
  private onError?: (errorInfo: ErrorInfo) => void

  constructor(onError?: (errorInfo: ErrorInfo) => void) {
    this.onError = onError
  }

  /**
   * Registra un error
   */
  handleError(
    error: Error,
    errorInfo?: { componentStack?: string },
    context?: Record<string, any>
  ): void {
    const errorInfoObj: ErrorInfo = {
      error,
      errorInfo,
      timestamp: Date.now(),
      context
    }

    this.errors.push(errorInfoObj)

    if (this.errors.length > this.maxErrors) {
      this.errors.shift()
    }

    if (this.onError) {
      this.onError(errorInfoObj)
    }
  }

  /**
   * Obtiene los errores
   */
  getErrors(): ErrorInfo[] {
    return [...this.errors]
  }

  /**
   * Obtiene el último error
   */
  getLastError(): ErrorInfo | null {
    return this.errors[this.errors.length - 1] || null
  }

  /**
   * Limpia los errores
   */
  clearErrors(): void {
    this.errors = []
  }

  /**
   * Exporta los errores
   */
  exportErrors(): string {
    return JSON.stringify(this.errors, null, 2)
  }

  /**
   * Obtiene estadísticas de errores
   */
  getStats(): {
    total: number
    byType: Record<string, number>
    recent: ErrorInfo[]
  } {
    const byType: Record<string, number> = {}

    for (const errorInfo of this.errors) {
      const type = errorInfo.error.name || 'Unknown'
      byType[type] = (byType[type] || 0) + 1
    }

    return {
      total: this.errors.length,
      byType,
      recent: this.errors.slice(-10)
    }
  }
}

/**
 * Handler global
 */
let globalHandler: ErrorBoundaryHandler | null = null

/**
 * Obtiene el handler global
 */
export function getErrorBoundaryHandler(
  onError?: (errorInfo: ErrorInfo) => void
): ErrorBoundaryHandler {
  if (!globalHandler) {
    globalHandler = new ErrorBoundaryHandler(onError)
  }
  return globalHandler
}

/**
 * Crea un mensaje de error amigable
 */
export function getFriendlyErrorMessage(error: Error): string {
  const errorMessages: Record<string, string> = {
    'NetworkError': 'Error de conexión. Por favor, verifica tu conexión a internet.',
    'TimeoutError': 'La operación tardó demasiado. Por favor, intenta de nuevo.',
    'ValidationError': 'Los datos ingresados no son válidos. Por favor, revisa el formulario.',
    'NotFoundError': 'El recurso solicitado no fue encontrado.',
    'UnauthorizedError': 'No tienes permisos para realizar esta acción.',
    'ForbiddenError': 'Acceso denegado.',
    'ServerError': 'Error del servidor. Por favor, intenta más tarde.',
    'UnknownError': 'Ocurrió un error inesperado. Por favor, intenta de nuevo.'
  }

  for (const [errorType, message] of Object.entries(errorMessages)) {
    if (error.name.includes(errorType) || error.message.includes(errorType)) {
      return message
    }
  }

  return error.message || 'Ocurrió un error inesperado. Por favor, intenta de nuevo.'
}

/**
 * Determina si un error es recuperable
 */
export function isRecoverableError(error: Error): boolean {
  const recoverableErrors = [
    'NetworkError',
    'TimeoutError',
    'ValidationError'
  ]

  return recoverableErrors.some(type => 
    error.name.includes(type) || error.message.includes(type)
  )
}

/**
 * Crea un error con contexto
 */
export function createContextualError(
  message: string,
  context?: Record<string, any>,
  originalError?: Error
): Error {
  const error = originalError || new Error(message)
  
  if (context) {
    (error as any).context = context
  }

  return error
}






