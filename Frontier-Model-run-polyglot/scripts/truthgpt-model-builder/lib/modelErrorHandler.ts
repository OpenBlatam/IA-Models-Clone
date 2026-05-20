/**
 * Manejo centralizado de errores de modelos
 * ==========================================
 */

export enum ModelErrorType {
  NETWORK_ERROR = 'NETWORK_ERROR',
  VALIDATION_ERROR = 'VALIDATION_ERROR',
  API_ERROR = 'API_ERROR',
  TIMEOUT_ERROR = 'TIMEOUT_ERROR',
  RATE_LIMIT_ERROR = 'RATE_LIMIT_ERROR',
  UNKNOWN_ERROR = 'UNKNOWN_ERROR',
}

export interface ModelError {
  type: ModelErrorType
  message: string
  originalError?: Error
  retryable: boolean
  retryAfter?: number // segundos
}

/**
 * Clasifica errores de modelos
 */
export function classifyModelError(error: unknown): ModelError {
  const errorMessage = error instanceof Error ? error.message : String(error)
  const lowerMessage = errorMessage.toLowerCase()

  // Rate limit
  if (
    lowerMessage.includes('rate limit') ||
    lowerMessage.includes('rate_limit') ||
    lowerMessage.includes('429')
  ) {
    return {
      type: ModelErrorType.RATE_LIMIT_ERROR,
      message: 'Límite de solicitudes excedido. Por favor, espera un momento.',
      originalError: error instanceof Error ? error : undefined,
      retryable: true,
      retryAfter: 60, // 1 minuto
    }
  }

  // Timeout
  if (
    lowerMessage.includes('timeout') ||
    lowerMessage.includes('aborted') ||
    lowerMessage.includes('timed out')
  ) {
    return {
      type: ModelErrorType.TIMEOUT_ERROR,
      message: 'La solicitud tardó demasiado tiempo. Intenta de nuevo.',
      originalError: error instanceof Error ? error : undefined,
      retryable: true,
      retryAfter: 5,
    }
  }

  // Network errors
  if (
    lowerMessage.includes('network') ||
    lowerMessage.includes('fetch') ||
    lowerMessage.includes('connection') ||
    lowerMessage.includes('failed to fetch')
  ) {
    return {
      type: ModelErrorType.NETWORK_ERROR,
      message: 'Error de conexión. Verifica tu conexión a internet.',
      originalError: error instanceof Error ? error : undefined,
      retryable: true,
      retryAfter: 10,
    }
  }

  // Validation errors
  if (
    lowerMessage.includes('invalid') ||
    lowerMessage.includes('validation') ||
    lowerMessage.includes('required')
  ) {
    return {
      type: ModelErrorType.VALIDATION_ERROR,
      message: errorMessage,
      originalError: error instanceof Error ? error : undefined,
      retryable: false,
    }
  }

  // API errors (4xx, 5xx)
  if (lowerMessage.includes('http error') || lowerMessage.includes('status:')) {
    const statusMatch = errorMessage.match(/status:\s*(\d+)/i)
    const status = statusMatch ? parseInt(statusMatch[1], 10) : 0

    if (status >= 500) {
      return {
        type: ModelErrorType.API_ERROR,
        message: 'Error del servidor. Intenta de nuevo más tarde.',
        originalError: error instanceof Error ? error : undefined,
        retryable: true,
        retryAfter: 30,
      }
    }

    return {
      type: ModelErrorType.API_ERROR,
      message: errorMessage,
      originalError: error instanceof Error ? error : undefined,
      retryable: false,
    }
  }

  // Unknown error
  return {
    type: ModelErrorType.UNKNOWN_ERROR,
    message: errorMessage || 'Error desconocido',
    originalError: error instanceof Error ? error : undefined,
    retryable: true,
    retryAfter: 5,
  }
}

/**
 * Obtiene mensaje amigable para el usuario
 */
export function getFriendlyErrorMessage(error: ModelError): string {
  const messages: Record<ModelErrorType, string> = {
    [ModelErrorType.NETWORK_ERROR]: 'Problema de conexión. Verifica tu internet.',
    [ModelErrorType.VALIDATION_ERROR]: error.message,
    [ModelErrorType.API_ERROR]: 'Error del servidor. Intenta más tarde.',
    [ModelErrorType.TIMEOUT_ERROR]: 'La solicitud tardó demasiado. Intenta de nuevo.',
    [ModelErrorType.RATE_LIMIT_ERROR]: 'Demasiadas solicitudes. Espera un momento.',
    [ModelErrorType.UNKNOWN_ERROR]: 'Algo salió mal. Intenta de nuevo.',
  }

  return messages[error.type] || error.message
}

/**
 * Determina si un error es recuperable
 */
export function isRecoverableError(error: ModelError): boolean {
  return error.retryable && error.type !== ModelErrorType.VALIDATION_ERROR
}










