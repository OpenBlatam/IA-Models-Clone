/**
 * Error handling utilities
 */

import { ChatInterfaceError } from '../types'
import { ERROR_MESSAGES } from './constants'

/**
 * Create a ChatInterfaceError
 */
export function createError(
  code: string,
  message: string,
  details?: any
): ChatInterfaceError {
  return new ChatInterfaceError(code, message, details)
}

/**
 * Handle and log errors
 */
export function handleError(error: unknown, context?: string): ChatInterfaceError {
  if (error instanceof ChatInterfaceError) {
    console.error(`[ChatInterface${context ? `:${context}` : ''}]`, error)
    return error
  }

  if (error instanceof Error) {
    const chatError = createError('UNKNOWN_ERROR', error.message, {
      originalError: error,
      stack: error.stack,
      context,
    })
    console.error(`[ChatInterface${context ? `:${context}` : ''}]`, chatError)
    return chatError
  }

  const chatError = createError(
    'UNKNOWN_ERROR',
    'An unknown error occurred',
    { originalError: error, context }
  )
  console.error(`[ChatInterface${context ? `:${context}` : ''}]`, chatError)
  return chatError
}

/**
 * Safe async function wrapper
 */
export async function safeAsync<T>(
  fn: () => Promise<T>,
  fallback?: T,
  errorHandler?: (error: unknown) => void
): Promise<T | undefined> {
  try {
    return await fn()
  } catch (error) {
    const handledError = handleError(error)
    if (errorHandler) {
      errorHandler(handledError)
    }
    return fallback
  }
}

/**
 * Safe sync function wrapper
 */
export function safeSync<T>(
  fn: () => T,
  fallback?: T,
  errorHandler?: (error: unknown) => void
): T | undefined {
  try {
    return fn()
  } catch (error) {
    const handledError = handleError(error)
    if (errorHandler) {
      errorHandler(handledError)
    }
    return fallback
  }
}

/**
 * Retry function with exponential backoff
 */
export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries: number = 3,
  initialDelay: number = 1000,
  onRetry?: (attempt: number, error: unknown) => void
): Promise<T> {
  let lastError: unknown

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn()
    } catch (error) {
      lastError = error

      if (attempt < maxRetries) {
        const delay = initialDelay * Math.pow(2, attempt)
        if (onRetry) {
          onRetry(attempt + 1, error)
        }
        await new Promise(resolve => setTimeout(resolve, delay))
      }
    }
  }

  throw lastError
}

/**
 * Error boundary helper
 */
export function isError(error: unknown): error is Error {
  return error instanceof Error
}

export function isChatInterfaceError(error: unknown): error is ChatInterfaceError {
  return error instanceof ChatInterfaceError
}

/**
 * Get user-friendly error message
 */
export function getUserFriendlyError(error: unknown): string {
  if (isChatInterfaceError(error)) {
    return error.message
  }

  if (isError(error)) {
    // Map common errors to user-friendly messages
    if (error.name === 'QuotaExceededError') {
      return ERROR_MESSAGES.STORAGE_QUOTA_EXCEEDED
    }
    if (error.name === 'NetworkError' || error.message.includes('network')) {
      return ERROR_MESSAGES.NETWORK_ERROR
    }
    if (error.name === 'PermissionDeniedError' || error.message.includes('permission')) {
      return ERROR_MESSAGES.PERMISSION_DENIED
    }
    return error.message
  }

  return ERROR_MESSAGES.NETWORK_ERROR
}

/**
 * Error reporting (for external services like Sentry)
 */
export interface ErrorReporter {
  captureException(error: Error, context?: Record<string, any>): void
  captureMessage(message: string, level?: 'info' | 'warning' | 'error'): void
}

let errorReporter: ErrorReporter | null = null

export function setErrorReporter(reporter: ErrorReporter): void {
  errorReporter = reporter
}

export function reportError(error: unknown, context?: Record<string, any>): void {
  if (!errorReporter) return

  if (isError(error)) {
    errorReporter.captureException(error, context)
  } else {
    errorReporter.captureMessage(String(error), 'error')
  }
}




