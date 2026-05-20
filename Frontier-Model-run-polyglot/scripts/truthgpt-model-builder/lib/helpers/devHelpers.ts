/**
 * Helpers para Desarrollo
 * =======================
 * 
 * Utilidades para desarrollo y debugging
 */

// ============================================================================
// LOGGING
// ============================================================================

const LOG_LEVELS = {
  DEBUG: 0,
  INFO: 1,
  WARN: 2,
  ERROR: 3
} as const

type LogLevel = keyof typeof LOG_LEVELS

class Logger {
  private level: LogLevel
  private enabled: boolean

  constructor() {
    this.level = (process.env.NODE_ENV === 'production' ? 'ERROR' : 'DEBUG') as LogLevel
    this.enabled = process.env.NODE_ENV !== 'production'
  }

  private shouldLog(level: LogLevel): boolean {
    if (!this.enabled) return false
    return LOG_LEVELS[level] >= LOG_LEVELS[this.level]
  }

  debug(...args: unknown[]): void {
    if (this.shouldLog('DEBUG')) {
      console.debug('[DEBUG]', ...args)
    }
  }

  info(...args: unknown[]): void {
    if (this.shouldLog('INFO')) {
      console.info('[INFO]', ...args)
    }
  }

  warn(...args: unknown[]): void {
    if (this.shouldLog('WARN')) {
      console.warn('[WARN]', ...args)
    }
  }

  error(...args: unknown[]): void {
    if (this.shouldLog('ERROR')) {
      console.error('[ERROR]', ...args)
    }
  }

  group(label: string): void {
    if (this.enabled) {
      console.group(label)
    }
  }

  groupEnd(): void {
    if (this.enabled) {
      console.groupEnd()
    }
  }

  table(data: unknown): void {
    if (this.enabled) {
      console.table(data)
    }
  }
}

export const logger = new Logger()

// ============================================================================
// PERFORMANCE
// ============================================================================

/**
 * Mide el tiempo de ejecución de una función
 */
export function measurePerformance<T>(
  label: string,
  fn: () => T
): T {
  const start = performance.now()
  const result = fn()
  const end = performance.now()
  
  logger.debug(`[Performance] ${label}: ${(end - start).toFixed(2)}ms`)
  
  return result
}

/**
 * Mide el tiempo de ejecución de una función asíncrona
 */
export async function measurePerformanceAsync<T>(
  label: string,
  fn: () => Promise<T>
): Promise<T> {
  const start = performance.now()
  const result = await fn()
  const end = performance.now()
  
  logger.debug(`[Performance] ${label}: ${(end - start).toFixed(2)}ms`)
  
  return result
}

// ============================================================================
// DEBUGGING
// ============================================================================

/**
 * Muestra información de debugging de un objeto
 */
export function debugObject(obj: unknown, label?: string): void {
  if (process.env.NODE_ENV === 'production') return

  if (label) {
    logger.group(label)
  }

  logger.debug('Type:', typeof obj)
  logger.debug('Value:', obj)
  
  if (obj && typeof obj === 'object') {
    logger.debug('Keys:', Object.keys(obj))
    logger.table(obj)
  }

  if (label) {
    logger.groupEnd()
  }
}

/**
 * Crea un proxy para debugging
 */
export function debugProxy<T extends object>(obj: T, label: string = 'Object'): T {
  if (process.env.NODE_ENV === 'production') return obj

  return new Proxy(obj, {
    get(target, prop) {
      const value = Reflect.get(target, prop)
      logger.debug(`[${label}] Get ${String(prop)}:`, value)
      return value
    },
    set(target, prop, value) {
      logger.debug(`[${label}] Set ${String(prop)}:`, value)
      return Reflect.set(target, prop, value)
    }
  })
}

// ============================================================================
// ASSERTIONS
// ============================================================================

/**
 * Assert para desarrollo
 */
export function assert(condition: boolean, message: string): asserts condition {
  if (process.env.NODE_ENV === 'production') return

  if (!condition) {
    throw new Error(`Assertion failed: ${message}`)
  }
}

/**
 * Assert que un valor no es null/undefined
 */
export function assertDefined<T>(value: T | null | undefined, message?: string): asserts value is T {
  if (process.env.NODE_ENV === 'production') return

  if (value === null || value === undefined) {
    throw new Error(`Assertion failed: ${message || 'Value is null or undefined'}`)
  }
}

// ============================================================================
// ENVIRONMENT
// ============================================================================

/**
 * Verifica si estamos en desarrollo
 */
export function isDevelopment(): boolean {
  return process.env.NODE_ENV === 'development'
}

/**
 * Verifica si estamos en producción
 */
export function isProduction(): boolean {
  return process.env.NODE_ENV === 'production'
}

/**
 * Verifica si estamos en test
 */
export function isTest(): boolean {
  return process.env.NODE_ENV === 'test'
}

// ============================================================================
// ERROR HELPERS
// ============================================================================

/**
 * Crea un error con stack trace mejorado
 */
export function createError(message: string, cause?: Error): Error {
  const error = new Error(message)
  if (cause) {
    error.cause = cause
  }
  return error
}

/**
 * Extrae información útil de un error
 */
export function extractErrorInfo(error: unknown): {
  message: string
  stack?: string
  name: string
  cause?: unknown
} {
  if (error instanceof Error) {
    return {
      message: error.message,
      stack: error.stack,
      name: error.name,
      cause: error.cause
    }
  }

  return {
    message: String(error),
    name: 'UnknownError'
  }
}

// ============================================================================
// TYPE HELPERS
// ============================================================================

/**
 * Type guard para verificar si un valor es un objeto
 */
export function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/**
 * Type guard para verificar si un valor es un array
 */
export function isArray(value: unknown): value is unknown[] {
  return Array.isArray(value)
}

/**
 * Type guard para verificar si un valor es una función
 */
export function isFunction(value: unknown): value is Function {
  return typeof value === 'function'
}

/**
 * Type guard para verificar si un valor es un string
 */
export function isString(value: unknown): value is string {
  return typeof value === 'string'
}

/**
 * Type guard para verificar si un valor es un número
 */
export function isNumber(value: unknown): value is number {
  return typeof value === 'number' && !isNaN(value) && isFinite(value)
}







