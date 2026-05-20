/**
 * Utilidades de Logging
 * =====================
 * 
 * Sistema de logging mejorado
 */

export type LogLevel = 'debug' | 'info' | 'warn' | 'error'

export interface LogEntry {
  level: LogLevel
  message: string
  data?: any
  timestamp: number
  context?: string
}

export interface LoggerOptions {
  level?: LogLevel
  enableConsole?: boolean
  enableStorage?: boolean
  maxStorageEntries?: number
  context?: string
}

/**
 * Clase Logger
 */
export class Logger {
  private level: LogLevel
  private enableConsole: boolean
  private enableStorage: boolean
  private maxStorageEntries: number
  private context?: string
  private storage: LogEntry[] = []

  constructor(options: LoggerOptions = {}) {
    this.level = options.level || 'info'
    this.enableConsole = options.enableConsole !== false
    this.enableStorage = options.enableStorage || false
    this.maxStorageEntries = options.maxStorageEntries || 1000
    this.context = options.context
  }

  /**
   * Verifica si un nivel debe ser logueado
   */
  private shouldLog(level: LogLevel): boolean {
    const levels: LogLevel[] = ['debug', 'info', 'warn', 'error']
    return levels.indexOf(level) >= levels.indexOf(this.level)
  }

  /**
   * Crea una entrada de log
   */
  private createEntry(level: LogLevel, message: string, data?: any): LogEntry {
    return {
      level,
      message,
      data,
      timestamp: Date.now(),
      context: this.context
    }
  }

  /**
   * Almacena una entrada
   */
  private storeEntry(entry: LogEntry): void {
    if (!this.enableStorage) return

    this.storage.push(entry)

    if (this.storage.length > this.maxStorageEntries) {
      this.storage.shift()
    }
  }

  /**
   * Log debug
   */
  debug(message: string, data?: any): void {
    if (!this.shouldLog('debug')) return

    const entry = this.createEntry('debug', message, data)
    this.storeEntry(entry)

    if (this.enableConsole) {
      console.debug(`[DEBUG]${this.context ? ` [${this.context}]` : ''}`, message, data || '')
    }
  }

  /**
   * Log info
   */
  info(message: string, data?: any): void {
    if (!this.shouldLog('info')) return

    const entry = this.createEntry('info', message, data)
    this.storeEntry(entry)

    if (this.enableConsole) {
      console.info(`[INFO]${this.context ? ` [${this.context}]` : ''}`, message, data || '')
    }
  }

  /**
   * Log warn
   */
  warn(message: string, data?: any): void {
    if (!this.shouldLog('warn')) return

    const entry = this.createEntry('warn', message, data)
    this.storeEntry(entry)

    if (this.enableConsole) {
      console.warn(`[WARN]${this.context ? ` [${this.context}]` : ''}`, message, data || '')
    }
  }

  /**
   * Log error
   */
  error(message: string, error?: Error | any): void {
    if (!this.shouldLog('error')) return

    const entry = this.createEntry('error', message, error)
    this.storeEntry(entry)

    if (this.enableConsole) {
      console.error(`[ERROR]${this.context ? ` [${this.context}]` : ''}`, message, error || '')
    }
  }

  /**
   * Crea un logger con contexto
   */
  withContext(context: string): Logger {
    return new Logger({
      level: this.level,
      enableConsole: this.enableConsole,
      enableStorage: this.enableStorage,
      maxStorageEntries: this.maxStorageEntries,
      context: this.context ? `${this.context}.${context}` : context
    })
  }

  /**
   * Obtiene los logs almacenados
   */
  getLogs(level?: LogLevel): LogEntry[] {
    if (level) {
      return this.storage.filter(entry => entry.level === level)
    }
    return [...this.storage]
  }

  /**
   * Limpia los logs almacenados
   */
  clearLogs(): void {
    this.storage = []
  }

  /**
   * Exporta los logs
   */
  exportLogs(): string {
    return JSON.stringify(this.storage, null, 2)
  }
}

/**
 * Logger global
 */
let globalLogger: Logger | null = null

/**
 * Obtiene el logger global
 */
export function getLogger(options?: LoggerOptions): Logger {
  if (!globalLogger) {
    globalLogger = new Logger(options)
  }
  return globalLogger
}

/**
 * Establece el logger global
 */
export function setLogger(logger: Logger): void {
  globalLogger = logger
}

/**
 * Crea un logger con contexto
 */
export function createLogger(context: string, options?: LoggerOptions): Logger {
  return new Logger({ ...options, context })
}

/**
 * Hook para logging en componentes
 */
export function useLogger(context?: string): Logger {
  return createLogger(context || 'Component', {
    enableConsole: true,
    enableStorage: false
  })
}






