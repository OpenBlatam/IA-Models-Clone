/**
 * Logger System
 * Sistema de logging detallado con niveles y auditoría
 */

export enum LogLevel {
  DEBUG = 'DEBUG',
  INFO = 'INFO',
  WARN = 'WARN',
  ERROR = 'ERROR',
}

export interface LogEntry {
  timestamp: number
  level: LogLevel
  message: string
  context?: Record<string, any>
  userId?: string
  sessionId?: string
}

export class Logger {
  private logs: LogEntry[] = []
  private maxLogs: number = 1000
  private currentLevel: LogLevel = LogLevel.INFO
  private sessionId: string

  constructor(sessionId?: string) {
    this.sessionId = sessionId || `session-${Date.now()}`
  }

  /**
   * Establecer nivel de log
   */
  setLevel(level: LogLevel): void {
    this.currentLevel = level
  }

  /**
   * Log debug
   */
  debug(message: string, context?: Record<string, any>): void {
    this.log(LogLevel.DEBUG, message, context)
  }

  /**
   * Log info
   */
  info(message: string, context?: Record<string, any>): void {
    this.log(LogLevel.INFO, message, context)
  }

  /**
   * Log warning
   */
  warn(message: string, context?: Record<string, any>): void {
    this.log(LogLevel.WARN, message, context)
  }

  /**
   * Log error
   */
  error(message: string, error?: Error | any, context?: Record<string, any>): void {
    const errorContext = {
      ...context,
      error: error instanceof Error ? {
        message: error.message,
        stack: error.stack,
        name: error.name,
      } : error,
    }
    this.log(LogLevel.ERROR, message, errorContext)
  }

  /**
   * Log genérico
   */
  private log(level: LogLevel, message: string, context?: Record<string, any>): void {
    // Solo log si el nivel es suficiente
    if (this.shouldLog(level)) {
      const entry: LogEntry = {
        timestamp: Date.now(),
        level,
        message,
        context,
        sessionId: this.sessionId,
      }

      this.logs.push(entry)

      // Limitar tamaño de logs
      if (this.logs.length > this.maxLogs) {
        this.logs.shift()
      }

      // También log a consola en desarrollo
      if (typeof window !== 'undefined' && process.env.NODE_ENV === 'development') {
        const consoleMethod = this.getConsoleMethod(level)
        consoleMethod(`[${level}]`, message, context || '')
      }
    }
  }

  /**
   * Verificar si debe loguear
   */
  private shouldLog(level: LogLevel): boolean {
    const levels = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARN, LogLevel.ERROR]
    const currentIndex = levels.indexOf(this.currentLevel)
    const messageIndex = levels.indexOf(level)
    return messageIndex >= currentIndex
  }

  /**
   * Obtener método de consola apropiado
   */
  private getConsoleMethod(level: LogLevel): (...args: any[]) => void {
    switch (level) {
      case LogLevel.DEBUG:
        return console.debug
      case LogLevel.INFO:
        return console.info
      case LogLevel.WARN:
        return console.warn
      case LogLevel.ERROR:
        return console.error
      default:
        return console.log
    }
  }

  /**
   * Obtener logs
   */
  getLogs(level?: LogLevel, limit?: number): LogEntry[] {
    let filtered = this.logs

    if (level) {
      filtered = filtered.filter(log => log.level === level)
    }

    if (limit) {
      filtered = filtered.slice(-limit)
    }

    return filtered
  }

  /**
   * Exportar logs
   */
  exportLogs(format: 'json' | 'text' = 'json'): string {
    if (format === 'json') {
      return JSON.stringify(this.logs, null, 2)
    } else {
      return this.logs.map(log => {
        const date = new Date(log.timestamp).toISOString()
        const context = log.context ? ` ${JSON.stringify(log.context)}` : ''
        return `[${date}] [${log.level}] ${log.message}${context}`
      }).join('\n')
    }
  }

  /**
   * Limpiar logs
   */
  clear(): void {
    this.logs = []
  }

  /**
   * Obtener estadísticas de logs
   */
  getStats(): {
    total: number
    byLevel: Record<LogLevel, number>
    errors: number
    warnings: number
  } {
    const byLevel: Record<LogLevel, number> = {
      [LogLevel.DEBUG]: 0,
      [LogLevel.INFO]: 0,
      [LogLevel.WARN]: 0,
      [LogLevel.ERROR]: 0,
    }

    this.logs.forEach(log => {
      byLevel[log.level]++
    })

    return {
      total: this.logs.length,
      byLevel,
      errors: byLevel[LogLevel.ERROR],
      warnings: byLevel[LogLevel.WARN],
    }
  }
}

// Singleton instance
let loggerInstance: Logger | null = null

export function getLogger(sessionId?: string): Logger {
  if (!loggerInstance) {
    loggerInstance = new Logger(sessionId)
  }
  return loggerInstance
}

export default Logger




