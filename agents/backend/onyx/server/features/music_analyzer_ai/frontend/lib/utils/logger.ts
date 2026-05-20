/**
 * Logger utility.
 * Provides structured logging with different log levels.
 */

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

interface LogEntry {
  level: LogLevel;
  message: string;
  data?: unknown;
  timestamp: string;
}

/**
 * Logger class for structured logging.
 */
class Logger {
  private isDevelopment: boolean;
  private isProduction: boolean;

  constructor() {
    this.isDevelopment = process.env.NODE_ENV === 'development';
    this.isProduction = process.env.NODE_ENV === 'production';
  }

  /**
   * Formats log entry.
   */
  private formatEntry(level: LogLevel, message: string, data?: unknown): LogEntry {
    return {
      level,
      message,
      data,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Logs debug message.
   */
  debug(message: string, data?: unknown): void {
    if (this.isDevelopment) {
      const entry = this.formatEntry('debug', message, data);
      console.debug(`[DEBUG] ${entry.timestamp} - ${message}`, data || '');
    }
  }

  /**
   * Logs info message.
   */
  info(message: string, data?: unknown): void {
    const entry = this.formatEntry('info', message, data);
    console.info(`[INFO] ${entry.timestamp} - ${message}`, data || '');
  }

  /**
   * Logs warning message.
   */
  warn(message: string, data?: unknown): void {
    const entry = this.formatEntry('warn', message, data);
    console.warn(`[WARN] ${entry.timestamp} - ${message}`, data || '');
  }

  /**
   * Logs error message.
   */
  error(message: string, error?: Error | unknown): void {
    const entry = this.formatEntry('error', message, error);
    console.error(`[ERROR] ${entry.timestamp} - ${message}`, error || '');
    
    // In production, you might want to send to error tracking service
    if (this.isProduction && error instanceof Error) {
      // Example: Send to error tracking service
      // errorTrackingService.captureException(error);
    }
  }

  /**
   * Logs grouped messages.
   */
  group(label: string, callback: () => void): void {
    if (this.isDevelopment) {
      console.group(label);
      callback();
      console.groupEnd();
    } else {
      callback();
    }
  }

  /**
   * Logs table data.
   */
  table(data: unknown): void {
    if (this.isDevelopment) {
      console.table(data);
    }
  }
}

/**
 * Default logger instance.
 */
export const logger = new Logger();

/**
 * Creates a scoped logger with prefix.
 */
export function createLogger(prefix: string) {
  return {
    debug: (message: string, data?: unknown) =>
      logger.debug(`[${prefix}] ${message}`, data),
    info: (message: string, data?: unknown) =>
      logger.info(`[${prefix}] ${message}`, data),
    warn: (message: string, data?: unknown) =>
      logger.warn(`[${prefix}] ${message}`, data),
    error: (message: string, error?: Error | unknown) =>
      logger.error(`[${prefix}] ${message}`, error),
  };
}

