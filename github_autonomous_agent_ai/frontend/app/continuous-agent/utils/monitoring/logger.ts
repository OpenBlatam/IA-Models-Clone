/**
 * Logging utilities for Continuous Agent module
 * 
 * Provides structured logging with different log levels and contexts
 */

/**
 * Log levels
 */
export enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3,
}

/**
 * Log entry structure
 */
export interface LogEntry {
  readonly level: LogLevel;
  readonly message: string;
  readonly context?: string;
  readonly data?: unknown;
  readonly timestamp: string;
  readonly error?: Error;
}

/**
 * Logger configuration
 */
export interface LoggerConfig {
  readonly minLevel: LogLevel;
  readonly enableConsole: boolean;
  readonly enableRemote?: boolean;
  readonly remoteEndpoint?: string;
  readonly context?: string;
}

const DEFAULT_CONFIG: LoggerConfig = {
  minLevel: LogLevel.INFO,
  enableConsole: true,
  enableRemote: false,
};

/**
 * Logger class
 */
class Logger {
  private config: LoggerConfig;

  constructor(config: Partial<LoggerConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Creates a log entry
   */
  private createEntry(
    level: LogLevel,
    message: string,
    data?: unknown,
    error?: Error
  ): LogEntry {
    return {
      level,
      message,
      context: this.config.context,
      data,
      error,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Logs a message
   */
  private log(entry: LogEntry): void {
    if (entry.level < this.config.minLevel) {
      return;
    }

    if (this.config.enableConsole) {
      this.logToConsole(entry);
    }

    if (this.config.enableRemote && this.config.remoteEndpoint) {
      this.logToRemote(entry);
    }
  }

  /**
   * Logs to console
   */
  private logToConsole(entry: LogEntry): void {
    const { level, message, context, data, error, timestamp } = entry;
    const prefix = context ? `[${context}]` : "";
    const logMessage = `${timestamp} ${prefix} ${message}`;

    switch (level) {
      case LogLevel.DEBUG:
        console.debug(logMessage, data, error);
        break;
      case LogLevel.INFO:
        console.info(logMessage, data);
        break;
      case LogLevel.WARN:
        console.warn(logMessage, data, error);
        break;
      case LogLevel.ERROR:
        console.error(logMessage, data, error);
        break;
    }
  }

  /**
   * Logs to remote endpoint
   */
  private async logToRemote(entry: LogEntry): Promise<void> {
    if (!this.config.remoteEndpoint) {
      return;
    }

    try {
      await fetch(this.config.remoteEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(entry),
      });
    } catch (error) {
      // Silently fail remote logging
      console.error("Failed to log to remote endpoint:", error);
    }
  }

  /**
   * Debug log
   */
  debug(message: string, data?: unknown): void {
    this.log(this.createEntry(LogLevel.DEBUG, message, data));
  }

  /**
   * Info log
   */
  info(message: string, data?: unknown): void {
    this.log(this.createEntry(LogLevel.INFO, message, data));
  }

  /**
   * Warning log
   */
  warn(message: string, data?: unknown, error?: Error): void {
    this.log(this.createEntry(LogLevel.WARN, message, data, error));
  }

  /**
   * Error log
   */
  error(message: string, error?: Error, data?: unknown): void {
    this.log(this.createEntry(LogLevel.ERROR, message, data, error));
  }
}

/**
 * Default logger instance
 */
export const logger = new Logger({
  context: "ContinuousAgent",
  minLevel: process.env.NODE_ENV === "development" ? LogLevel.DEBUG : LogLevel.INFO,
});

/**
 * Creates a logger with custom context
 */
export function createLogger(context: string, config?: Partial<LoggerConfig>): Logger {
  return new Logger({ ...config, context });
}

/**
 * Performance monitoring utilities
 */
export class PerformanceMonitor {
  private measurements: Map<string, number> = new Map();

  /**
   * Starts a performance measurement
   */
  start(label: string): void {
    if (typeof performance !== "undefined" && performance.mark) {
      performance.mark(`${label}-start`);
    }
    this.measurements.set(label, performance.now());
  }

  /**
   * Ends a performance measurement and returns duration
   */
  end(label: string): number {
    const startTime = this.measurements.get(label);
    if (!startTime) {
      logger.warn(`Performance measurement "${label}" was not started`);
      return 0;
    }

    const duration = performance.now() - startTime;
    this.measurements.delete(label);

    if (typeof performance !== "undefined" && performance.measure) {
      performance.mark(`${label}-end`);
      performance.measure(label, `${label}-start`, `${label}-end`);
    }

    logger.debug(`Performance: ${label} took ${duration.toFixed(2)}ms`);
    return duration;
  }

  /**
   * Measures async function execution time
   */
  async measure<T>(label: string, fn: () => Promise<T>): Promise<T> {
    this.start(label);
    try {
      const result = await fn();
      this.end(label);
      return result;
    } catch (error) {
      this.end(label);
      throw error;
    }
  }
}

/**
 * Default performance monitor
 */
export const performanceMonitor = new PerformanceMonitor();




