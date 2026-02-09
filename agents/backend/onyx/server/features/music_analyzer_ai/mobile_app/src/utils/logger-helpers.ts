/**
 * Logger utility functions
 * Logging and debugging utilities
 */

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

interface LogEntry {
  level: LogLevel;
  message: string;
  timestamp: number;
  data?: unknown;
}

class Logger {
  private logs: LogEntry[] = [];
  private maxLogs = 100;
  private enabled = true;

  log(level: LogLevel, message: string, data?: unknown): void {
    if (!this.enabled) return;

    const entry: LogEntry = {
      level,
      message,
      timestamp: Date.now(),
      data,
    };

    this.logs.push(entry);

    // Keep only last maxLogs entries
    if (this.logs.length > this.maxLogs) {
      this.logs.shift();
    }

    // Console output
    const logMethod = level === 'error' ? console.error : level === 'warn' ? console.warn : console.log;
    logMethod(`[${level.toUpperCase()}] ${message}`, data || '');
  }

  debug(message: string, data?: unknown): void {
    this.log('debug', message, data);
  }

  info(message: string, data?: unknown): void {
    this.log('info', message, data);
  }

  warn(message: string, data?: unknown): void {
    this.log('warn', message, data);
  }

  error(message: string, data?: unknown): void {
    this.log('error', message, data);
  }

  getLogs(level?: LogLevel): LogEntry[] {
    if (level) {
      return this.logs.filter((log) => log.level === level);
    }
    return [...this.logs];
  }

  clearLogs(): void {
    this.logs = [];
  }

  enable(): void {
    this.enabled = true;
  }

  disable(): void {
    this.enabled = false;
  }

  setMaxLogs(max: number): void {
    this.maxLogs = max;
  }
}

export const logger = new Logger();

/**
 * Create scoped logger
 */
export function createLogger(scope: string) {
  return {
    debug: (message: string, data?: unknown) => logger.debug(`[${scope}] ${message}`, data),
    info: (message: string, data?: unknown) => logger.info(`[${scope}] ${message}`, data),
    warn: (message: string, data?: unknown) => logger.warn(`[${scope}] ${message}`, data),
    error: (message: string, data?: unknown) => logger.error(`[${scope}] ${message}`, data),
  };
}

