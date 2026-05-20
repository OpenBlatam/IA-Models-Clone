/**
 * Logger utility with different log levels
 */

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

interface LogEntry {
  level: LogLevel;
  message: string;
  data?: unknown;
  timestamp: string;
}

class Logger {
  private logs: LogEntry[] = [];
  private maxLogs = 100;
  private enabled = process.env.NODE_ENV === 'development';

  private formatMessage(level: LogLevel, message: string, data?: unknown): string {
    const timestamp = new Date().toISOString();
    const prefix = `[${timestamp}] [${level.toUpperCase()}]`;
    return data ? `${prefix} ${message}`, data : `${prefix} ${message}`;
  }

  private addLog(level: LogLevel, message: string, data?: unknown) {
    const entry: LogEntry = {
      level,
      message,
      data,
      timestamp: new Date().toISOString(),
    };

    this.logs.push(entry);
    if (this.logs.length > this.maxLogs) {
      this.logs.shift();
    }
  }

  debug(message: string, data?: unknown) {
    this.addLog('debug', message, data);
    if (this.enabled) {
      console.debug(this.formatMessage('debug', message, data));
    }
  }

  info(message: string, data?: unknown) {
    this.addLog('info', message, data);
    if (this.enabled) {
      console.info(this.formatMessage('info', message, data));
    }
  }

  warn(message: string, data?: unknown) {
    this.addLog('warn', message, data);
    console.warn(this.formatMessage('warn', message, data));
  }

  error(message: string, data?: unknown) {
    this.addLog('error', message, data);
    console.error(this.formatMessage('error', message, data));
  }

  getLogs(level?: LogLevel): LogEntry[] {
    if (level) {
      return this.logs.filter((log) => log.level === level);
    }
    return [...this.logs];
  }

  clearLogs() {
    this.logs = [];
  }

  exportLogs(): string {
    return JSON.stringify(this.logs, null, 2);
  }
}

export const logger = new Logger();



