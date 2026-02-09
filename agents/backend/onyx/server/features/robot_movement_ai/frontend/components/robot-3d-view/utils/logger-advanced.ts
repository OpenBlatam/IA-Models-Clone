/**
 * Advanced logging system
 * @module robot-3d-view/utils/logger-advanced
 */

/**
 * Log level
 */
export type LogLevel = 'debug' | 'info' | 'warn' | 'error' | 'fatal';

/**
 * Log entry
 */
export interface LogEntry {
  level: LogLevel;
  message: string;
  timestamp: number;
  context?: Record<string, unknown>;
  stack?: string;
  category?: string;
}

/**
 * Log handler
 */
export type LogHandler = (entry: LogEntry) => void;

/**
 * Advanced Logger class
 */
export class AdvancedLogger {
  private handlers: LogHandler[] = [];
  private minLevel: LogLevel = 'debug';
  private buffer: LogEntry[] = [];
  private maxBufferSize = 1000;
  private enabled = true;

  /**
   * Sets minimum log level
   */
  setMinLevel(level: LogLevel): void {
    this.minLevel = level;
  }

  /**
   * Adds a log handler
   */
  addHandler(handler: LogHandler): void {
    this.handlers.push(handler);
  }

  /**
   * Removes a log handler
   */
  removeHandler(handler: LogHandler): void {
    const index = this.handlers.indexOf(handler);
    if (index >= 0) {
      this.handlers.splice(index, 1);
    }
  }

  /**
   * Enables/disables logging
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
  }

  /**
   * Logs a message
   */
  private log(
    level: LogLevel,
    message: string,
    context?: Record<string, unknown>,
    category?: string
  ): void {
    if (!this.enabled) return;

    const levels: LogLevel[] = ['debug', 'info', 'warn', 'error', 'fatal'];
    const currentLevelIndex = levels.indexOf(level);
    const minLevelIndex = levels.indexOf(this.minLevel);

    if (currentLevelIndex < minLevelIndex) return;

    const entry: LogEntry = {
      level,
      message,
      timestamp: Date.now(),
      context,
      category,
      stack: level === 'error' || level === 'fatal' ? new Error().stack : undefined,
    };

    // Add to buffer
    this.buffer.push(entry);
    if (this.buffer.length > this.maxBufferSize) {
      this.buffer.shift();
    }

    // Call handlers
    this.handlers.forEach((handler) => {
      try {
        handler(entry);
      } catch (error) {
        // Prevent handler errors from breaking logging
        console.error('Log handler error:', error);
      }
    });
  }

  /**
   * Logs debug message
   */
  debug(message: string, context?: Record<string, unknown>, category?: string): void {
    this.log('debug', message, context, category);
  }

  /**
   * Logs info message
   */
  info(message: string, context?: Record<string, unknown>, category?: string): void {
    this.log('info', message, context, category);
  }

  /**
   * Logs warning message
   */
  warn(message: string, context?: Record<string, unknown>, category?: string): void {
    this.log('warn', message, context, category);
  }

  /**
   * Logs error message
   */
  error(message: string, context?: Record<string, unknown>, category?: string): void {
    this.log('error', message, context, category);
  }

  /**
   * Logs fatal message
   */
  fatal(message: string, context?: Record<string, unknown>, category?: string): void {
    this.log('fatal', message, context, category);
  }

  /**
   * Gets log buffer
   */
  getBuffer(): readonly LogEntry[] {
    return [...this.buffer];
  }

  /**
   * Gets logs by level
   */
  getLogsByLevel(level: LogLevel): LogEntry[] {
    return this.buffer.filter((entry) => entry.level === level);
  }

  /**
   * Gets logs by category
   */
  getLogsByCategory(category: string): LogEntry[] {
    return this.buffer.filter((entry) => entry.category === category);
  }

  /**
   * Clears log buffer
   */
  clear(): void {
    this.buffer = [];
  }

  /**
   * Exports logs
   */
  export(): string {
    return JSON.stringify(this.buffer, null, 2);
  }

  /**
   * Groups logs by category
   */
  getLogsGroupedByCategory(): Map<string, LogEntry[]> {
    const grouped = new Map<string, LogEntry[]>();

    this.buffer.forEach((entry) => {
      const category = entry.category || 'uncategorized';
      const logs = grouped.get(category) || [];
      logs.push(entry);
      grouped.set(category, logs);
    });

    return grouped;
  }
}

/**
 * Console log handler
 */
export const consoleLogHandler: LogHandler = (entry) => {
  const timestamp = new Date(entry.timestamp).toISOString();
  const prefix = `[${timestamp}] [${entry.level.toUpperCase()}]`;
  const message = entry.category
    ? `${prefix} [${entry.category}] ${entry.message}`
    : `${prefix} ${entry.message}`;

  switch (entry.level) {
    case 'debug':
      console.debug(message, entry.context || '');
      break;
    case 'info':
      console.info(message, entry.context || '');
      break;
    case 'warn':
      console.warn(message, entry.context || '');
      break;
    case 'error':
    case 'fatal':
      console.error(message, entry.context || '', entry.stack || '');
      break;
  }
};

/**
 * Global logger instance
 */
export const advancedLogger = new AdvancedLogger();

// Add console handler by default
advancedLogger.addHandler(consoleLogHandler);



