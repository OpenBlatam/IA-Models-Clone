/**
 * Logger utility for 3D view debugging
 * @module robot-3d-view/utils/logger
 */

/**
 * Log levels
 */
type LogLevel = 'debug' | 'info' | 'warn' | 'error';

/**
 * Logger configuration
 */
interface LoggerConfig {
  enabled: boolean;
  level: LogLevel;
  prefix: string;
}

const defaultConfig: LoggerConfig = {
  enabled: process.env.NODE_ENV === 'development',
  level: 'info',
  prefix: '[Robot3DView]',
};

const logLevels: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

/**
 * Logger class for 3D view
 */
class Logger {
  private config: LoggerConfig;

  constructor(config: Partial<LoggerConfig> = {}) {
    this.config = { ...defaultConfig, ...config };
  }

  private shouldLog(level: LogLevel): boolean {
    if (!this.config.enabled) return false;
    return logLevels[level] >= logLevels[this.config.level];
  }

  debug(...args: unknown[]): void {
    if (this.shouldLog('debug')) {
      console.debug(this.config.prefix, ...args);
    }
  }

  info(...args: unknown[]): void {
    if (this.shouldLog('info')) {
      console.info(this.config.prefix, ...args);
    }
  }

  warn(...args: unknown[]): void {
    if (this.shouldLog('warn')) {
      console.warn(this.config.prefix, ...args);
    }
  }

  error(...args: unknown[]): void {
    if (this.shouldLog('error')) {
      console.error(this.config.prefix, ...args);
    }
  }
}

/**
 * Default logger instance
 */
export const logger = new Logger();

/**
 * Creates a custom logger instance
 * 
 * @param config - Logger configuration
 * @returns Logger instance
 */
export function createLogger(config: Partial<LoggerConfig> = {}): Logger {
  return new Logger(config);
}



