/**
 * Re-export logger from monitoring module for convenience
 * This provides a simpler API while using the full-featured logger
 */
export { logger, createLogger, LogLevel } from "./monitoring/logger";

/**
 * Convenience functions for common logging operations
 */
export const logError = (message: string, error?: Error, data?: unknown): void => {
  logger.error(message, error, data);
};

export const logWarn = (message: string, data?: unknown, error?: Error): void => {
  logger.warn(message, data, error);
};

export const logInfo = (message: string, data?: unknown): void => {
  logger.info(message, data);
};

export const logDebug = (message: string, data?: unknown): void => {
  logger.debug(message, data);
};

