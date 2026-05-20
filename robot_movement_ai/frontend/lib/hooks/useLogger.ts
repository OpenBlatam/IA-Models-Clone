import { useCallback } from 'react';
import { logger } from '@/lib/utils/logger';

export function useLogger() {
  const logDebug = useCallback((message: string, data?: unknown) => {
    logger.debug(message, data);
  }, []);

  const logInfo = useCallback((message: string, data?: unknown) => {
    logger.info(message, data);
  }, []);

  const logWarn = useCallback((message: string, data?: unknown) => {
    logger.warn(message, data);
  }, []);

  const logError = useCallback((message: string, data?: unknown) => {
    logger.error(message, data);
  }, []);

  return {
    logDebug,
    logInfo,
    logWarn,
    logError,
    getLogs: logger.getLogs.bind(logger),
    clearLogs: logger.clearLogs.bind(logger),
    exportLogs: logger.exportLogs.bind(logger),
  };
}



