/**
 * Hook useLogger
 * ==============
 * 
 * Hook para logging en componentes
 */

import { useMemo } from 'react'
import { createLogger, Logger } from '../utils/loggerUtils'

/**
 * Hook para obtener un logger con contexto del componente
 */
export function useLogger(context?: string): Logger {
  return useMemo(() => {
    return createLogger(context || 'Component', {
      enableConsole: true,
      enableStorage: false
    })
  }, [context])
}






