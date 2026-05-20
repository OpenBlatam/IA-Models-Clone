/**
 * Componente NetworkStatus
 * ========================
 * 
 * Componente para mostrar el estado de la conexión
 */

'use client'

import { useOnline } from '@/lib/hooks/useOnline'
import { Wifi, WifiOff, AlertCircle } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

export function NetworkStatus() {
  const { isOnline, isSlowConnection, isSaveDataMode, connectionInfo } = useOnline()

  if (isOnline && !isSlowConnection && !isSaveDataMode) {
    return null
  }

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        className="fixed top-4 left-1/2 -translate-x-1/2 z-50"
      >
        <div className={`
          flex items-center gap-2 px-4 py-2 rounded-lg shadow-lg
          ${!isOnline 
            ? 'bg-red-100 dark:bg-red-900/20 border border-red-300 dark:border-red-800' 
            : isSlowConnection || isSaveDataMode
            ? 'bg-yellow-100 dark:bg-yellow-900/20 border border-yellow-300 dark:border-yellow-800'
            : 'bg-blue-100 dark:bg-blue-900/20 border border-blue-300 dark:border-blue-800'
          }
        `}>
          {!isOnline ? (
            <>
              <WifiOff className="w-5 h-5 text-red-600 dark:text-red-400" />
              <span className="text-sm font-medium text-red-900 dark:text-red-100">
                Sin conexión a internet
              </span>
            </>
          ) : isSlowConnection ? (
            <>
              <AlertCircle className="w-5 h-5 text-yellow-600 dark:text-yellow-400" />
              <span className="text-sm font-medium text-yellow-900 dark:text-yellow-100">
                Conexión lenta detectada
              </span>
            </>
          ) : isSaveDataMode ? (
            <>
              <AlertCircle className="w-5 h-5 text-yellow-600 dark:text-yellow-400" />
              <span className="text-sm font-medium text-yellow-900 dark:text-yellow-100">
                Modo ahorro de datos activo
              </span>
            </>
          ) : (
            <>
              <Wifi className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              <span className="text-sm font-medium text-blue-900 dark:text-blue-100">
                Conectado
                {connectionInfo?.effectiveType && (
                  <span className="ml-2 text-xs">
                    ({connectionInfo.effectiveType})
                  </span>
                )}
              </span>
            </>
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  )
}







