/**
 * Hook useOnline
 * ==============
 * 
 * Hook para detectar estado de conexión
 */

import { useState, useEffect } from 'react'
import { isOnline, getConnectionInfo } from '../utils/networkUtils'

export interface UseOnlineResult {
  isOnline: boolean
  connectionInfo: {
    effectiveType?: string
    downlink?: number
    rtt?: number
    saveData?: boolean
  } | null
  isSlowConnection: boolean
  isSaveDataMode: boolean
}

/**
 * Hook para detectar estado de conexión
 */
export function useOnline(): UseOnlineResult {
  const [online, setOnline] = useState(() => isOnline())
  const [connectionInfo, setConnectionInfo] = useState(() => getConnectionInfo())

  useEffect(() => {
    const handleOnline = () => {
      setOnline(true)
      setConnectionInfo(getConnectionInfo())
    }

    const handleOffline = () => {
      setOnline(false)
      setConnectionInfo(null)
    }

    const handleConnectionChange = () => {
      setConnectionInfo(getConnectionInfo())
    }

    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)

    const connection = (navigator as any).connection || 
                       (navigator as any).mozConnection || 
                       (navigator as any).webkitConnection

    if (connection) {
      connection.addEventListener('change', handleConnectionChange)
    }

    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
      if (connection) {
        connection.removeEventListener('change', handleConnectionChange)
      }
    }
  }, [])

  const isSlowConnection = connectionInfo?.effectiveType === 'slow-2g' || 
                          connectionInfo?.effectiveType === '2g' || false
  const isSaveDataMode = connectionInfo?.saveData === true

  return {
    isOnline: online,
    connectionInfo,
    isSlowConnection,
    isSaveDataMode
  }
}







