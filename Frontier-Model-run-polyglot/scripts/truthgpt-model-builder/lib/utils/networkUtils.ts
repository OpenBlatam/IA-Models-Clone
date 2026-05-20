/**
 * Utilidades de Red
 * =================
 * 
 * Funciones para trabajar con la red
 */

/**
 * Verifica si hay conexión a internet
 */
export function isOnline(): boolean {
  if (typeof navigator === 'undefined') return true
  return navigator.onLine
}

/**
 * Obtiene información de la conexión (si está disponible)
 */
export function getConnectionInfo(): {
  effectiveType?: string
  downlink?: number
  rtt?: number
  saveData?: boolean
} | null {
  if (typeof navigator === 'undefined') return null
  
  const connection = (navigator as any).connection || 
                     (navigator as any).mozConnection || 
                     (navigator as any).webkitConnection

  if (!connection) return null

  return {
    effectiveType: connection.effectiveType,
    downlink: connection.downlink,
    rtt: connection.rtt,
    saveData: connection.saveData
  }
}

/**
 * Verifica si la conexión es lenta
 */
export function isSlowConnection(): boolean {
  const info = getConnectionInfo()
  if (!info) return false

  return info.effectiveType === 'slow-2g' || info.effectiveType === '2g'
}

/**
 * Verifica si está en modo ahorro de datos
 */
export function isSaveDataMode(): boolean {
  const info = getConnectionInfo()
  return info?.saveData === true
}

/**
 * Ping a una URL
 */
export async function ping(url: string, timeout: number = 5000): Promise<number> {
  const start = performance.now()
  
  try {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), timeout)

    await fetch(url, {
      method: 'HEAD',
      mode: 'no-cors',
      signal: controller.signal
    })

    clearTimeout(timeoutId)
    return performance.now() - start
  } catch {
    return -1
  }
}

/**
 * Verifica si una URL es accesible
 */
export async function isURLAccessible(url: string, timeout: number = 5000): Promise<boolean> {
  try {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), timeout)

    const response = await fetch(url, {
      method: 'HEAD',
      signal: controller.signal
    })

    clearTimeout(timeoutId)
    return response.ok
  } catch {
    return false
  }
}







