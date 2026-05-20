/**
 * Utilidades de Almacenamiento
 * ============================
 * 
 * Funciones mejoradas para localStorage y sessionStorage
 */

/**
 * Obtiene un valor del localStorage con tipo seguro
 */
export function getLocalStorage<T>(key: string, defaultValue: T | null = null): T | null {
  if (typeof window === 'undefined') return defaultValue

  try {
    const item = window.localStorage.getItem(key)
    if (item === null) return defaultValue
    return JSON.parse(item) as T
  } catch {
    return defaultValue
  }
}

/**
 * Establece un valor en el localStorage con manejo de errores
 */
export function setLocalStorage<T>(key: string, value: T): boolean {
  if (typeof window === 'undefined') return false

  try {
    window.localStorage.setItem(key, JSON.stringify(value))
    return true
  } catch (error) {
    // QuotaExceededError o otros errores
    console.error('Error al guardar en localStorage:', error)
    return false
  }
}

/**
 * Elimina un valor del localStorage
 */
export function removeLocalStorage(key: string): boolean {
  if (typeof window === 'undefined') return false

  try {
    window.localStorage.removeItem(key)
    return true
  } catch {
    return false
  }
}

/**
 * Limpia todo el localStorage
 */
export function clearLocalStorage(): boolean {
  if (typeof window === 'undefined') return false

  try {
    window.localStorage.clear()
    return true
  } catch {
    return false
  }
}

/**
 * Obtiene todas las claves del localStorage
 */
export function getAllLocalStorageKeys(): string[] {
  if (typeof window === 'undefined') return []

  try {
    return Object.keys(window.localStorage)
  } catch {
    return []
  }
}

/**
 * Obtiene el tamaño aproximado del localStorage usado
 */
export function getLocalStorageSize(): number {
  if (typeof window === 'undefined') return 0

  let total = 0
  try {
    for (const key in window.localStorage) {
      if (window.localStorage.hasOwnProperty(key)) {
        total += window.localStorage[key].length + key.length
      }
    }
  } catch {
    // Ignorar errores
  }
  return total
}

/**
 * Obtiene el tamaño máximo del localStorage
 */
export function getLocalStorageQuota(): number {
  // Típicamente 5-10MB dependiendo del navegador
  return 5 * 1024 * 1024 // 5MB
}

/**
 * Verifica si hay espacio disponible en localStorage
 */
export function hasLocalStorageSpace(requiredBytes: number): boolean {
  const used = getLocalStorageSize()
  const quota = getLocalStorageQuota()
  return (quota - used) >= requiredBytes
}

// Funciones similares para sessionStorage

/**
 * Obtiene un valor del sessionStorage con tipo seguro
 */
export function getSessionStorage<T>(key: string, defaultValue: T | null = null): T | null {
  if (typeof window === 'undefined') return defaultValue

  try {
    const item = window.sessionStorage.getItem(key)
    if (item === null) return defaultValue
    return JSON.parse(item) as T
  } catch {
    return defaultValue
  }
}

/**
 * Establece un valor en el sessionStorage
 */
export function setSessionStorage<T>(key: string, value: T): boolean {
  if (typeof window === 'undefined') return false

  try {
    window.sessionStorage.setItem(key, JSON.stringify(value))
    return true
  } catch (error) {
    console.error('Error al guardar en sessionStorage:', error)
    return false
  }
}

/**
 * Elimina un valor del sessionStorage
 */
export function removeSessionStorage(key: string): boolean {
  if (typeof window === 'undefined') return false

  try {
    window.sessionStorage.removeItem(key)
    return true
  } catch {
    return false
  }
}

/**
 * Limpia todo el sessionStorage
 */
export function clearSessionStorage(): boolean {
  if (typeof window === 'undefined') return false

  try {
    window.sessionStorage.clear()
    return true
  } catch {
    return false
  }
}

/**
 * Almacenamiento con expiración (TTL)
 */
export interface StorageItem<T> {
  value: T
  expires: number
}

/**
 * Establece un valor con expiración en localStorage
 */
export function setLocalStorageWithTTL<T>(key: string, value: T, ttlMs: number): boolean {
  const item: StorageItem<T> = {
    value,
    expires: Date.now() + ttlMs
  }
  return setLocalStorage(key, item)
}

/**
 * Obtiene un valor con expiración del localStorage
 */
export function getLocalStorageWithTTL<T>(key: string, defaultValue: T | null = null): T | null {
  const item = getLocalStorage<StorageItem<T>>(key, null)
  
  if (!item) return defaultValue

  if (Date.now() > item.expires) {
    removeLocalStorage(key)
    return defaultValue
  }

  return item.value
}

/**
 * Limpia elementos expirados del localStorage
 */
export function cleanExpiredLocalStorage(): number {
  if (typeof window === 'undefined') return 0

  let cleaned = 0
  const keys = getAllLocalStorageKeys()

  for (const key of keys) {
    try {
      const item = getLocalStorage<StorageItem<any>>(key, null)
      if (item && 'expires' in item && Date.now() > item.expires) {
        removeLocalStorage(key)
        cleaned++
      }
    } catch {
      // Ignorar errores
    }
  }

  return cleaned
}






