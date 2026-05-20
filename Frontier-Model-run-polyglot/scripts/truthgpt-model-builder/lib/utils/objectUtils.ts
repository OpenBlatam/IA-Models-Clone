/**
 * Utilidades para Objetos
 * =======================
 * 
 * Funciones útiles para trabajar con objetos
 */

// ============================================================================
// OPERACIONES BÁSICAS
// ============================================================================

/**
 * Obtiene un valor anidado de un objeto usando una ruta
 */
export function get<T = unknown>(
  obj: Record<string, unknown>,
  path: string,
  defaultValue?: T
): T | undefined {
  const keys = path.split('.')
  let result: unknown = obj

  for (const key of keys) {
    if (result === null || result === undefined) {
      return defaultValue
    }
    result = (result as Record<string, unknown>)[key]
  }

  return (result as T) ?? defaultValue
}

/**
 * Establece un valor anidado en un objeto usando una ruta
 */
export function set(
  obj: Record<string, unknown>,
  path: string,
  value: unknown
): Record<string, unknown> {
  const keys = path.split('.')
  const result = { ...obj }
  let current: Record<string, unknown> = result

  for (let i = 0; i < keys.length - 1; i++) {
    const key = keys[i]
    if (!(key in current) || typeof current[key] !== 'object' || current[key] === null) {
      current[key] = {}
    }
    current = current[key] as Record<string, unknown>
  }

  current[keys[keys.length - 1]] = value
  return result
}

/**
 * Omite propiedades de un objeto
 */
export function omit<T extends Record<string, unknown>, K extends keyof T>(
  obj: T,
  keys: K[]
): Omit<T, K> {
  const result = { ...obj }
  for (const key of keys) {
    delete result[key]
  }
  return result
}

/**
 * Selecciona propiedades específicas de un objeto
 */
export function pick<T extends Record<string, unknown>, K extends keyof T>(
  obj: T,
  keys: K[]
): Pick<T, K> {
  const result = {} as Pick<T, K>
  for (const key of keys) {
    if (key in obj) {
      result[key] = obj[key]
    }
  }
  return result
}

/**
 * Fusiona objetos profundamente
 */
export function merge<T extends Record<string, unknown>>(
  target: T,
  ...sources: Partial<T>[]
): T {
  const result = { ...target }

  for (const source of sources) {
    for (const key in source) {
      if (source[key] !== undefined) {
        if (
          typeof source[key] === 'object' &&
          source[key] !== null &&
          !Array.isArray(source[key]) &&
          typeof result[key] === 'object' &&
          result[key] !== null &&
          !Array.isArray(result[key])
        ) {
          result[key] = merge(
            result[key] as Record<string, unknown>,
            source[key] as Record<string, unknown>
          ) as T[Extract<keyof T, string>]
        } else {
          result[key] = source[key] as T[Extract<keyof T, string>]
        }
      }
    }
  }

  return result
}

// ============================================================================
// TRANSFORMACIONES
// ============================================================================

/**
 * Invierte las claves y valores de un objeto
 */
export function invert<T extends Record<string, string | number>>(
  obj: T
): Record<string, string> {
  const result: Record<string, string> = {}
  for (const key in obj) {
    result[String(obj[key])] = key
  }
  return result
}

/**
 * Mapea los valores de un objeto
 */
export function mapValues<T, U>(
  obj: Record<string, T>,
  fn: (value: T, key: string) => U
): Record<string, U> {
  const result: Record<string, U> = {}
  for (const key in obj) {
    result[key] = fn(obj[key], key)
  }
  return result
}

/**
 * Mapea las claves de un objeto
 */
export function mapKeys<T>(
  obj: Record<string, T>,
  fn: (key: string, value: T) => string
): Record<string, T> {
  const result: Record<string, T> = {}
  for (const key in obj) {
    result[fn(key, obj[key])] = obj[key]
  }
  return result
}

/**
 * Filtra propiedades de un objeto según un predicado
 */
export function filterObject<T extends Record<string, unknown>>(
  obj: T,
  predicate: (value: unknown, key: string) => boolean
): Partial<T> {
  const result: Partial<T> = {}
  for (const key in obj) {
    if (predicate(obj[key], key)) {
      result[key] = obj[key]
    }
  }
  return result
}

// ============================================================================
// UTILIDADES
// ============================================================================

/**
 * Verifica si un objeto está vacío
 */
export function isEmpty(obj: Record<string, unknown>): boolean {
  return Object.keys(obj).length === 0
}

/**
 * Obtiene todas las claves anidadas de un objeto
 */
export function keys(obj: Record<string, unknown>, prefix = ''): string[] {
  const result: string[] = []
  
  for (const key in obj) {
    const fullKey = prefix ? `${prefix}.${key}` : key
    result.push(fullKey)
    
    if (
      typeof obj[key] === 'object' &&
      obj[key] !== null &&
      !Array.isArray(obj[key])
    ) {
      result.push(...keys(obj[key] as Record<string, unknown>, fullKey))
    }
  }
  
  return result
}

/**
 * Crea un objeto desde pares clave-valor
 */
export function fromPairs<T>(pairs: Array<[string, T]>): Record<string, T> {
  const result: Record<string, T> = {}
  for (const [key, value] of pairs) {
    result[key] = value
  }
  return result
}

/**
 * Convierte un objeto a pares clave-valor
 */
export function toPairs<T>(obj: Record<string, T>): Array<[string, T]> {
  return Object.entries(obj)
}







