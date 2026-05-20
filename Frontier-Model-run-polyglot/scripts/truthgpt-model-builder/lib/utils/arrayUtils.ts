/**
 * Utilidades para Arrays
 * ======================
 * 
 * Funciones útiles para trabajar con arrays
 */

// ============================================================================
// OPERACIONES BÁSICAS
// ============================================================================

/**
 * Agrupa elementos de un array por una clave
 */
export function groupBy<T>(
  array: T[],
  keyFn: (item: T) => string | number
): Record<string, T[]> {
  return array.reduce((groups, item) => {
    const key = String(keyFn(item))
    if (!groups[key]) {
      groups[key] = []
    }
    groups[key].push(item)
    return groups
  }, {} as Record<string, T[]>)
}

/**
 * Ordena un array por múltiples criterios
 */
export function sortBy<T>(
  array: T[],
  ...sortFns: Array<(item: T) => number | string>
): T[] {
  return [...array].sort((a, b) => {
    for (const sortFn of sortFns) {
      const aVal = sortFn(a)
      const bVal = sortFn(b)
      
      if (aVal < bVal) return -1
      if (aVal > bVal) return 1
    }
    return 0
  })
}

/**
 * Elimina duplicados de un array
 */
export function unique<T>(array: T[]): T[] {
  return Array.from(new Set(array))
}

/**
 * Elimina duplicados usando una función de comparación
 */
export function uniqueBy<T>(
  array: T[],
  keyFn: (item: T) => string | number
): T[] {
  const seen = new Set<string | number>()
  return array.filter(item => {
    const key = keyFn(item)
    if (seen.has(key)) {
      return false
    }
    seen.add(key)
    return true
  })
}

/**
 * Divide un array en chunks
 */
export function chunk<T>(array: T[], size: number): T[][] {
  const chunks: T[][] = []
  for (let i = 0; i < array.length; i += size) {
    chunks.push(array.slice(i, i + size))
  }
  return chunks
}

/**
 * Aplana un array anidado
 */
export function flatten<T>(array: (T | T[])[]): T[] {
  return array.reduce((acc, item) => {
    return acc.concat(Array.isArray(item) ? flatten(item) : item)
  }, [] as T[])
}

// ============================================================================
// OPERACIONES AVANZADAS
// ============================================================================

/**
 * Obtiene la diferencia entre dos arrays
 */
export function difference<T>(array1: T[], array2: T[]): T[] {
  const set2 = new Set(array2)
  return array1.filter(item => !set2.has(item))
}

/**
 * Obtiene la intersección de dos arrays
 */
export function intersection<T>(array1: T[], array2: T[]): T[] {
  const set2 = new Set(array2)
  return array1.filter(item => set2.has(item))
}

/**
 * Obtiene la unión de dos arrays
 */
export function union<T>(array1: T[], array2: T[]): T[] {
  return unique([...array1, ...array2])
}

/**
 * Particiona un array en dos según un predicado
 */
export function partition<T>(
  array: T[],
  predicate: (item: T) => boolean
): [T[], T[]] {
  const truthy: T[] = []
  const falsy: T[] = []

  for (const item of array) {
    if (predicate(item)) {
      truthy.push(item)
    } else {
      falsy.push(item)
    }
  }

  return [truthy, falsy]
}

/**
 * Obtiene el elemento con el valor máximo según una función
 */
export function maxBy<T>(
  array: T[],
  fn: (item: T) => number
): T | undefined {
  if (array.length === 0) return undefined

  let maxItem = array[0]
  let maxValue = fn(maxItem)

  for (const item of array) {
    const value = fn(item)
    if (value > maxValue) {
      maxValue = value
      maxItem = item
    }
  }

  return maxItem
}

/**
 * Obtiene el elemento con el valor mínimo según una función
 */
export function minBy<T>(
  array: T[],
  fn: (item: T) => number
): T | undefined {
  if (array.length === 0) return undefined

  let minItem = array[0]
  let minValue = fn(minItem)

  for (const item of array) {
    const value = fn(item)
    if (value < minValue) {
      minValue = value
      minItem = item
    }
  }

  return minItem
}

/**
 * Calcula la suma de valores en un array
 */
export function sumBy<T>(array: T[], fn: (item: T) => number): number {
  return array.reduce((sum, item) => sum + fn(item), 0)
}

/**
 * Calcula el promedio de valores en un array
 */
export function averageBy<T>(array: T[], fn: (item: T) => number): number {
  if (array.length === 0) return 0
  return sumBy(array, fn) / array.length
}

// ============================================================================
// UTILIDADES DE BÚSQUEDA
// ============================================================================

/**
 * Encuentra el índice del último elemento que cumple un predicado
 */
export function findLastIndex<T>(
  array: T[],
  predicate: (item: T, index: number) => boolean
): number {
  for (let i = array.length - 1; i >= 0; i--) {
    if (predicate(array[i], i)) {
      return i
    }
  }
  return -1
}

/**
 * Obtiene un elemento aleatorio de un array
 */
export function sample<T>(array: T[]): T | undefined {
  if (array.length === 0) return undefined
  return array[Math.floor(Math.random() * array.length)]
}

/**
 * Obtiene múltiples elementos aleatorios de un array
 */
export function sampleSize<T>(array: T[], size: number): T[] {
  const shuffled = [...array].sort(() => Math.random() - 0.5)
  return shuffled.slice(0, size)
}

/**
 * Mezcla un array (Fisher-Yates shuffle)
 */
export function shuffle<T>(array: T[]): T[] {
  const shuffled = [...array]
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
  }
  return shuffled
}







