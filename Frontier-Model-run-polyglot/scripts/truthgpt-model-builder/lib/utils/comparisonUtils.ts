/**
 * Utilidades de Comparación
 * =========================
 * 
 * Funciones para comparar valores y objetos
 */

/**
 * Compara dos valores profundamente
 */
export function deepEqual(a: any, b: any): boolean {
  if (a === b) return true

  if (a == null || b == null) return false
  if (typeof a !== typeof b) return false

  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false
    return a.every((item, index) => deepEqual(item, b[index]))
  }

  if (typeof a === 'object') {
    const keysA = Object.keys(a)
    const keysB = Object.keys(b)

    if (keysA.length !== keysB.length) return false

    return keysA.every(key => {
      return keysB.includes(key) && deepEqual(a[key], b[key])
    })
  }

  return false
}

/**
 * Obtiene las diferencias entre dos objetos
 */
export function getObjectDiff<T extends Record<string, any>>(
  oldObj: T,
  newObj: T
): Partial<T> {
  const diff: Partial<T> = {}

  const allKeys = new Set([...Object.keys(oldObj), ...Object.keys(newObj)])

  for (const key of allKeys) {
    const oldValue = oldObj[key]
    const newValue = newObj[key]

    if (!deepEqual(oldValue, newValue)) {
      diff[key as keyof T] = newValue
    }
  }

  return diff
}

/**
 * Verifica si un objeto tiene cambios
 */
export function hasChanges<T extends Record<string, any>>(
  oldObj: T,
  newObj: T
): boolean {
  return Object.keys(getObjectDiff(oldObj, newObj)).length > 0
}

/**
 * Compara dos arrays y encuentra elementos únicos
 */
export function arrayDiff<T>(
  array1: T[],
  array2: T[],
  compareFn?: (a: T, b: T) => boolean
): {
  added: T[]
  removed: T[]
  unchanged: T[]
} {
  const compare = compareFn || ((a, b) => a === b)

  const added = array2.filter(
    item2 => !array1.some(item1 => compare(item1, item2))
  )

  const removed = array1.filter(
    item1 => !array2.some(item2 => compare(item1, item2))
  )

  const unchanged = array1.filter(item1 =>
    array2.some(item2 => compare(item1, item2))
  )

  return { added, removed, unchanged }
}

/**
 * Compara dos números con tolerancia
 */
export function compareNumbers(
  a: number,
  b: number,
  tolerance: number = 0.0001
): number {
  const diff = Math.abs(a - b)
  if (diff <= tolerance) return 0
  return a < b ? -1 : 1
}

/**
 * Compara dos fechas
 */
export function compareDates(a: Date | number, b: Date | number): number {
  const timeA = typeof a === 'number' ? a : a.getTime()
  const timeB = typeof b === 'number' ? b : b.getTime()
  return timeA - timeB
}

/**
 * Compara dos strings ignorando mayúsculas/minúsculas
 */
export function compareStringsIgnoreCase(a: string, b: string): number {
  return a.localeCompare(b, undefined, { sensitivity: 'base' })
}

/**
 * Crea una función de comparación para ordenar
 */
export function createComparator<T>(
  getValue: (item: T) => any,
  direction: 'asc' | 'desc' = 'asc'
): (a: T, b: T) => number {
  return (a, b) => {
    const valueA = getValue(a)
    const valueB = getValue(b)

    if (valueA === valueB) return 0
    if (valueA == null) return 1
    if (valueB == null) return -1

    let comparison = 0

    if (typeof valueA === 'number' && typeof valueB === 'number') {
      comparison = valueA - valueB
    } else if (typeof valueA === 'string' && typeof valueB === 'string') {
      comparison = valueA.localeCompare(valueB)
    } else if (valueA instanceof Date && valueB instanceof Date) {
      comparison = valueA.getTime() - valueB.getTime()
    } else {
      comparison = String(valueA).localeCompare(String(valueB))
    }

    return direction === 'asc' ? comparison : -comparison
  }
}

/**
 * Encuentra el valor mínimo en un array
 */
export function findMin<T>(
  array: T[],
  getValue: (item: T) => number
): T | null {
  if (array.length === 0) return null

  let min = array[0]
  let minValue = getValue(min)

  for (let i = 1; i < array.length; i++) {
    const value = getValue(array[i])
    if (value < minValue) {
      min = array[i]
      minValue = value
    }
  }

  return min
}

/**
 * Encuentra el valor máximo en un array
 */
export function findMax<T>(
  array: T[],
  getValue: (item: T) => number
): T | null {
  if (array.length === 0) return null

  let max = array[0]
  let maxValue = getValue(max)

  for (let i = 1; i < array.length; i++) {
    const value = getValue(array[i])
    if (value > maxValue) {
      max = array[i]
      maxValue = value
    }
  }

  return max
}

/**
 * Calcula la similitud entre dos strings (Jaro-Winkler simplificado)
 */
export function stringSimilarity(a: string, b: string): number {
  if (a === b) return 1
  if (a.length === 0 || b.length === 0) return 0

  const longer = a.length > b.length ? a : b
  const shorter = a.length > b.length ? b : a

  if (longer.length === 0) return 1

  const distance = levenshteinDistance(longer, shorter)
  return (longer.length - distance) / longer.length
}

/**
 * Calcula la distancia de Levenshtein entre dos strings
 */
function levenshteinDistance(a: string, b: string): number {
  const matrix: number[][] = []

  for (let i = 0; i <= b.length; i++) {
    matrix[i] = [i]
  }

  for (let j = 0; j <= a.length; j++) {
    matrix[0][j] = j
  }

  for (let i = 1; i <= b.length; i++) {
    for (let j = 1; j <= a.length; j++) {
      if (b.charAt(i - 1) === a.charAt(j - 1)) {
        matrix[i][j] = matrix[i - 1][j - 1]
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1,
          matrix[i][j - 1] + 1,
          matrix[i - 1][j] + 1
        )
      }
    }
  }

  return matrix[b.length][a.length]
}






