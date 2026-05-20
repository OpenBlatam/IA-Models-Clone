/**
 * Array utility functions
 */

/**
 * Chunk array into smaller arrays
 */
export function chunk<T>(array: T[], size: number): T[][] {
  const chunks: T[][] = []
  for (let i = 0; i < array.length; i += size) {
    chunks.push(array.slice(i, i + size))
  }
  return chunks
}

/**
 * Remove duplicates
 */
export function unique<T>(array: T[]): T[] {
  return Array.from(new Set(array))
}

/**
 * Remove duplicates by key
 */
export function uniqueBy<T>(array: T[], keyFn: (item: T) => any): T[] {
  const seen = new Set()
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
 * Group by key
 */
export function groupBy<T>(array: T[], keyFn: (item: T) => string): Record<string, T[]> {
  return array.reduce((groups, item) => {
    const key = keyFn(item)
    if (!groups[key]) {
      groups[key] = []
    }
    groups[key].push(item)
    return groups
  }, {} as Record<string, T[]>)
}

/**
 * Sort by key
 */
export function sortBy<T>(array: T[], keyFn: (item: T) => any, order: 'asc' | 'desc' = 'asc'): T[] {
  return [...array].sort((a, b) => {
    const aVal = keyFn(a)
    const bVal = keyFn(b)
    const comparison = aVal < bVal ? -1 : aVal > bVal ? 1 : 0
    return order === 'asc' ? comparison : -comparison
  })
}

/**
 * Flatten nested array
 */
export function flatten<T>(array: (T | T[])[]): T[] {
  return array.reduce((acc, item) => {
    return acc.concat(Array.isArray(item) ? flatten(item) : item)
  }, [] as T[])
}

/**
 * Shuffle array
 */
export function shuffle<T>(array: T[]): T[] {
  const shuffled = [...array]
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
  }
  return shuffled
}

/**
 * Sample random items
 */
export function sample<T>(array: T[], count: number = 1): T[] {
  const shuffled = shuffle(array)
  return shuffled.slice(0, Math.min(count, array.length))
}

/**
 * Partition array
 */
export function partition<T>(array: T[], predicate: (item: T) => boolean): [T[], T[]] {
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
 * Intersection of arrays
 */
export function intersection<T>(...arrays: T[][]): T[] {
  if (arrays.length === 0) return []
  if (arrays.length === 1) return arrays[0]
  
  return arrays.reduce((acc, arr) => {
    return acc.filter(item => arr.includes(item))
  })
}

/**
 * Union of arrays
 */
export function union<T>(...arrays: T[][]): T[] {
  return unique(flatten(arrays))
}

/**
 * Difference (items in first but not in others)
 */
export function difference<T>(array: T[], ...others: T[][]): T[] {
  const otherSet = new Set(flatten(others))
  return array.filter(item => !otherSet.has(item))
}

/**
 * Zip arrays
 */
export function zip<T, U>(array1: T[], array2: U[]): Array<[T, U]> {
  const length = Math.min(array1.length, array2.length)
  const result: Array<[T, U]> = []
  for (let i = 0; i < length; i++) {
    result.push([array1[i], array2[i]])
  }
  return result
}

/**
 * Unzip array of tuples
 */
export function unzip<T, U>(array: Array<[T, U]>): [T[], U[]] {
  const array1: T[] = []
  const array2: U[] = []
  
  for (const [item1, item2] of array) {
    array1.push(item1)
    array2.push(item2)
  }
  
  return [array1, array2]
}

/**
 * Range of numbers
 */
export function range(start: number, end: number, step: number = 1): number[] {
  const result: number[] = []
  if (step > 0) {
    for (let i = start; i < end; i += step) {
      result.push(i)
    }
  } else {
    for (let i = start; i > end; i += step) {
      result.push(i)
    }
  }
  return result
}

/**
 * Move item in array
 */
export function move<T>(array: T[], fromIndex: number, toIndex: number): T[] {
  const result = [...array]
  const [item] = result.splice(fromIndex, 1)
  result.splice(toIndex, 0, item)
  return result
}

/**
 * Swap items in array
 */
export function swap<T>(array: T[], index1: number, index2: number): T[] {
  const result = [...array]
  ;[result[index1], result[index2]] = [result[index2], result[index1]]
  return result
}

/**
 * Remove item by index
 */
export function removeAt<T>(array: T[], index: number): T[] {
  return array.filter((_, i) => i !== index)
}

/**
 * Insert item at index
 */
export function insertAt<T>(array: T[], index: number, item: T): T[] {
  const result = [...array]
  result.splice(index, 0, item)
  return result
}

/**
 * Find index by predicate
 */
export function findIndex<T>(array: T[], predicate: (item: T, index: number) => boolean): number {
  for (let i = 0; i < array.length; i++) {
    if (predicate(array[i], i)) {
      return i
    }
  }
  return -1
}

/**
 * Find all indices by predicate
 */
export function findAllIndices<T>(array: T[], predicate: (item: T, index: number) => boolean): number[] {
  const indices: number[] = []
  for (let i = 0; i < array.length; i++) {
    if (predicate(array[i], i)) {
      indices.push(i)
    }
  }
  return indices
}




