/**
 * Object utility functions
 */

/**
 * Deep clone object
 */
export function deepClone<T>(obj: T): T {
  if (obj === null || typeof obj !== 'object') {
    return obj
  }

  if (obj instanceof Date) {
    return new Date(obj.getTime()) as any
  }

  if (obj instanceof Array) {
    return obj.map(item => deepClone(item)) as any
  }

  if (obj instanceof Object) {
    const cloned = {} as T
    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        cloned[key] = deepClone(obj[key])
      }
    }
    return cloned
  }

  return obj
}

/**
 * Deep merge objects
 */
export function deepMerge<T extends Record<string, any>>(
  target: T,
  ...sources: Partial<T>[]
): T {
  if (!sources.length) return target

  const source = sources.shift()
  if (!source) return target

  if (isObject(target) && isObject(source)) {
    for (const key in source) {
      if (isObject(source[key])) {
        if (!target[key]) Object.assign(target, { [key]: {} })
        deepMerge(target[key], source[key])
      } else {
        Object.assign(target, { [key]: source[key] })
      }
    }
  }

  return deepMerge(target, ...sources)
}

/**
 * Check if value is object
 */
function isObject(item: any): boolean {
  return item && typeof item === 'object' && !Array.isArray(item)
}

/**
 * Pick properties from object
 */
export function pick<T, K extends keyof T>(obj: T, keys: K[]): Pick<T, K> {
  const result = {} as Pick<T, K>
  for (const key of keys) {
    if (key in obj) {
      result[key] = obj[key]
    }
  }
  return result
}

/**
 * Omit properties from object
 */
export function omit<T, K extends keyof T>(obj: T, keys: K[]): Omit<T, K> {
  const result = { ...obj }
  for (const key of keys) {
    delete result[key]
  }
  return result
}

/**
 * Get nested property value
 */
export function getNestedValue<T>(obj: any, path: string, defaultValue?: T): T | undefined {
  const keys = path.split('.')
  let result = obj

  for (const key of keys) {
    if (result === null || result === undefined) {
      return defaultValue
    }
    result = result[key]
  }

  return result !== undefined ? result : defaultValue
}

/**
 * Set nested property value
 */
export function setNestedValue(obj: any, path: string, value: any): void {
  const keys = path.split('.')
  const lastKey = keys.pop()!

  let current = obj
  for (const key of keys) {
    if (!(key in current) || typeof current[key] !== 'object') {
      current[key] = {}
    }
    current = current[key]
  }

  current[lastKey] = value
}

/**
 * Flatten object
 */
export function flattenObject(obj: any, prefix: string = ''): Record<string, any> {
  const flattened: Record<string, any> = {}

  for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
      const newKey = prefix ? `${prefix}.${key}` : key

      if (typeof obj[key] === 'object' && obj[key] !== null && !Array.isArray(obj[key])) {
        Object.assign(flattened, flattenObject(obj[key], newKey))
      } else {
        flattened[newKey] = obj[key]
      }
    }
  }

  return flattened
}

/**
 * Unflatten object
 */
export function unflattenObject(obj: Record<string, any>): any {
  const result: any = {}

  for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
      setNestedValue(result, key, obj[key])
    }
  }

  return result
}

/**
 * Compare objects deeply
 */
export function deepEqual(obj1: any, obj2: any): boolean {
  if (obj1 === obj2) return true

  if (obj1 == null || obj2 == null) return false
  if (typeof obj1 !== 'object' || typeof obj2 !== 'object') return false

  const keys1 = Object.keys(obj1)
  const keys2 = Object.keys(obj2)

  if (keys1.length !== keys2.length) return false

  for (const key of keys1) {
    if (!keys2.includes(key)) return false
    if (!deepEqual(obj1[key], obj2[key])) return false
  }

  return true
}

/**
 * Transform object keys
 */
export function transformKeys<T>(
  obj: Record<string, any>,
  transformFn: (key: string) => string
): Record<string, any> {
  const result: Record<string, any> = {}

  for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
      result[transformFn(key)] = obj[key]
    }
  }

  return result
}

/**
 * Invert object (swap keys and values)
 */
export function invertObject<T extends Record<string, string>>(obj: T): Record<string, string> {
  const result: Record<string, string> = {}

  for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
      result[obj[key]] = key
    }
  }

  return result
}

/**
 * Map object values
 */
export function mapValues<T, U>(
  obj: Record<string, T>,
  mapFn: (value: T, key: string) => U
): Record<string, U> {
  const result: Record<string, U> = {}

  for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
      result[key] = mapFn(obj[key], key)
    }
  }

  return result
}

/**
 * Filter object by predicate
 */
export function filterObject<T extends Record<string, any>>(
  obj: T,
  predicate: (value: any, key: string) => boolean
): Partial<T> {
  const result: Partial<T> = {}

  for (const key in obj) {
    if (obj.hasOwnProperty(key) && predicate(obj[key], key)) {
      result[key] = obj[key]
    }
  }

  return result
}

/**
 * Get object size (number of keys)
 */
export function objectSize(obj: Record<string, any>): number {
  return Object.keys(obj).length
}

/**
 * Check if object is empty
 */
export function isEmptyObject(obj: Record<string, any>): boolean {
  return objectSize(obj) === 0
}




