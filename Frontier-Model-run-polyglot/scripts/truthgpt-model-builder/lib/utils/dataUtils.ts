/**
 * Utilidades de Datos Avanzadas
 * ==============================
 * 
 * Funciones para manipulación y transformación de datos
 */

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
 * Agrupa elementos y aplica una función de reducción
 */
export function groupByAndReduce<T, R>(
  array: T[],
  keyFn: (item: T) => string | number,
  reduceFn: (group: T[]) => R
): Record<string, R> {
  const grouped = groupBy(array, keyFn)
  const result: Record<string, R> = {}

  for (const [key, items] of Object.entries(grouped)) {
    result[key] = reduceFn(items)
  }

  return result
}

/**
 * Particiona un array en chunks
 */
export function chunk<T>(array: T[], size: number): T[][] {
  const chunks: T[][] = []
  for (let i = 0; i < array.length; i += size) {
    chunks.push(array.slice(i, i + size))
  }
  return chunks
}

/**
 * Crea un array de números en un rango
 */
export function range(start: number, end: number, step: number = 1): number[] {
  const result: number[] = []
  if (step > 0) {
    for (let i = start; i < end; i += step) {
      result.push(i)
    }
  } else if (step < 0) {
    for (let i = start; i > end; i += step) {
      result.push(i)
    }
  }
  return result
}

/**
 * Obtiene valores únicos de un array
 */
export function unique<T>(array: T[]): T[] {
  return Array.from(new Set(array))
}

/**
 * Obtiene valores únicos por una función de comparación
 */
export function uniqueBy<T>(
  array: T[],
  keyFn: (item: T) => any
): T[] {
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
 * Ordena un array por múltiples criterios
 */
export function sortBy<T>(
  array: T[],
  ...keyFns: Array<(item: T) => any>
): T[] {
  return [...array].sort((a, b) => {
    for (const keyFn of keyFns) {
      const aVal = keyFn(a)
      const bVal = keyFn(b)

      if (aVal < bVal) return -1
      if (aVal > bVal) return 1
    }
    return 0
  })
}

/**
 * Ordena un array en orden descendente
 */
export function sortByDesc<T>(
  array: T[],
  ...keyFns: Array<(item: T) => any>
): T[] {
  return sortBy(array, ...keyFns).reverse()
}

/**
 * Obtiene el elemento con el valor máximo
 */
export function maxBy<T>(
  array: T[],
  keyFn: (item: T) => number
): T | undefined {
  if (array.length === 0) return undefined

  let max = array[0]
  let maxValue = keyFn(max)

  for (let i = 1; i < array.length; i++) {
    const value = keyFn(array[i])
    if (value > maxValue) {
      max = array[i]
      maxValue = value
    }
  }

  return max
}

/**
 * Obtiene el elemento con el valor mínimo
 */
export function minBy<T>(
  array: T[],
  keyFn: (item: T) => number
): T | undefined {
  if (array.length === 0) return undefined

  let min = array[0]
  let minValue = keyFn(min)

  for (let i = 1; i < array.length; i++) {
    const value = keyFn(array[i])
    if (value < minValue) {
      min = array[i]
      minValue = value
    }
  }

  return min
}

/**
 * Calcula la suma de valores
 */
export function sumBy<T>(
  array: T[],
  keyFn: (item: T) => number
): number {
  return array.reduce((sum, item) => sum + keyFn(item), 0)
}

/**
 * Calcula el promedio de valores
 */
export function averageBy<T>(
  array: T[],
  keyFn: (item: T) => number
): number {
  if (array.length === 0) return 0
  return sumBy(array, keyFn) / array.length
}

/**
 * Cuenta elementos que cumplen una condición
 */
export function countBy<T>(
  array: T[],
  predicate: (item: T) => boolean
): number {
  return array.filter(predicate).length
}

/**
 * Crea un objeto desde un array usando una clave
 */
export function keyBy<T>(
  array: T[],
  keyFn: (item: T) => string | number
): Record<string, T> {
  return array.reduce((obj, item) => {
    obj[String(keyFn(item))] = item
    return obj
  }, {} as Record<string, T>)
}

/**
 * Plana un array de arrays
 */
export function flatten<T>(array: (T | T[])[]): T[] {
  return array.reduce((flat, item) => {
    return flat.concat(Array.isArray(item) ? flatten(item) : item)
  }, [] as T[])
}

/**
 * Plana un array con profundidad específica
 */
export function flattenDepth<T>(array: any[], depth: number = 1): T[] {
  if (depth === 0) return array as T[]
  
  return array.reduce((flat, item) => {
    return flat.concat(
      Array.isArray(item) ? flattenDepth(item, depth - 1) : item
    )
  }, [] as T[])
}

/**
 * Crea un array de pares [key, value] desde un objeto
 */
export function entries<T extends Record<string, any>>(
  obj: T
): Array<[keyof T, T[keyof T]]> {
  return Object.entries(obj) as Array<[keyof T, T[keyof T]]>
}

/**
 * Crea un objeto desde un array de pares [key, value]
 */
export function fromEntries<T = any>(
  entries: Array<[string, T]>
): Record<string, T> {
  return Object.fromEntries(entries)
}

/**
 * Omite propiedades de un objeto
 */
export function omit<T extends Record<string, any>, K extends keyof T>(
  obj: T,
  keys: K[]
): Omit<T, K> {
  const result = { ...obj }
  keys.forEach(key => {
    delete result[key]
  })
  return result
}

/**
 * Selecciona propiedades específicas de un objeto
 */
export function pick<T extends Record<string, any>, K extends keyof T>(
  obj: T,
  keys: K[]
): Pick<T, K> {
  const result = {} as Pick<T, K>
  keys.forEach(key => {
    if (key in obj) {
      result[key] = obj[key]
    }
  })
  return result
}

/**
 * Aplana un objeto anidado
 */
export function flattenObject(
  obj: Record<string, any>,
  prefix: string = '',
  separator: string = '.'
): Record<string, any> {
  const flattened: Record<string, any> = {}

  for (const [key, value] of Object.entries(obj)) {
    const newKey = prefix ? `${prefix}${separator}${key}` : key

    if (value && typeof value === 'object' && !Array.isArray(value)) {
      Object.assign(flattened, flattenObject(value, newKey, separator))
    } else {
      flattened[newKey] = value
    }
  }

  return flattened
}

/**
 * Desaplana un objeto
 */
export function unflattenObject(
  obj: Record<string, any>,
  separator: string = '.'
): Record<string, any> {
  const result: Record<string, any> = {}

  for (const [key, value] of Object.entries(obj)) {
    const keys = key.split(separator)
    let current = result

    for (let i = 0; i < keys.length - 1; i++) {
      const k = keys[i]
      if (!(k in current)) {
        current[k] = {}
      }
      current = current[k]
    }

    current[keys[keys.length - 1]] = value
  }

  return result
}

/**
 * Combina múltiples objetos
 */
export function merge<T extends Record<string, any>>(
  ...objects: T[]
): T {
  return Object.assign({}, ...objects)
}

/**
 * Combina objetos profundamente
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
        if (!target[key]) {
          Object.assign(target, { [key]: {} })
        }
        deepMerge(target[key], source[key])
      } else {
        Object.assign(target, { [key]: source[key] })
      }
    }
  }

  return deepMerge(target, ...sources)
}

function isObject(item: any): item is Record<string, any> {
  return item && typeof item === 'object' && !Array.isArray(item)
}

/**
 * Clona un objeto profundamente
 */
export function deepClone<T>(obj: T): T {
  if (obj === null || typeof obj !== 'object') return obj
  if (obj instanceof Date) return new Date(obj.getTime()) as any
  if (obj instanceof Array) return obj.map(item => deepClone(item)) as any
  if (typeof obj === 'object') {
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
 * Convierte un objeto a FormData
 */
export function objectToFormData(
  obj: Record<string, any>,
  formData: FormData = new FormData(),
  parentKey?: string
): FormData {
  for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
      const value = obj[key]
      const formKey = parentKey ? `${parentKey}[${key}]` : key

      if (value === null || value === undefined) {
        continue
      } else if (value instanceof Date) {
        formData.append(formKey, value.toISOString())
      } else if (value instanceof File || value instanceof Blob) {
        formData.append(formKey, value)
      } else if (Array.isArray(value)) {
        value.forEach((item, index) => {
          objectToFormData({ [index]: item }, formData, formKey)
        })
      } else if (typeof value === 'object') {
        objectToFormData(value, formData, formKey)
      } else {
        formData.append(formKey, String(value))
      }
    }
  }

  return formData
}

/**
 * Convierte FormData a objeto
 */
export function formDataToObject(formData: FormData): Record<string, any> {
  const obj: Record<string, any> = {}

  formData.forEach((value, key) => {
    if (obj[key]) {
      if (Array.isArray(obj[key])) {
        obj[key].push(value)
      } else {
        obj[key] = [obj[key], value]
      }
    } else {
      obj[key] = value
    }
  })

  return obj
}






