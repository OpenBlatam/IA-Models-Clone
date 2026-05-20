/**
 * Utilidades de Transformación
 * =============================
 * 
 * Funciones para transformar datos entre formatos
 */

/**
 * Transforma un objeto a otro formato usando un mapper
 */
export function transformObject<TInput, TOutput>(
  input: TInput,
  mapper: (input: TInput) => TOutput
): TOutput {
  return mapper(input)
}

/**
 * Transforma un array de objetos
 */
export function transformArray<TInput, TOutput>(
  array: TInput[],
  mapper: (item: TInput, index: number) => TOutput
): TOutput[] {
  return array.map(mapper)
}

/**
 * Transforma un objeto a otro con mapeo de campos
 */
export function mapFields<TInput extends Record<string, any>, TOutput extends Record<string, any>>(
  input: TInput,
  fieldMap: {
    [K in keyof TOutput]: keyof TInput | ((input: TInput) => TOutput[K])
  }
): TOutput {
  const output = {} as TOutput

  for (const [outputKey, inputKeyOrFn] of Object.entries(fieldMap)) {
    if (typeof inputKeyOrFn === 'function') {
      output[outputKey as keyof TOutput] = inputKeyOrFn(input)
    } else {
      output[outputKey as keyof TOutput] = input[inputKeyOrFn as keyof TInput] as any
    }
  }

  return output
}

/**
 * Transforma un objeto plano a estructura anidada
 */
export function nestObject(
  flat: Record<string, any>,
  separator: string = '.'
): Record<string, any> {
  const nested: Record<string, any> = {}

  for (const [key, value] of Object.entries(flat)) {
    const keys = key.split(separator)
    let current = nested

    for (let i = 0; i < keys.length - 1; i++) {
      const k = keys[i]
      if (!(k in current)) {
        current[k] = {}
      }
      current = current[k]
    }

    current[keys[keys.length - 1]] = value
  }

  return nested
}

/**
 * Transforma un objeto anidado a estructura plana
 */
export function flattenObject(
  nested: Record<string, any>,
  separator: string = '.',
  prefix: string = ''
): Record<string, any> {
  const flat: Record<string, any> = {}

  for (const [key, value] of Object.entries(nested)) {
    const newKey = prefix ? `${prefix}${separator}${key}` : key

    if (value && typeof value === 'object' && !Array.isArray(value)) {
      Object.assign(flat, flattenObject(value, separator, newKey))
    } else {
      flat[newKey] = value
    }
  }

  return flat
}

/**
 * Transforma un array a objeto usando una clave
 */
export function arrayToObject<T>(
  array: T[],
  keyFn: (item: T, index: number) => string | number
): Record<string, T> {
  const obj: Record<string, T> = {}
  
  array.forEach((item, index) => {
    const key = String(keyFn(item, index))
    obj[key] = item
  })
  
  return obj
}

/**
 * Transforma un objeto a array
 */
export function objectToArray<T>(
  obj: Record<string, T>,
  keyName: string = 'key'
): Array<T & { [key: string]: string }> {
  return Object.entries(obj).map(([key, value]) => ({
    ...value,
    [keyName]: key
  }))
}

/**
 * Transforma valores de un objeto
 */
export function transformValues<T extends Record<string, any>>(
  obj: T,
  transformFn: <K extends keyof T>(value: T[K], key: K) => any
): Record<string, any> {
  const result: Record<string, any> = {}

  for (const [key, value] of Object.entries(obj)) {
    result[key] = transformFn(value as T[keyof T], key as keyof T)
  }

  return result
}

/**
 * Transforma claves de un objeto
 */
export function transformKeys<T extends Record<string, any>>(
  obj: T,
  transformFn: (key: string) => string
): Record<string, any> {
  const result: Record<string, any> = {}

  for (const [key, value] of Object.entries(obj)) {
    result[transformFn(key)] = value
  }

  return result
}

/**
 * Transforma un objeto a query string
 */
export function objectToQueryString(
  obj: Record<string, any>,
  options: {
    encode?: boolean
    arrayFormat?: 'brackets' | 'indices' | 'repeat'
  } = {}
): string {
  const { encode = true, arrayFormat = 'brackets' } = options
  const params = new URLSearchParams()

  for (const [key, value] of Object.entries(obj)) {
    if (value === null || value === undefined) {
      continue
    }

    if (Array.isArray(value)) {
      value.forEach((item, index) => {
        const arrayKey = arrayFormat === 'indices'
          ? `${key}[${index}]`
          : arrayFormat === 'brackets'
          ? `${key}[]`
          : key
        params.append(arrayKey, String(item))
      })
    } else if (typeof value === 'object') {
      // Nested objects
      for (const [nestedKey, nestedValue] of Object.entries(value)) {
        params.append(`${key}[${nestedKey}]`, String(nestedValue))
      }
    } else {
      params.append(key, String(value))
    }
  }

  return params.toString()
}

/**
 * Transforma query string a objeto
 */
export function queryStringToObject(
  query: string
): Record<string, any> {
  const params = new URLSearchParams(query)
  const result: Record<string, any> = {}

  for (const [key, value] of params.entries()) {
    // Handle array notation
    if (key.endsWith('[]')) {
      const baseKey = key.slice(0, -2)
      if (!(baseKey in result)) {
        result[baseKey] = []
      }
      result[baseKey].push(value)
    } else if (key.includes('[') && key.endsWith(']')) {
      // Handle nested objects
      const match = key.match(/^([^\[]+)\[([^\]]+)\]$/)
      if (match) {
        const [, baseKey, nestedKey] = match
        if (!(baseKey in result)) {
          result[baseKey] = {}
        }
        result[baseKey][nestedKey] = value
      }
    } else {
      result[key] = value
    }
  }

  return result
}

/**
 * Transforma un objeto a FormData
 */
export function objectToFormData(
  obj: Record<string, any>,
  formData: FormData = new FormData(),
  parentKey?: string
): FormData {
  for (const [key, value] of Object.entries(obj)) {
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

  return formData
}

/**
 * Transforma FormData a objeto
 */
export function formDataToObject(formData: FormData): Record<string, any> {
  const obj: Record<string, any> = {}

  formData.forEach((value, key) => {
    if (key in obj) {
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






