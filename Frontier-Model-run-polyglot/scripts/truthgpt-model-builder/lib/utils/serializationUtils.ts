/**
 * Utilidades de Serialización
 * ===========================
 * 
 * Funciones para serializar y deserializar datos
 */

/**
 * Serializa un objeto a JSON con manejo de errores
 */
export function serializeJSON<T>(data: T, space?: number): string {
  try {
    return JSON.stringify(data, null, space)
  } catch (error) {
    throw new Error(`Error serializing JSON: ${error instanceof Error ? error.message : String(error)}`)
  }
}

/**
 * Deserializa JSON a objeto con manejo de errores
 */
export function deserializeJSON<T>(json: string): T {
  try {
    return JSON.parse(json) as T
  } catch (error) {
    throw new Error(`Error deserializing JSON: ${error instanceof Error ? error.message : String(error)}`)
  }
}

/**
 * Serializa con soporte para funciones y undefined
 */
export function serializeAdvanced<T>(data: T): string {
  return JSON.stringify(data, (key, value) => {
    if (typeof value === 'function') {
      return `[Function ${value.name || 'anonymous'}]`
    }
    if (value === undefined) {
      return '[undefined]'
    }
    if (value instanceof Date) {
      return { __type: 'Date', value: value.toISOString() }
    }
    if (value instanceof RegExp) {
      return { __type: 'RegExp', value: value.toString() }
    }
    if (value instanceof Map) {
      return { __type: 'Map', value: Array.from(value.entries()) }
    }
    if (value instanceof Set) {
      return { __type: 'Set', value: Array.from(value) }
    }
    return value
  })
}

/**
 * Deserializa con soporte para tipos especiales
 */
export function deserializeAdvanced<T>(json: string): T {
  return JSON.parse(json, (key, value) => {
    if (value && typeof value === 'object' && '__type' in value) {
      switch (value.__type) {
        case 'Date':
          return new Date(value.value)
        case 'RegExp':
          return new RegExp(value.value)
        case 'Map':
          return new Map(value.value)
        case 'Set':
          return new Set(value.value)
        default:
          return value
      }
    }
    if (value === '[undefined]') {
      return undefined
    }
    return value
  }) as T
}

/**
 * Serializa a Base64
 */
export function serializeBase64<T>(data: T): string {
  const json = serializeJSON(data)
  if (typeof btoa !== 'undefined') {
    return btoa(json)
  }
  // Fallback para Node.js
  return Buffer.from(json).toString('base64')
}

/**
 * Deserializa desde Base64
 */
export function deserializeBase64<T>(base64: string): T {
  let json: string
  if (typeof atob !== 'undefined') {
    json = atob(base64)
  } else {
    // Fallback para Node.js
    json = Buffer.from(base64, 'base64').toString('utf-8')
  }
  return deserializeJSON<T>(json)
}

/**
 * Serializa a URL-encoded string
 */
export function serializeURLEncoded(data: Record<string, any>): string {
  const params = new URLSearchParams()
  
  for (const [key, value] of Object.entries(data)) {
    if (value !== null && value !== undefined) {
      if (Array.isArray(value)) {
        value.forEach(item => params.append(key, String(item)))
      } else {
        params.append(key, String(value))
      }
    }
  }
  
  return params.toString()
}

/**
 * Deserializa desde URL-encoded string
 */
export function deserializeURLEncoded(query: string): Record<string, string | string[]> {
  const params = new URLSearchParams(query)
  const result: Record<string, string | string[]> = {}
  
  for (const [key, value] of params.entries()) {
    if (key in result) {
      const existing = result[key]
      if (Array.isArray(existing)) {
        existing.push(value)
      } else {
        result[key] = [existing as string, value]
      }
    } else {
      result[key] = value
    }
  }
  
  return result
}

/**
 * Serializa a CSV
 */
export function serializeCSV(
  data: Record<string, any>[],
  options: {
    headers?: string[]
    delimiter?: string
  } = {}
): string {
  const { headers, delimiter = ',' } = options
  const keys = headers || (data.length > 0 ? Object.keys(data[0]) : [])
  
  const rows: string[] = []
  
  // Headers
  rows.push(keys.map(key => `"${String(key).replace(/"/g, '""')}"`).join(delimiter))
  
  // Data
  for (const row of data) {
    rows.push(
      keys.map(key => {
        const value = row[key]
        if (value === null || value === undefined) {
          return ''
        }
        return `"${String(value).replace(/"/g, '""')}"`
      }).join(delimiter)
    )
  }
  
  return rows.join('\n')
}

/**
 * Deserializa desde CSV
 */
export function deserializeCSV(
  csv: string,
  options: {
    delimiter?: string
    hasHeaders?: boolean
  } = {}
): Record<string, any>[] {
  const { delimiter = ',', hasHeaders = true } = options
  const lines = csv.split('\n').filter(line => line.trim())
  
  if (lines.length === 0) {
    return []
  }
  
  const headers = hasHeaders
    ? lines[0].split(delimiter).map(h => h.trim().replace(/^"|"$/g, ''))
    : []
  
  const data: Record<string, any>[] = []
  const startIndex = hasHeaders ? 1 : 0
  
  for (let i = startIndex; i < lines.length; i++) {
    const values = lines[i].split(delimiter).map(v => v.trim().replace(/^"|"$/g, ''))
    const row: Record<string, any> = {}
    
    if (hasHeaders) {
      headers.forEach((header, index) => {
        row[header] = values[index] || ''
      })
    } else {
      values.forEach((value, index) => {
        row[`column${index + 1}`] = value
      })
    }
    
    data.push(row)
  }
  
  return data
}

/**
 * Clona un objeto usando serialización
 */
export function cloneViaSerialization<T>(obj: T): T {
  return deserializeJSON<T>(serializeJSON(obj))
}

/**
 * Compara objetos usando serialización
 */
export function compareViaSerialization<T>(obj1: T, obj2: T): boolean {
  return serializeJSON(obj1) === serializeJSON(obj2)
}






