/**
 * Utilidades de URL
 * ================
 * 
 * Funciones para trabajar con URLs
 */

/**
 * Construye una URL con parámetros de consulta
 */
export function buildURL(
  base: string,
  params?: Record<string, string | number | boolean | null | undefined>
): string {
  if (!params || Object.keys(params).length === 0) return base

  const url = new URL(base, window.location.origin)
  
  Object.entries(params).forEach(([key, value]) => {
    if (value !== null && value !== undefined) {
      url.searchParams.append(key, String(value))
    }
  })

  return url.toString()
}

/**
 * Parsea parámetros de consulta de una URL
 */
export function parseQueryParams(url: string = window.location.href): Record<string, string> {
  const params: Record<string, string> = {}
  const urlObj = new URL(url, window.location.origin)
  
  urlObj.searchParams.forEach((value, key) => {
    params[key] = value
  })

  return params
}

/**
 * Obtiene un parámetro de consulta específico
 */
export function getQueryParam(key: string, url?: string): string | null {
  const urlObj = new URL(url || window.location.href, window.location.origin)
  return urlObj.searchParams.get(key)
}

/**
 * Establece un parámetro de consulta en la URL
 */
export function setQueryParam(key: string, value: string | number | boolean, replace: boolean = false): void {
  const url = new URL(window.location.href)
  url.searchParams.set(key, String(value))
  
  if (replace) {
    window.history.replaceState({}, '', url.toString())
  } else {
    window.history.pushState({}, '', url.toString())
  }
}

/**
 * Elimina un parámetro de consulta de la URL
 */
export function removeQueryParam(key: string, replace: boolean = false): void {
  const url = new URL(window.location.href)
  url.searchParams.delete(key)
  
  if (replace) {
    window.history.replaceState({}, '', url.toString())
  } else {
    window.history.pushState({}, '', url.toString())
  }
}

/**
 * Verifica si una URL es absoluta
 */
export function isAbsoluteURL(url: string): boolean {
  return /^([a-z][a-z\d+\-.]*:)?\/\//i.test(url)
}

/**
 * Resuelve una URL relativa contra una base
 */
export function resolveURL(base: string, relative: string): string {
  if (isAbsoluteURL(relative)) return relative
  return new URL(relative, base).toString()
}

/**
 * Normaliza una URL
 */
export function normalizeURL(url: string): string {
  try {
    const urlObj = new URL(url)
    return urlObj.toString()
  } catch {
    return url
  }
}

/**
 * Obtiene el dominio de una URL
 */
export function getDomain(url: string): string | null {
  try {
    const urlObj = new URL(url)
    return urlObj.hostname
  } catch {
    return null
  }
}

/**
 * Obtiene el pathname de una URL
 */
export function getPathname(url: string): string | null {
  try {
    const urlObj = new URL(url)
    return urlObj.pathname
  } catch {
    return null
  }
}

/**
 * Obtiene el protocolo de una URL
 */
export function getProtocol(url: string): string | null {
  try {
    const urlObj = new URL(url)
    return urlObj.protocol.replace(':', '')
  } catch {
    return null
  }
}

/**
 * Verifica si una URL es HTTPS
 */
export function isHTTPS(url: string): boolean {
  return getProtocol(url) === 'https'
}

/**
 * Crea una URL segura (HTTPS) desde una URL
 */
export function makeHTTPS(url: string): string {
  try {
    const urlObj = new URL(url)
    urlObj.protocol = 'https:'
    return urlObj.toString()
  } catch {
    return url
  }
}







