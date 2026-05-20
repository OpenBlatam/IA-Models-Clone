/**
 * Utilidades de Routing
 * ======================
 * 
 * Funciones para manejo de rutas y navegación
 */

/**
 * Normaliza una ruta
 */
export function normalizePath(path: string): string {
  // Remover trailing slashes excepto para root
  let normalized = path.replace(/\/+$/, '') || '/'
  
  // Remover múltiples slashes
  normalized = normalized.replace(/\/+/g, '/')
  
  return normalized
}

/**
 * Une rutas
 */
export function joinPaths(...paths: string[]): string {
  return paths
    .filter(Boolean)
    .map(path => path.replace(/^\/|\/$/g, ''))
    .join('/')
    .replace(/^/, '/')
}

/**
 * Resuelve una ruta relativa
 */
export function resolvePath(base: string, relative: string): string {
  if (relative.startsWith('/')) {
    return normalizePath(relative)
  }

  const baseParts = base.split('/').filter(Boolean)
  const relativeParts = relative.split('/').filter(Boolean)

  for (const part of relativeParts) {
    if (part === '.') {
      continue
    } else if (part === '..') {
      baseParts.pop()
    } else {
      baseParts.push(part)
    }
  }

  return '/' + baseParts.join('/')
}

/**
 * Obtiene parámetros de una ruta
 */
export function extractRouteParams(
  path: string,
  pattern: string
): Record<string, string> | null {
  const pathParts = path.split('/').filter(Boolean)
  const patternParts = pattern.split('/').filter(Boolean)

  if (pathParts.length !== patternParts.length) {
    return null
  }

  const params: Record<string, string> = {}

  for (let i = 0; i < patternParts.length; i++) {
    const patternPart = patternParts[i]
    const pathPart = pathParts[i]

    if (patternPart.startsWith(':')) {
      const paramName = patternPart.slice(1)
      params[paramName] = pathPart
    } else if (patternPart !== pathPart) {
      return null
    }
  }

  return params
}

/**
 * Construye una ruta con parámetros
 */
export function buildRoute(
  pattern: string,
  params: Record<string, string | number>
): string {
  let route = pattern

  for (const [key, value] of Object.entries(params)) {
    route = route.replace(`:${key}`, String(value))
  }

  return normalizePath(route)
}

/**
 * Verifica si una ruta coincide con un patrón
 */
export function matchRoute(path: string, pattern: string): boolean {
  const pathParts = path.split('/').filter(Boolean)
  const patternParts = pattern.split('/').filter(Boolean)

  if (pathParts.length !== patternParts.length) {
    return false
  }

  for (let i = 0; i < patternParts.length; i++) {
    const patternPart = patternParts[i]
    const pathPart = pathParts[i]

    if (!patternPart.startsWith(':') && patternPart !== pathPart) {
      return false
    }
  }

  return true
}

/**
 * Obtiene el pathname de una URL
 */
export function getPathname(url: string = window.location.href): string {
  try {
    const urlObj = new URL(url, window.location.origin)
    return urlObj.pathname
  } catch {
    return '/'
  }
}

/**
 * Obtiene el hash de una URL
 */
export function getHash(url: string = window.location.href): string {
  try {
    const urlObj = new URL(url, window.location.origin)
    return urlObj.hash.slice(1) // Remove #
  } catch {
    return ''
  }
}

/**
 * Obtiene query params de una URL
 */
export function getQueryParams(url: string = window.location.href): Record<string, string> {
  try {
    const urlObj = new URL(url, window.location.origin)
    const params: Record<string, string> = {}
    
    urlObj.searchParams.forEach((value, key) => {
      params[key] = value
    })
    
    return params
  } catch {
    return {}
  }
}

/**
 * Navega a una ruta
 */
export function navigateTo(path: string, options: {
  replace?: boolean
  state?: any
} = {}): void {
  const { replace = false, state } = options

  if (replace) {
    window.history.replaceState(state, '', path)
  } else {
    window.history.pushState(state, '', path)
  }

  // Disparar evento personalizado
  window.dispatchEvent(new PopStateEvent('popstate'))
}

/**
 * Navega hacia atrás
 */
export function navigateBack(): void {
  window.history.back()
}

/**
 * Navega hacia adelante
 */
export function navigateForward(): void {
  window.history.forward()
}

/**
 * Obtiene la ruta actual
 */
export function getCurrentPath(): string {
  return window.location.pathname
}

/**
 * Verifica si estamos en una ruta específica
 */
export function isCurrentPath(path: string): boolean {
  return normalizePath(getCurrentPath()) === normalizePath(path)
}

/**
 * Crea un router simple
 */
export class SimpleRouter {
  private routes: Map<string, () => void> = new Map()
  private currentRoute: string = ''

  /**
   * Registra una ruta
   */
  register(path: string, handler: () => void): void {
    this.routes.set(normalizePath(path), handler)
  }

  /**
   * Navega a una ruta
   */
  navigate(path: string): void {
    const normalized = normalizePath(path)
    const handler = this.routes.get(normalized)

    if (handler) {
      this.currentRoute = normalized
      handler()
    } else {
      // Buscar ruta con parámetros
      for (const [route, routeHandler] of this.routes.entries()) {
        if (matchRoute(normalized, route)) {
          this.currentRoute = normalized
          routeHandler()
          return
        }
      }
    }
  }

  /**
   * Obtiene la ruta actual
   */
  getCurrentRoute(): string {
    return this.currentRoute
  }

  /**
   * Inicia el router
   */
  start(): void {
    // Navegar a la ruta actual
    this.navigate(getCurrentPath())

    // Escuchar cambios de navegación
    window.addEventListener('popstate', () => {
      this.navigate(getCurrentPath())
    })
  }
}






