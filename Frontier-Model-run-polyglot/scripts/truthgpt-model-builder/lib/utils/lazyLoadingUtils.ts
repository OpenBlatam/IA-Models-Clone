/**
 * Utilidades de Lazy Loading
 * ==========================
 * 
 * Funciones para lazy loading de recursos
 */

/**
 * Carga una imagen de forma lazy
 */
export function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = src
  })
}

/**
 * Carga un script de forma lazy
 */
export function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    // Verificar si ya está cargado
    const existing = document.querySelector(`script[src="${src}"]`)
    if (existing) {
      resolve()
      return
    }

    const script = document.createElement('script')
    script.src = src
    script.async = true
    script.onload = () => resolve()
    script.onerror = reject
    document.head.appendChild(script)
  })
}

/**
 * Carga un stylesheet de forma lazy
 */
export function loadStylesheet(href: string): Promise<void> {
  return new Promise((resolve, reject) => {
    // Verificar si ya está cargado
    const existing = document.querySelector(`link[href="${href}"]`)
    if (existing) {
      resolve()
      return
    }

    const link = document.createElement('link')
    link.rel = 'stylesheet'
    link.href = href
    link.onload = () => resolve()
    link.onerror = reject
    document.head.appendChild(link)
  })
}

/**
 * Carga un módulo dinámicamente
 */
export async function loadModule<T = any>(modulePath: string): Promise<T> {
  if (typeof window !== 'undefined' && 'import' in window) {
    const module = await (window as any).import(modulePath)
    return module.default || module
  }
  throw new Error('Dynamic imports not supported')
}

/**
 * Preloada un recurso
 */
export function preloadResource(
  href: string,
  as: 'script' | 'style' | 'image' | 'font' | 'fetch'
): void {
  const link = document.createElement('link')
  link.rel = 'preload'
  link.href = href
  link.as = as
  document.head.appendChild(link)
}

/**
 * Prefetcha un recurso
 */
export function prefetchResource(href: string): void {
  const link = document.createElement('link')
  link.rel = 'prefetch'
  link.href = href
  document.head.appendChild(link)
}

/**
 * Carga múltiples recursos en paralelo
 */
export async function loadResources(
  resources: Array<{
    type: 'script' | 'stylesheet' | 'image'
    src: string
  }>
): Promise<void> {
  const promises = resources.map(resource => {
    switch (resource.type) {
      case 'script':
        return loadScript(resource.src)
      case 'stylesheet':
        return loadStylesheet(resource.src)
      case 'image':
        return loadImage(resource.src).then(() => {})
      default:
        return Promise.resolve()
    }
  })

  await Promise.all(promises)
}

/**
 * Crea un Intersection Observer para lazy loading
 */
export function createLazyObserver(
  callback: (entries: IntersectionObserverEntry[]) => void,
  options?: IntersectionObserverInit
): IntersectionObserver {
  return new IntersectionObserver(callback, {
    rootMargin: '50px',
    threshold: 0.01,
    ...options
  })
}






