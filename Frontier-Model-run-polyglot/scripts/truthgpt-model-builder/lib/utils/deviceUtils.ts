/**
 * Utilidades de Dispositivos
 * ==========================
 * 
 * Funciones para detectar dispositivos y características
 */

/**
 * Detecta el tipo de dispositivo
 */
export function getDeviceType(): 'mobile' | 'tablet' | 'desktop' {
  if (typeof window === 'undefined') return 'desktop'

  const width = window.innerWidth
  if (width < 768) return 'mobile'
  if (width < 1024) return 'tablet'
  return 'desktop'
}

/**
 * Detecta el sistema operativo
 */
export function getOS(): 'windows' | 'macos' | 'linux' | 'android' | 'ios' | 'unknown' {
  if (typeof navigator === 'undefined') return 'unknown'

  const userAgent = navigator.userAgent.toLowerCase()

  if (userAgent.includes('win')) return 'windows'
  if (userAgent.includes('mac')) return 'macos'
  if (userAgent.includes('linux')) return 'linux'
  if (userAgent.includes('android')) return 'android'
  if (userAgent.includes('iphone') || userAgent.includes('ipad')) return 'ios'

  return 'unknown'
}

/**
 * Detecta el navegador
 */
export function getBrowser(): 'chrome' | 'firefox' | 'safari' | 'edge' | 'opera' | 'unknown' {
  if (typeof navigator === 'undefined') return 'unknown'

  const userAgent = navigator.userAgent.toLowerCase()

  if (userAgent.includes('chrome') && !userAgent.includes('edg')) return 'chrome'
  if (userAgent.includes('firefox')) return 'firefox'
  if (userAgent.includes('safari') && !userAgent.includes('chrome')) return 'safari'
  if (userAgent.includes('edg')) return 'edge'
  if (userAgent.includes('opera') || userAgent.includes('opr')) return 'opera'

  return 'unknown'
}

/**
 * Verifica si es un dispositivo táctil
 */
export function isTouchDevice(): boolean {
  if (typeof window === 'undefined') return false
  return 'ontouchstart' in window || navigator.maxTouchPoints > 0
}

/**
 * Verifica si está en modo oscuro del sistema
 */
export function prefersDarkMode(): boolean {
  if (typeof window === 'undefined') return false
  return window.matchMedia('(prefers-color-scheme: dark)').matches
}

/**
 * Verifica si prefiere movimiento reducido
 */
export function prefersReducedMotion(): boolean {
  if (typeof window === 'undefined') return false
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

/**
 * Obtiene información completa del dispositivo
 */
export function getDeviceInfo(): {
  type: 'mobile' | 'tablet' | 'desktop'
  os: 'windows' | 'macos' | 'linux' | 'android' | 'ios' | 'unknown'
  browser: 'chrome' | 'firefox' | 'safari' | 'edge' | 'opera' | 'unknown'
  isTouch: boolean
  prefersDark: boolean
  prefersReducedMotion: boolean
  userAgent: string
  language: string
  platform: string
} {
  return {
    type: getDeviceType(),
    os: getOS(),
    browser: getBrowser(),
    isTouch: isTouchDevice(),
    prefersDark: prefersDarkMode(),
    prefersReducedMotion: prefersReducedMotion(),
    userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : '',
    language: typeof navigator !== 'undefined' ? navigator.language : 'en',
    platform: typeof navigator !== 'undefined' ? navigator.platform : ''
  }
}







