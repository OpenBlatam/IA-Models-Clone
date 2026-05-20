/**
 * Utilidades de Accesibilidad
 * ===========================
 * 
 * Funciones para mejorar la accesibilidad
 */

// ============================================================================
// ARIA HELPERS
// ============================================================================

/**
 * Genera un ID único para ARIA
 */
let ariaIdCounter = 0
export function generateAriaId(prefix: string = 'aria'): string {
  return `${prefix}-${++ariaIdCounter}-${Date.now()}`
}

/**
 * Crea atributos ARIA para un campo de formulario
 */
export function createFieldAriaProps(
  id: string,
  labelId: string,
  errorId?: string,
  describedBy?: string
): {
  id: string
  'aria-labelledby': string
  'aria-describedby'?: string
  'aria-invalid'?: boolean
} {
  const describedByIds = [errorId, describedBy].filter(Boolean).join(' ')

  return {
    id,
    'aria-labelledby': labelId,
    ...(describedByIds && { 'aria-describedby': describedByIds }),
    ...(errorId && { 'aria-invalid': true })
  }
}

/**
 * Crea atributos ARIA para un botón
 */
export function createButtonAriaProps(
  label: string,
  pressed?: boolean,
  expanded?: boolean
): {
  'aria-label': string
  'aria-pressed'?: boolean
  'aria-expanded'?: boolean
} {
  return {
    'aria-label': label,
    ...(pressed !== undefined && { 'aria-pressed': pressed }),
    ...(expanded !== undefined && { 'aria-expanded': expanded })
  }
}

/**
 * Crea atributos ARIA para un modal
 */
export function createModalAriaProps(
  id: string,
  titleId: string,
  describedBy?: string
): {
  role: 'dialog'
  'aria-modal': boolean
  'aria-labelledby': string
  'aria-describedby'?: string
} {
  return {
    role: 'dialog',
    'aria-modal': true,
    'aria-labelledby': titleId,
    ...(describedBy && { 'aria-describedby': describedBy })
  }
}

// ============================================================================
// KEYBOARD NAVIGATION
// ============================================================================

/**
 * Códigos de teclas comunes
 */
export const KEY_CODES = {
  ENTER: 'Enter',
  ESCAPE: 'Escape',
  SPACE: ' ',
  TAB: 'Tab',
  ARROW_UP: 'ArrowUp',
  ARROW_DOWN: 'ArrowDown',
  ARROW_LEFT: 'ArrowLeft',
  ARROW_RIGHT: 'ArrowRight',
  HOME: 'Home',
  END: 'End',
  PAGE_UP: 'PageUp',
  PAGE_DOWN: 'PageDown'
} as const

/**
 * Maneja navegación con flechas en una lista
 */
export function handleArrowNavigation(
  currentIndex: number,
  direction: 'up' | 'down',
  itemCount: number,
  callback: (newIndex: number) => void
): void {
  let newIndex = currentIndex

  if (direction === 'up') {
    newIndex = currentIndex > 0 ? currentIndex - 1 : itemCount - 1
  } else {
    newIndex = currentIndex < itemCount - 1 ? currentIndex + 1 : 0
  }

  callback(newIndex)
}

/**
 * Maneja navegación con Home/End
 */
export function handleHomeEndNavigation(
  key: 'Home' | 'End',
  itemCount: number,
  callback: (newIndex: number) => void
): void {
  const newIndex = key === 'Home' ? 0 : itemCount - 1
  callback(newIndex)
}

// ============================================================================
// FOCUS MANAGEMENT
// ============================================================================

/**
 * Enfoca un elemento de forma segura
 */
export function focusElement(element: HTMLElement | null): void {
  if (element && typeof element.focus === 'function') {
    try {
      element.focus()
    } catch (error) {
      console.warn('Failed to focus element:', error)
    }
  }
}

/**
 * Enfoca el primer elemento focusable dentro de un contenedor
 */
export function focusFirstFocusable(container: HTMLElement | null): void {
  if (!container) return

  const focusableSelectors = [
    'a[href]',
    'button:not([disabled])',
    'textarea:not([disabled])',
    'input:not([disabled])',
    'select:not([disabled])',
    '[tabindex]:not([tabindex="-1"])'
  ].join(', ')

  const firstFocusable = container.querySelector<HTMLElement>(focusableSelectors)
  focusElement(firstFocusable)
}

/**
 * Enfoca el último elemento focusable dentro de un contenedor
 */
export function focusLastFocusable(container: HTMLElement | null): void {
  if (!container) return

  const focusableSelectors = [
    'a[href]',
    'button:not([disabled])',
    'textarea:not([disabled])',
    'input:not([disabled])',
    'select:not([disabled])',
    '[tabindex]:not([tabindex="-1"])'
  ].join(', ')

  const focusableElements = Array.from(
    container.querySelectorAll<HTMLElement>(focusableSelectors)
  )

  if (focusableElements.length > 0) {
    focusElement(focusableElements[focusableElements.length - 1])
  }
}

/**
 * Atrapa el foco dentro de un contenedor (para modales)
 */
export function trapFocus(container: HTMLElement | null): () => void {
  if (!container) return () => {}

  const handleTab = (event: KeyboardEvent) => {
    if (event.key !== 'Tab') return

    const focusableSelectors = [
      'a[href]',
      'button:not([disabled])',
      'textarea:not([disabled])',
      'input:not([disabled])',
      'select:not([disabled])',
      '[tabindex]:not([tabindex="-1"])'
    ].join(', ')

    const focusableElements = Array.from(
      container.querySelectorAll<HTMLElement>(focusableSelectors)
    )

    if (focusableElements.length === 0) {
      event.preventDefault()
      return
    }

    const firstElement = focusableElements[0]
    const lastElement = focusableElements[focusableElements.length - 1]

    if (event.shiftKey) {
      // Shift + Tab
      if (document.activeElement === firstElement) {
        event.preventDefault()
        focusElement(lastElement)
      }
    } else {
      // Tab
      if (document.activeElement === lastElement) {
        event.preventDefault()
        focusElement(firstElement)
      }
    }
  }

  container.addEventListener('keydown', handleTab)
  
  return () => {
    container.removeEventListener('keydown', handleTab)
  }
}

// ============================================================================
// SCREEN READER
// ============================================================================

/**
 * Anuncia un mensaje a los screen readers
 */
export function announceToScreenReader(message: string, priority: 'polite' | 'assertive' = 'polite'): void {
  const announcement = document.createElement('div')
  announcement.setAttribute('role', 'status')
  announcement.setAttribute('aria-live', priority)
  announcement.setAttribute('aria-atomic', 'true')
  announcement.className = 'sr-only'
  announcement.textContent = message

  document.body.appendChild(announcement)

  setTimeout(() => {
    document.body.removeChild(announcement)
  }, 1000)
}

/**
 * Oculta visualmente pero mantiene accesible para screen readers
 */
export const srOnlyClass = 'sr-only'

// ============================================================================
// COLOR CONTRAST
// ============================================================================

/**
 * Calcula el contraste entre dos colores (WCAG)
 */
export function calculateContrast(color1: string, color2: string): number {
  const getLuminance = (color: string): number => {
    const rgb = hexToRgb(color)
    if (!rgb) return 0

    const [r, g, b] = [rgb.r, rgb.g, rgb.b].map(val => {
      val = val / 255
      return val <= 0.03928 ? val / 12.92 : Math.pow((val + 0.055) / 1.055, 2.4)
    })

    return 0.2126 * r + 0.7152 * g + 0.0722 * b
  }

  const l1 = getLuminance(color1)
  const l2 = getLuminance(color2)
  const lighter = Math.max(l1, l2)
  const darker = Math.min(l1, l2)

  return (lighter + 0.05) / (darker + 0.05)
}

/**
 * Verifica si el contraste cumple con WCAG AA
 */
export function meetsWCAGAA(contrast: number): boolean {
  return contrast >= 4.5
}

/**
 * Verifica si el contraste cumple con WCAG AAA
 */
export function meetsWCAGAAA(contrast: number): boolean {
  return contrast >= 7
}

/**
 * Convierte hex a RGB
 */
function hexToRgb(hex: string): { r: number; g: number; b: number } | null {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
  return result
    ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
      }
    : null
}







