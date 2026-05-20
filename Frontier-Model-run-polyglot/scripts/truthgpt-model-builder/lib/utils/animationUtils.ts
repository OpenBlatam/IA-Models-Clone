/**
 * Utilidades de Animación
 * ========================
 * 
 * Funciones para animaciones y transiciones
 */

/**
 * Easing functions
 */
export const easing = {
  linear: (t: number) => t,
  easeIn: (t: number) => t * t,
  easeOut: (t: number) => t * (2 - t),
  easeInOut: (t: number) => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,
  easeInQuad: (t: number) => t * t,
  easeOutQuad: (t: number) => t * (2 - t),
  easeInOutQuad: (t: number) => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,
  easeInCubic: (t: number) => t * t * t,
  easeOutCubic: (t: number) => --t * t * t + 1,
  easeInOutCubic: (t: number) => t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1,
  bounce: (t: number) => {
    if (t < 1 / 2.75) {
      return 7.5625 * t * t
    } else if (t < 2 / 2.75) {
      return 7.5625 * (t -= 1.5 / 2.75) * t + 0.75
    } else if (t < 2.5 / 2.75) {
      return 7.5625 * (t -= 2.25 / 2.75) * t + 0.9375
    } else {
      return 7.5625 * (t -= 2.625 / 2.75) * t + 0.984375
    }
  }
}

/**
 * Anima un valor de un punto a otro
 */
export function animate(
  from: number,
  to: number,
  duration: number,
  callback: (value: number) => void,
  easingFn: (t: number) => number = easing.easeInOut
): Promise<void> {
  return new Promise((resolve) => {
    const startTime = performance.now()
    const difference = to - from

    const step = (currentTime: number) => {
      const elapsed = currentTime - startTime
      const progress = Math.min(elapsed / duration, 1)
      const eased = easingFn(progress)
      const current = from + difference * eased

      callback(current)

      if (progress < 1) {
        requestAnimationFrame(step)
      } else {
        resolve()
      }
    }

    requestAnimationFrame(step)
  })
}

/**
 * Anima múltiples valores
 */
export function animateMultiple(
  values: Array<{ from: number; to: number }>,
  duration: number,
  callback: (values: number[]) => void,
  easingFn: (t: number) => number = easing.easeInOut
): Promise<void> {
  return new Promise((resolve) => {
    const startTime = performance.now()
    const differences = values.map(v => v.to - v.from)

    const step = (currentTime: number) => {
      const elapsed = currentTime - startTime
      const progress = Math.min(elapsed / duration, 1)
      const eased = easingFn(progress)
      const currentValues = values.map((v, i) => v.from + differences[i] * eased)

      callback(currentValues)

      if (progress < 1) {
        requestAnimationFrame(step)
      } else {
        resolve()
      }
    }

    requestAnimationFrame(step)
  })
}

/**
 * Crea una animación de fade
 */
export function fade(
  element: HTMLElement,
  to: number,
  duration: number = 300
): Promise<void> {
  const from = parseFloat(getComputedStyle(element).opacity) || 0
  
  return animate(from, to, duration, (value) => {
    element.style.opacity = String(value)
  })
}

/**
 * Crea una animación de slide
 */
export function slide(
  element: HTMLElement,
  direction: 'up' | 'down' | 'left' | 'right',
  distance: number,
  duration: number = 300
): Promise<void> {
  const property = direction === 'up' || direction === 'down' ? 'top' : 'left'
  const multiplier = direction === 'up' || direction === 'left' ? -1 : 1
  
  const from = parseFloat(getComputedStyle(element)[property as any]) || 0
  const to = from + distance * multiplier

  return animate(from, to, duration, (value) => {
    element.style[property as any] = `${value}px`
  })
}

/**
 * Crea una animación de scale
 */
export function scale(
  element: HTMLElement,
  to: number,
  duration: number = 300
): Promise<void> {
  const from = parseFloat(getComputedStyle(element).transform.match(/scale\(([^)]+)\)/)?.[1] || '1') || 1

  return animate(from, to, duration, (value) => {
    element.style.transform = `scale(${value})`
  })
}

/**
 * Crea una animación de rotate
 */
export function rotate(
  element: HTMLElement,
  to: number,
  duration: number = 300
): Promise<void> {
  const from = parseFloat(getComputedStyle(element).transform.match(/rotate\(([^)]+)deg\)/)?.[1] || '0') || 0

  return animate(from, to, duration, (value) => {
    element.style.transform = `rotate(${value}deg)`
  })
}

/**
 * Crea una animación de color
 */
export function animateColor(
  element: HTMLElement,
  property: 'color' | 'backgroundColor' | 'borderColor',
  to: string,
  duration: number = 300
): Promise<void> {
  const fromColor = getComputedStyle(element)[property as any]
  const fromRGB = hexToRgb(fromColor) || { r: 0, g: 0, b: 0 }
  const toRGB = hexToRgb(to) || { r: 0, g: 0, b: 0 }

  return animateMultiple(
    [
      { from: fromRGB.r, to: toRGB.r },
      { from: fromRGB.g, to: toRGB.g },
      { from: fromRGB.b, to: toRGB.b }
    ],
    duration,
    (values) => {
      element.style[property as any] = `rgb(${Math.round(values[0])}, ${Math.round(values[1])}, ${Math.round(values[2])})`
    }
  )
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

/**
 * Crea una animación de spring
 */
export function spring(
  from: number,
  to: number,
  callback: (value: number) => void,
  options: {
    stiffness?: number
    damping?: number
    mass?: number
  } = {}
): () => void {
  const { stiffness = 100, damping = 10, mass = 1 } = options
  
  let velocity = 0
  let position = from
  let animationId: number | null = null

  const step = () => {
    const force = (to - position) * stiffness
    const acceleration = force / mass
    velocity += acceleration
    velocity *= 1 - damping / 100
    position += velocity

    callback(position)

    if (Math.abs(to - position) > 0.01 || Math.abs(velocity) > 0.01) {
      animationId = requestAnimationFrame(step)
    }
  }

  animationId = requestAnimationFrame(step)

  return () => {
    if (animationId !== null) {
      cancelAnimationFrame(animationId)
    }
  }
}






