/**
 * Hook useAnimation
 * =================
 * 
 * Hook para animaciones
 */

import { useRef, useCallback } from 'react'
import { animate, animateMultiple, easing, EasingFunction } from '../utils/animationUtils'

export interface UseAnimationOptions {
  duration?: number
  easing?: EasingFunction
  onComplete?: () => void
}

/**
 * Hook para animaciones
 */
export function useAnimation(options: UseAnimationOptions = {}) {
  const { duration = 300, easing: easingFn = easing.easeInOut, onComplete } = options
  const animationRef = useRef<Promise<void> | null>(null)

  const animateValue = useCallback(
    (from: number, to: number, callback: (value: number) => void) => {
      if (animationRef.current) {
        animationRef.current.then(() => {
          animationRef.current = animate(from, to, duration, callback, easingFn)
          animationRef.current.then(() => onComplete?.())
        })
      } else {
        animationRef.current = animate(from, to, duration, callback, easingFn)
        animationRef.current.then(() => onComplete?.())
      }
    },
    [duration, easingFn, onComplete]
  )

  const animateValues = useCallback(
    (values: Array<{ from: number; to: number }>, callback: (values: number[]) => void) => {
      if (animationRef.current) {
        animationRef.current.then(() => {
          animationRef.current = animateMultiple(values, duration, callback, easingFn)
          animationRef.current.then(() => onComplete?.())
        })
      } else {
        animationRef.current = animateMultiple(values, duration, callback, easingFn)
        animationRef.current.then(() => onComplete?.())
      }
    },
    [duration, easingFn, onComplete]
  )

  return {
    animate: animateValue,
    animateMultiple: animateValues
  }
}






