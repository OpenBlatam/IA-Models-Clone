/**
 * Hook useCounter
 * ===============
 * 
 * Hook para manejar un contador
 */

import { useState, useCallback } from 'react'

export interface UseCounterOptions {
  initialValue?: number
  min?: number
  max?: number
  step?: number
}

export interface UseCounterReturn {
  count: number
  increment: () => void
  decrement: () => void
  reset: () => void
  setCount: (value: number) => void
}

/**
 * Hook para manejar un contador
 */
export function useCounter(options: UseCounterOptions = {}): UseCounterReturn {
  const {
    initialValue = 0,
    min,
    max,
    step = 1
  } = options

  const [count, setCount] = useState(initialValue)

  const increment = useCallback(() => {
    setCount(prev => {
      const newValue = prev + step
      return max !== undefined ? Math.min(newValue, max) : newValue
    })
  }, [step, max])

  const decrement = useCallback(() => {
    setCount(prev => {
      const newValue = prev - step
      return min !== undefined ? Math.max(newValue, min) : newValue
    })
  }, [step, min])

  const reset = useCallback(() => {
    setCount(initialValue)
  }, [initialValue])

  const setCountValue = useCallback((value: number) => {
    setCount(prev => {
      let newValue = value
      if (min !== undefined) newValue = Math.max(newValue, min)
      if (max !== undefined) newValue = Math.min(newValue, max)
      return newValue
    })
  }, [min, max])

  return {
    count,
    increment,
    decrement,
    reset,
    setCount: setCountValue
  }
}






