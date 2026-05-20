/**
 * Hook useToggle
 * ================
 * 
 * Hook para alternar un valor booleano
 */

import { useState, useCallback } from 'react'

/**
 * Hook para alternar un valor booleano
 */
export function useToggle(
  initialValue: boolean = false
): [boolean, () => void, (value: boolean) => void] {
  const [value, setValue] = useState(initialValue)

  const toggle = useCallback(() => {
    setValue(prev => !prev)
  }, [])

  const set = useCallback((newValue: boolean) => {
    setValue(newValue)
  }, [])

  return [value, toggle, set]
}






