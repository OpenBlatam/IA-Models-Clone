import { useState, useCallback, useRef, useEffect, type RefObject } from 'react'

export const useHover = <T extends HTMLElement = HTMLElement>(): [RefObject<T>, boolean] => {
  const [isHovered, setIsHovered] = useState(false)
  const ref = useRef<T>(null)

  const handleMouseEnter = useCallback(() => {
    setIsHovered(true)
  }, [])

  const handleMouseLeave = useCallback(() => {
    setIsHovered(false)
  }, [])

  useEffect(() => {
    const element = ref.current
    if (!element) {
      return
    }

    element.addEventListener('mouseenter', handleMouseEnter)
    element.addEventListener('mouseleave', handleMouseLeave)

    return () => {
      element.removeEventListener('mouseenter', handleMouseEnter)
      element.removeEventListener('mouseleave', handleMouseLeave)
    }
  }, [handleMouseEnter, handleMouseLeave])

  return [ref, isHovered]
}

