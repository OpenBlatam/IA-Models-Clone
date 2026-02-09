'use client'

import { ReactNode, useEffect, useRef, useState } from 'react'
import { cn } from '@/lib/utils'

interface StickyProps {
  children: ReactNode
  offset?: number
  className?: string
  zIndex?: number
}

const Sticky = ({ children, offset = 0, className, zIndex = 10 }: StickyProps) => {
  const [isSticky, setIsSticky] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!ref.current) {
      return
    }

    const element = ref.current
    const originalTop = element.offsetTop

    const handleScroll = () => {
      const scrollTop = window.pageYOffset || document.documentElement.scrollTop
      setIsSticky(scrollTop >= originalTop - offset)
    }

    window.addEventListener('scroll', handleScroll, { passive: true })
    handleScroll()

    return () => {
      window.removeEventListener('scroll', handleScroll)
    }
  }, [offset])

  return (
    <div
      ref={ref}
      className={cn(
        'transition-all',
        isSticky && 'fixed',
        className
      )}
      style={isSticky ? { top: `${offset}px`, zIndex } : undefined}
    >
      {children}
    </div>
  )
}

export default Sticky

