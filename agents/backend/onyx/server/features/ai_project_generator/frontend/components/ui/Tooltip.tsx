'use client'

import { useState, useRef, ReactNode, useEffect } from 'react'
import { cn } from '@/lib/utils'
import { useHover } from '@/hooks/ui'
import Portal from './Portal'

interface TooltipProps {
  content: ReactNode
  children: ReactNode
  position?: 'top' | 'bottom' | 'left' | 'right'
  delay?: number
  className?: string
  disabled?: boolean
}

const Tooltip = ({
  content,
  children,
  position = 'top',
  delay = 200,
  className,
  disabled = false,
}: TooltipProps) => {
  const [isVisible, setIsVisible] = useState(false)
  const [tooltipPosition, setTooltipPosition] = useState({ top: 0, left: 0 })
  const triggerRef = useRef<HTMLDivElement>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)
  const timeoutRef = useRef<NodeJS.Timeout>()

  const { isHovered } = useHover(triggerRef)

  useEffect(() => {
    if (disabled) {
      return
    }

    if (isHovered) {
      timeoutRef.current = setTimeout(() => {
        setIsVisible(true)
        updatePosition()
      }, delay)
    } else {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
      setIsVisible(false)
    }

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
    }
  }, [isHovered, delay, disabled])

  const updatePosition = () => {
    if (!triggerRef.current || !tooltipRef.current) {
      return
    }

    const triggerRect = triggerRef.current.getBoundingClientRect()
    const tooltipRect = tooltipRef.current.getBoundingClientRect()
    const scrollY = window.scrollY
    const scrollX = window.scrollX

    let top = 0
    let left = 0

    switch (position) {
      case 'top':
        top = triggerRect.top + scrollY - tooltipRect.height - 8
        left = triggerRect.left + scrollX + triggerRect.width / 2 - tooltipRect.width / 2
        break
      case 'bottom':
        top = triggerRect.bottom + scrollY + 8
        left = triggerRect.left + scrollX + triggerRect.width / 2 - tooltipRect.width / 2
        break
      case 'left':
        top = triggerRect.top + scrollY + triggerRect.height / 2 - tooltipRect.height / 2
        left = triggerRect.left + scrollX - tooltipRect.width - 8
        break
      case 'right':
        top = triggerRect.top + scrollY + triggerRect.height / 2 - tooltipRect.height / 2
        left = triggerRect.right + scrollX + 8
        break
    }

    setTooltipPosition({ top, left })
  }

  useEffect(() => {
    if (isVisible) {
      updatePosition()
      window.addEventListener('scroll', updatePosition, true)
      window.addEventListener('resize', updatePosition)

      return () => {
        window.removeEventListener('scroll', updatePosition, true)
        window.removeEventListener('resize', updatePosition)
      }
    }
  }, [isVisible])

  return (
    <>
      <div ref={triggerRef} className="inline-block">
        {children}
      </div>
      {isVisible && !disabled && typeof window !== 'undefined' && (
        <Portal>
          <div
            ref={tooltipRef}
            className={cn(
              'absolute z-50 px-2 py-1 text-sm text-white bg-gray-900 rounded shadow-lg pointer-events-none',
              className
            )}
            style={{
              top: `${tooltipPosition.top}px`,
              left: `${tooltipPosition.left}px`,
            }}
            role="tooltip"
          >
            {content}
            <div
              className={cn(
                'absolute w-2 h-2 bg-gray-900 transform rotate-45',
                position === 'top' && 'bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2',
                position === 'bottom' && 'top-0 left-1/2 -translate-x-1/2 -translate-y-1/2',
                position === 'left' && 'right-0 top-1/2 -translate-y-1/2 translate-x-1/2',
                position === 'right' && 'left-0 top-1/2 -translate-y-1/2 -translate-x-1/2'
              )}
            />
          </div>
        </Portal>
      )}
    </>
  )
}

export default Tooltip

