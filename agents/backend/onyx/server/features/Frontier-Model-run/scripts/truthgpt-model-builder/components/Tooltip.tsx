/**
 * Componente Tooltip
 * ==================
 * 
 * Componente de tooltip mejorado
 */

'use client'

import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useEventListener } from '@/lib/hooks/useEventListener'

export interface TooltipProps {
  content: React.ReactNode
  children: React.ReactElement
  position?: 'top' | 'bottom' | 'left' | 'right'
  delay?: number
  disabled?: boolean
  className?: string
}

export default function Tooltip({
  content,
  children,
  position = 'top',
  delay = 200,
  disabled = false,
  className = ''
}: TooltipProps) {
  const [isVisible, setIsVisible] = useState(false)
  const [tooltipPosition, setTooltipPosition] = useState({ top: 0, left: 0 })
  const triggerRef = useRef<HTMLElement>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)
  const timeoutRef = useRef<NodeJS.Timeout | null>(null)

  const showTooltip = () => {
    if (disabled) return

    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
    }

    timeoutRef.current = setTimeout(() => {
      if (triggerRef.current) {
        const rect = triggerRef.current.getBoundingClientRect()
        const tooltipRect = tooltipRef.current?.getBoundingClientRect() || { width: 0, height: 0 }

        let top = 0
        let left = 0

        switch (position) {
          case 'top':
            top = rect.top - (tooltipRect.height || 50) - 8
            left = rect.left + rect.width / 2
            break
          case 'bottom':
            top = rect.bottom + 8
            left = rect.left + rect.width / 2
            break
          case 'left':
            top = rect.top + rect.height / 2
            left = rect.left - (tooltipRect.width || 100) - 8
            break
          case 'right':
            top = rect.top + rect.height / 2
            left = rect.right + 8
            break
        }

        // Ajustar para que no se salga de la pantalla
        const padding = 8
        if (left < padding) left = padding
        if (left + (tooltipRect.width || 100) > window.innerWidth - padding) {
          left = window.innerWidth - (tooltipRect.width || 100) - padding
        }
        if (top < padding) top = padding
        if (top + (tooltipRect.height || 50) > window.innerHeight - padding) {
          top = window.innerHeight - (tooltipRect.height || 50) - padding
        }

        setTooltipPosition({ top, left })
        setIsVisible(true)
      }
    }, delay)
  }

  const hideTooltip = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
    }
    setIsVisible(false)
  }

  // Escuchar cambios de scroll y resize
  useEventListener('scroll', hideTooltip, typeof window !== 'undefined' ? window : null, { capture: true })
  useEventListener('resize', hideTooltip, typeof window !== 'undefined' ? window : null)

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
    }
  }, [])

  const childWithRef = React.cloneElement(children, {
    ref: triggerRef,
    onMouseEnter: showTooltip,
    onMouseLeave: hideTooltip,
    onFocus: showTooltip,
    onBlur: hideTooltip
  })

  return (
    <>
      {childWithRef}
      <AnimatePresence>
        {isVisible && (
          <motion.div
            ref={tooltipRef}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            transition={{ duration: 0.15 }}
            className={`
              fixed z-50 px-3 py-1.5 text-sm text-white bg-gray-900 dark:bg-gray-700 
              rounded-md shadow-lg pointer-events-none
              ${className}
            `}
            style={{
              top: `${tooltipPosition.top}px`,
              left: `${tooltipPosition.left}px`,
              transform: `translateX(${position === 'top' || position === 'bottom' ? '-50%' : '0'}) translateY(${position === 'left' || position === 'right' ? '-50%' : '0'})`
            }}
          >
            {content}
            {/* Arrow */}
            <div
              className={`
                absolute w-2 h-2 bg-gray-900 dark:bg-gray-700
                ${position === 'top' ? 'bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2 rotate-45' : ''}
                ${position === 'bottom' ? 'top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 rotate-45' : ''}
                ${position === 'left' ? 'right-0 top-1/2 -translate-y-1/2 translate-x-1/2 rotate-45' : ''}
                ${position === 'right' ? 'left-0 top-1/2 -translate-y-1/2 -translate-x-1/2 rotate-45' : ''}
              `}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}

