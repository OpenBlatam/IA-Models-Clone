'use client'

import { useState, useRef, ReactNode, useEffect, useCallback } from 'react'
import { X } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useClickOutside } from '@/hooks/ui'
import Portal from './Portal'
import Button from './Button'

interface PopoverProps {
  trigger: ReactNode
  content: ReactNode
  position?: 'top' | 'bottom' | 'left' | 'right'
  className?: string
  closeOnClickOutside?: boolean
  showCloseButton?: boolean
}

const Popover = ({
  trigger,
  content,
  position = 'bottom',
  className,
  closeOnClickOutside = true,
  showCloseButton = false,
}: PopoverProps) => {
  const [isOpen, setIsOpen] = useState(false)
  const [popoverPosition, setPopoverPosition] = useState({ top: 0, left: 0 })
  const triggerRef = useRef<HTMLDivElement>(null)
  const popoverRef = useRef<HTMLDivElement>(null)

  const handleToggle = useCallback(() => {
    setIsOpen((prev) => !prev)
  }, [])

  const handleClose = useCallback(() => {
    setIsOpen(false)
  }, [])

  const updatePosition = useCallback(() => {
    if (!triggerRef.current || !popoverRef.current) {
      return
    }

    const triggerRect = triggerRef.current.getBoundingClientRect()
    const popoverRect = popoverRef.current.getBoundingClientRect()
    const scrollY = window.scrollY
    const scrollX = window.scrollX

    let top = 0
    let left = 0

    switch (position) {
      case 'top':
        top = triggerRect.top + scrollY - popoverRect.height - 8
        left = triggerRect.left + scrollX + triggerRect.width / 2 - popoverRect.width / 2
        break
      case 'bottom':
        top = triggerRect.bottom + scrollY + 8
        left = triggerRect.left + scrollX + triggerRect.width / 2 - popoverRect.width / 2
        break
      case 'left':
        top = triggerRect.top + scrollY + triggerRect.height / 2 - popoverRect.height / 2
        left = triggerRect.left + scrollX - popoverRect.width - 8
        break
      case 'right':
        top = triggerRect.top + scrollY + triggerRect.height / 2 - popoverRect.height / 2
        left = triggerRect.right + scrollX + 8
        break
    }

    setPopoverPosition({ top, left })
  }, [position])

  useEffect(() => {
    if (isOpen) {
      updatePosition()
      window.addEventListener('scroll', updatePosition, true)
      window.addEventListener('resize', updatePosition)

      return () => {
        window.removeEventListener('scroll', updatePosition, true)
        window.removeEventListener('resize', updatePosition)
      }
    }
  }, [isOpen, updatePosition])

  const containerRef = useClickOutside<HTMLDivElement>(() => {
    if (closeOnClickOutside) {
      handleClose()
    }
  })

  return (
    <>
      <div ref={triggerRef} onClick={handleToggle} className="inline-block">
        {trigger}
      </div>
      {isOpen && typeof window !== 'undefined' && (
        <Portal>
          <div
            ref={(node) => {
              popoverRef.current = node
              if (containerRef) {
                if (typeof containerRef === 'function') {
                  containerRef(node)
                } else {
                  containerRef.current = node
                }
              }
            }}
            className={cn(
              'absolute z-50 bg-white rounded-lg shadow-lg border border-gray-200 p-4',
              className
            )}
            style={{
              top: `${popoverPosition.top}px`,
              left: `${popoverPosition.left}px`,
            }}
          >
            {showCloseButton && (
              <div className="flex justify-end mb-2">
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={handleClose}
                  leftIcon={<X className="w-4 h-4" />}
                />
              </div>
            )}
            {content}
          </div>
        </Portal>
      )}
    </>
  )
}

export default Popover
