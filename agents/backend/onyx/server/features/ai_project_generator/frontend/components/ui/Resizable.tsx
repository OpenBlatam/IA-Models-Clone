'use client'

import { useState, useCallback, useRef, ReactNode, useEffect } from 'react'
import { GripVertical } from 'lucide-react'
import { cn } from '@/lib/utils'

interface ResizableProps {
  children: ReactNode
  defaultWidth?: number
  minWidth?: number
  maxWidth?: number
  className?: string
  direction?: 'horizontal' | 'vertical'
}

const Resizable = ({
  children,
  defaultWidth = 300,
  minWidth = 200,
  maxWidth = 800,
  className,
  direction = 'horizontal',
}: ResizableProps) => {
  const [width, setWidth] = useState(defaultWidth)
  const [isResizing, setIsResizing] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  const handleMouseDown = useCallback(() => {
    setIsResizing(true)
  }, [])

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isResizing || !containerRef.current) {
        return
      }

      const rect = containerRef.current.getBoundingClientRect()
      const newWidth = direction === 'horizontal' ? e.clientX - rect.left : e.clientY - rect.top
      const clampedWidth = Math.min(Math.max(newWidth, minWidth), maxWidth)
      setWidth(clampedWidth)
    },
    [isResizing, minWidth, maxWidth, direction]
  )

  const handleMouseUp = useCallback(() => {
    setIsResizing(false)
  }, [])

  useEffect(() => {
    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = direction === 'horizontal' ? 'col-resize' : 'row-resize'
      document.body.style.userSelect = 'none'

      return () => {
        document.removeEventListener('mousemove', handleMouseMove)
        document.removeEventListener('mouseup', handleMouseUp)
        document.body.style.cursor = ''
        document.body.style.userSelect = ''
      }
    }
  }, [isResizing, handleMouseMove, handleMouseUp, direction])

  return (
    <div
      ref={containerRef}
      className={cn('flex', direction === 'horizontal' ? 'flex-row' : 'flex-col', className)}
    >
      <div
        className={cn('overflow-auto', direction === 'horizontal' ? 'flex-shrink-0' : 'flex-shrink-0')}
        style={direction === 'horizontal' ? { width: `${width}px` } : { height: `${width}px` }}
      >
        {children}
      </div>
      <div
        className={cn(
          'flex items-center justify-center bg-gray-100 hover:bg-gray-200 cursor-col-resize transition-colors',
          direction === 'horizontal' ? 'w-1' : 'h-1 w-full',
          isResizing && 'bg-primary-500'
        )}
        onMouseDown={handleMouseDown}
        role="separator"
        aria-orientation={direction}
        aria-label="Resize"
      >
        <GripVertical
          className={cn(
            'text-gray-400',
            direction === 'vertical' && 'rotate-90'
          )}
        />
      </div>
    </div>
  )
}

export default Resizable

