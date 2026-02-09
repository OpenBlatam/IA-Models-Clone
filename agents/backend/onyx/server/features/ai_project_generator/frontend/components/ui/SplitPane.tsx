'use client'

import { useState, useCallback, useRef, ReactNode, useEffect } from 'react'
import { cn } from '@/lib/utils'

interface SplitPaneProps {
  children: [ReactNode, ReactNode]
  defaultSize?: number
  minSize?: number
  maxSize?: number
  direction?: 'horizontal' | 'vertical'
  className?: string
  resizerClassName?: string
}

const SplitPane = ({
  children,
  defaultSize = 50,
  minSize = 10,
  maxSize = 90,
  direction = 'horizontal',
  className,
  resizerClassName,
}: SplitPaneProps) => {
  const [size, setSize] = useState(defaultSize)
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
      const newSize =
        direction === 'horizontal'
          ? ((e.clientX - rect.left) / rect.width) * 100
          : ((e.clientY - rect.top) / rect.height) * 100

      const clampedSize = Math.min(Math.max(newSize, minSize), maxSize)
      setSize(clampedSize)
    },
    [isResizing, minSize, maxSize, direction]
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
      className={cn(
        'flex',
        direction === 'horizontal' ? 'flex-row' : 'flex-col',
        className
      )}
    >
      <div
        className={cn('overflow-auto', direction === 'horizontal' ? 'flex-shrink-0' : 'flex-shrink-0')}
        style={direction === 'horizontal' ? { width: `${size}%` } : { height: `${size}%` }}
      >
        {children[0]}
      </div>
      <div
        className={cn(
          'flex items-center justify-center bg-gray-200 hover:bg-gray-300 cursor-col-resize transition-colors',
          direction === 'horizontal' ? 'w-1' : 'h-1 w-full',
          isResizing && 'bg-primary-500',
          resizerClassName
        )}
        onMouseDown={handleMouseDown}
        role="separator"
        aria-orientation={direction}
        aria-label="Resize split pane"
      />
      <div
        className={cn('overflow-auto flex-1')}
        style={direction === 'horizontal' ? {} : {}}
      >
        {children[1]}
      </div>
    </div>
  )
}

export default SplitPane

