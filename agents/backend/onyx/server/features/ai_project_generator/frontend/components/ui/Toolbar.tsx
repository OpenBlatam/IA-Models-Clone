'use client'

import { ReactNode } from 'react'
import { cn } from '@/lib/utils'

interface ToolbarProps {
  children: ReactNode
  className?: string
  position?: 'top' | 'bottom' | 'left' | 'right'
}

const Toolbar = ({ children, className, position = 'top' }: ToolbarProps) => {
  const positionClasses = {
    top: 'top-0 left-0 right-0',
    bottom: 'bottom-0 left-0 right-0',
    left: 'left-0 top-0 bottom-0',
    right: 'right-0 top-0 bottom-0',
  }

  return (
    <div
      className={cn(
        'bg-white border-b border-gray-200 px-4 py-3 flex items-center gap-3',
        positionClasses[position],
        className
      )}
      role="toolbar"
      aria-label="Toolbar"
    >
      {children}
    </div>
  )
}

export default Toolbar

