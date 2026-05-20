'use client'

import { ReactNode } from 'react'
import { cn } from '@/lib/utils'

interface ScrollAreaProps {
  children: ReactNode
  className?: string
  maxHeight?: string
}

const ScrollArea = ({ children, className, maxHeight = '400px' }: ScrollAreaProps) => {
  return (
    <div
      className={cn('overflow-auto', className)}
      style={{ maxHeight }}
    >
      <div className="pr-4">{children}</div>
    </div>
  )
}

export default ScrollArea

