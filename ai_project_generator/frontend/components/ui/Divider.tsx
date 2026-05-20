'use client'

import { ReactNode } from 'react'
import { cn } from '@/lib/utils'

interface DividerProps {
  orientation?: 'horizontal' | 'vertical'
  text?: string
  className?: string
  children?: ReactNode
}

const Divider = ({
  orientation = 'horizontal',
  text,
  className,
  children,
}: DividerProps) => {
  if (orientation === 'vertical') {
    return (
      <div
        className={cn('w-px bg-gray-300 self-stretch', className)}
        role="separator"
        aria-orientation="vertical"
      />
    )
  }

  return (
    <div className={cn('flex items-center gap-4 my-4', className)}>
      <div className="flex-1 h-px bg-gray-300" />
      {(text || children) && (
        <span className="text-sm text-gray-500 px-2">{text || children}</span>
      )}
      <div className="flex-1 h-px bg-gray-300" />
    </div>
  )
}

export default Divider
