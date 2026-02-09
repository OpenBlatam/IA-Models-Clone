'use client'

import { cn } from '@/lib/utils'

interface SeparatorProps {
  orientation?: 'horizontal' | 'vertical'
  className?: string
}

const Separator = ({ orientation = 'horizontal', className }: SeparatorProps) => {
  return (
    <div
      className={cn(
        'shrink-0 bg-gray-200',
        orientation === 'horizontal' ? 'h-px w-full' : 'h-full w-px',
        className
      )}
      role="separator"
      aria-orientation={orientation}
    />
  )
}

export default Separator

