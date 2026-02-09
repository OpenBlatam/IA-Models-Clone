'use client'

import { ReactNode } from 'react'
import { cn } from '@/lib/utils'

interface StackProps {
  children: ReactNode
  direction?: 'row' | 'column'
  spacing?: 0 | 1 | 2 | 3 | 4 | 6 | 8
  align?: 'start' | 'center' | 'end' | 'stretch'
  justify?: 'start' | 'center' | 'end' | 'between' | 'around' | 'evenly'
  className?: string
}

const Stack = ({
  children,
  direction = 'column',
  spacing = 2,
  align,
  justify,
  className,
}: StackProps) => {
  const spacingClasses = {
    0: 'gap-0',
    1: 'gap-1',
    2: 'gap-2',
    3: 'gap-3',
    4: 'gap-4',
    6: 'gap-6',
    8: 'gap-8',
  }

  const alignClasses = {
    start: 'items-start',
    center: 'items-center',
    end: 'items-end',
    stretch: 'items-stretch',
  }

  const justifyClasses = {
    start: 'justify-start',
    center: 'justify-center',
    end: 'justify-end',
    between: 'justify-between',
    around: 'justify-around',
    evenly: 'justify-evenly',
  }

  return (
    <div
      className={cn(
        'flex',
        direction === 'row' ? 'flex-row' : 'flex-col',
        spacingClasses[spacing],
        align && alignClasses[align],
        justify && justifyClasses[justify],
        className
      )}
    >
      {children}
    </div>
  )
}

export default Stack

