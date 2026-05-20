'use client'

import { ReactNode } from 'react'
import { cn } from '@/lib/utils'

interface GridProps {
  children: ReactNode
  cols?: 1 | 2 | 3 | 4 | 5 | 6 | 12
  gap?: 0 | 1 | 2 | 3 | 4 | 6 | 8
  className?: string
  responsive?: {
    sm?: 1 | 2 | 3 | 4 | 5 | 6 | 12
    md?: 1 | 2 | 3 | 4 | 5 | 6 | 12
    lg?: 1 | 2 | 3 | 4 | 5 | 6 | 12
    xl?: 1 | 2 | 3 | 4 | 5 | 6 | 12
  }
}

const Grid = ({
  children,
  cols = 1,
  gap = 4,
  className,
  responsive,
}: GridProps) => {
  const colsClasses = {
    1: 'grid-cols-1',
    2: 'grid-cols-2',
    3: 'grid-cols-3',
    4: 'grid-cols-4',
    5: 'grid-cols-5',
    6: 'grid-cols-6',
    12: 'grid-cols-12',
  }

  const gapClasses = {
    0: 'gap-0',
    1: 'gap-1',
    2: 'gap-2',
    3: 'gap-3',
    4: 'gap-4',
    6: 'gap-6',
    8: 'gap-8',
  }

  const responsiveClasses = responsive
    ? Object.entries(responsive)
        .map(([breakpoint, cols]) => {
          const colsMap: Record<number, string> = {
            1: `sm:grid-cols-1`,
            2: `sm:grid-cols-2`,
            3: `sm:grid-cols-3`,
            4: `sm:grid-cols-4`,
            5: `sm:grid-cols-5`,
            6: `sm:grid-cols-6`,
            12: `sm:grid-cols-12`,
          }

          const breakpointMap: Record<string, string> = {
            sm: 'sm',
            md: 'md',
            lg: 'lg',
            xl: 'xl',
          }

          return `${breakpointMap[breakpoint]}:${colsMap[cols]}`
        })
        .join(' ')
    : ''

  return (
    <div
      className={cn(
        'grid',
        colsClasses[cols],
        gapClasses[gap],
        responsiveClasses,
        className
      )}
    >
      {children}
    </div>
  )
}

export default Grid

