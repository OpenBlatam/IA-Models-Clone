'use client'

import { ReactNode } from 'react'
import { cn } from '@/lib/utils'

interface SectionProps {
  title?: string
  description?: string
  children: ReactNode
  className?: string
  headerActions?: ReactNode
}

const Section = ({
  title,
  description,
  children,
  className,
  headerActions,
}: SectionProps) => {
  return (
    <section className={cn('space-y-4', className)}>
      {(title || description || headerActions) && (
        <div className="flex items-start justify-between">
          <div>
            {title && <h2 className="text-xl font-semibold text-gray-900 mb-1">{title}</h2>}
            {description && <p className="text-sm text-gray-600">{description}</p>}
          </div>
          {headerActions && <div className="flex items-center gap-2">{headerActions}</div>}
        </div>
      )}
      <div>{children}</div>
    </section>
  )
}

export default Section

