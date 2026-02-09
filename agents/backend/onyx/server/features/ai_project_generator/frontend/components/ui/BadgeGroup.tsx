'use client'

import { ReactNode } from 'react'
import { cn } from '@/lib/utils'
import Badge from './Badge'

interface BadgeGroupProps {
  badges: Array<{
    id: string
    label: string
    variant?: 'default' | 'primary' | 'success' | 'warning' | 'danger'
    onRemove?: () => void
  }>
  className?: string
  maxVisible?: number
  onShowMore?: () => void
}

const BadgeGroup = ({
  badges,
  className,
  maxVisible,
  onShowMore,
}: BadgeGroupProps) => {
  const visibleBadges = maxVisible ? badges.slice(0, maxVisible) : badges
  const hiddenCount = maxVisible ? badges.length - maxVisible : 0

  return (
    <div className={cn('flex flex-wrap gap-2', className)}>
      {visibleBadges.map((badge) => (
        <Badge
          key={badge.id}
          label={badge.label}
          variant={badge.variant}
          onRemove={badge.onRemove}
        />
      ))}
      {hiddenCount > 0 && (
        <Badge
          label={`+${hiddenCount} more`}
          variant="default"
          onClick={onShowMore}
          className="cursor-pointer"
        />
      )}
    </div>
  )
}

export default BadgeGroup

