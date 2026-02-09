'use client'

import { ReactNode } from 'react'
import { X } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button, Chip } from '@/components/ui'

interface Filter {
  id: string
  label: string
  value: string
}

interface FilterBarProps {
  filters: Filter[]
  onRemoveFilter: (filterId: string) => void
  onClearAll?: () => void
  className?: string
  children?: ReactNode
}

const FilterBar = ({
  filters,
  onRemoveFilter,
  onClearAll,
  className,
  children,
}: FilterBarProps) => {
  if (filters.length === 0 && !children) {
    return null
  }

  return (
    <div className={cn('flex items-center gap-2 flex-wrap p-3 bg-gray-50 rounded-lg', className)}>
      {filters.map((filter) => (
        <Chip
          key={filter.id}
          label={`${filter.label}: ${filter.value}`}
          onRemove={() => onRemoveFilter(filter.id)}
          variant="default"
          size="sm"
        />
      ))}
      {children}
      {filters.length > 0 && onClearAll && (
        <Button
          variant="secondary"
          size="sm"
          onClick={onClearAll}
          leftIcon={<X className="w-4 h-4" />}
        >
          Clear All
        </Button>
      )}
    </div>
  )
}

export default FilterBar

