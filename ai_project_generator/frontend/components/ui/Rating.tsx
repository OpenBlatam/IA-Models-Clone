'use client'

import { useState, useCallback } from 'react'
import { Star } from 'lucide-react'
import { cn } from '@/lib/utils'

interface RatingProps {
  value?: number
  max?: number
  onChange?: (value: number) => void
  readonly?: boolean
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

const Rating = ({ value = 0, max = 5, onChange, readonly = false, size = 'md', className }: RatingProps) => {
  const [hoverValue, setHoverValue] = useState<number | null>(null)

  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-5 h-5',
    lg: 'w-6 h-6',
  }

  const handleClick = useCallback(
    (index: number) => {
      if (!readonly && onChange) {
        onChange(index + 1)
      }
    },
    [readonly, onChange]
  )

  const handleMouseEnter = useCallback((index: number) => {
    if (!readonly) {
      setHoverValue(index + 1)
    }
  }, [readonly])

  const handleMouseLeave = useCallback(() => {
    if (!readonly) {
      setHoverValue(null)
    }
  }, [readonly])

  const displayValue = hoverValue ?? value

  return (
    <div className={cn('flex items-center gap-1', className)} role="img" aria-label={`Rating: ${value} out of ${max}`}>
      {Array.from({ length: max }).map((_, index) => {
        const isFilled = index < displayValue
        return (
          <button
            key={index}
            type="button"
            onClick={() => handleClick(index)}
            onMouseEnter={() => handleMouseEnter(index)}
            onMouseLeave={handleMouseLeave}
            disabled={readonly}
            className={cn(
              'transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 rounded',
              !readonly && 'cursor-pointer',
              readonly && 'cursor-default'
            )}
            tabIndex={readonly ? -1 : 0}
            aria-label={`Rate ${index + 1} out of ${max}`}
          >
            <Star
              className={cn(
                sizeClasses[size],
                isFilled ? 'fill-yellow-400 text-yellow-400' : 'fill-gray-200 text-gray-200'
              )}
            />
          </button>
        )
      })}
    </div>
  )
}

export default Rating

