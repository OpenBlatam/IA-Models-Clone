'use client'

import { ReactNode } from 'react'
import { Plus } from 'lucide-react'
import { cn } from '@/lib/utils'
import IconButton from './IconButton'

interface FloatingActionButtonProps {
  icon?: ReactNode
  onClick: () => void
  position?: 'bottom-right' | 'bottom-left' | 'top-right' | 'top-left'
  className?: string
  'aria-label': string
  size?: 'sm' | 'md' | 'lg'
}

const FloatingActionButton = ({
  icon = <Plus className="w-6 h-6" />,
  onClick,
  position = 'bottom-right',
  className,
  size = 'lg',
  'aria-label': ariaLabel,
}: FloatingActionButtonProps) => {
  const positionClasses = {
    'bottom-right': 'bottom-6 right-6',
    'bottom-left': 'bottom-6 left-6',
    'top-right': 'top-6 right-6',
    'top-left': 'top-6 left-6',
  }

  const sizeClasses = {
    sm: 'w-12 h-12',
    md: 'w-14 h-14',
    lg: 'w-16 h-16',
  }

  return (
    <IconButton
      icon={icon}
      onClick={onClick}
      aria-label={ariaLabel}
      size={size}
      variant="primary"
      className={cn(
        'fixed z-50 rounded-full shadow-lg hover:shadow-xl transition-shadow',
        positionClasses[position],
        sizeClasses[size],
        className
      )}
    />
  )
}

export default FloatingActionButton

