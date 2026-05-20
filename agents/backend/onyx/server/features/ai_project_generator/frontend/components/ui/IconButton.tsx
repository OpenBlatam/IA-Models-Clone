'use client'

import { ReactNode } from 'react'
import { cn } from '@/lib/utils'
import Button, { ButtonProps } from './Button'

interface IconButtonProps extends Omit<ButtonProps, 'leftIcon' | 'rightIcon' | 'children'> {
  icon: ReactNode
  'aria-label': string
  size?: 'sm' | 'md' | 'lg'
}

const IconButton = ({
  icon,
  size = 'md',
  className,
  ...props
}: IconButtonProps) => {
  const sizeClasses = {
    sm: 'p-1.5',
    md: 'p-2',
    lg: 'p-3',
  }

  return (
    <Button
      {...props}
      className={cn(sizeClasses[size], className)}
      variant={props.variant || 'secondary'}
    >
      {icon}
    </Button>
  )
}

export default IconButton

