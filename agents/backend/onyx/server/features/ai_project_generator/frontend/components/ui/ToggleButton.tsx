'use client'

import { useState, useCallback, ReactNode } from 'react'
import { cn } from '@/lib/utils'
import Button, { ButtonProps } from './Button'

interface ToggleButtonProps extends Omit<ButtonProps, 'onClick'> {
  defaultPressed?: boolean
  pressed?: boolean
  onPressedChange?: (pressed: boolean) => void
  children: ReactNode
}

const ToggleButton = ({
  defaultPressed = false,
  pressed: controlledPressed,
  onPressedChange,
  variant = 'secondary',
  className,
  children,
  ...props
}: ToggleButtonProps) => {
  const [internalPressed, setInternalPressed] = useState(defaultPressed)
  const isControlled = controlledPressed !== undefined
  const pressed = isControlled ? controlledPressed : internalPressed

  const handleClick = useCallback(() => {
    const newPressed = !pressed
    if (!isControlled) {
      setInternalPressed(newPressed)
    }
    onPressedChange?.(newPressed)
  }, [pressed, isControlled, onPressedChange])

  return (
    <Button
      {...props}
      variant={pressed ? 'primary' : variant}
      onClick={handleClick}
      className={cn(pressed && 'ring-2 ring-primary-500', className)}
      aria-pressed={pressed}
    >
      {children}
    </Button>
  )
}

export default ToggleButton

