'use client'

import { ReactNode } from 'react'
import { cn } from '@/lib/utils'
import Button, { ButtonProps } from './Button'

interface ButtonGroupProps {
  buttons: Array<ButtonProps & { key: string }>
  orientation?: 'horizontal' | 'vertical'
  className?: string
  attached?: boolean
}

const ButtonGroup = ({
  buttons,
  orientation = 'horizontal',
  className,
  attached = false,
}: ButtonGroupProps) => {
  return (
    <div
      className={cn(
        'flex',
        orientation === 'horizontal' ? 'flex-row' : 'flex-col',
        attached && 'gap-0',
        !attached && 'gap-2',
        className
      )}
      role="group"
    >
      {buttons.map((button, index) => {
        const { key, ...buttonProps } = button
        return (
          <Button
            key={key}
            {...buttonProps}
            className={cn(
              attached && orientation === 'horizontal' && index > 0 && '-ml-px rounded-l-none',
              attached && orientation === 'horizontal' && index < buttons.length - 1 && 'rounded-r-none',
              attached && orientation === 'vertical' && index > 0 && '-mt-px rounded-t-none',
              attached && orientation === 'vertical' && index < buttons.length - 1 && 'rounded-b-none',
              buttonProps.className
            )}
          />
        )
      })}
    </div>
  )
}

export default ButtonGroup

