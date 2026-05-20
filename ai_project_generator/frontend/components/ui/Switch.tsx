'use client'

import { useCallback } from 'react'
import { cn } from '@/lib/utils'

interface SwitchProps {
  checked: boolean
  onChange: (checked: boolean) => void
  label?: string
  disabled?: boolean
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

const Switch = ({ checked, onChange, label, disabled = false, size = 'md', className }: SwitchProps) => {
  const handleToggle = useCallback(() => {
    if (!disabled) {
      onChange(!checked)
    }
  }, [checked, onChange, disabled])

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if ((e.key === 'Enter' || e.key === ' ') && !disabled) {
        e.preventDefault()
        handleToggle()
      }
    },
    [handleToggle, disabled]
  )

  const sizeClasses = {
    sm: 'w-8 h-4',
    md: 'w-11 h-6',
    lg: 'w-14 h-7',
  }

  const thumbSizeClasses = {
    sm: 'w-3 h-3',
    md: 'w-5 h-5',
    lg: 'w-6 h-6',
  }

  const translateClasses = {
    sm: 'translate-x-4',
    md: 'translate-x-5',
    lg: 'translate-x-7',
  }

  return (
    <label className={cn('flex items-center gap-3 cursor-pointer', disabled && 'opacity-50 cursor-not-allowed', className)}>
      <div
        className={cn(
          'relative inline-flex items-center rounded-full transition-colors focus-within:outline-none focus-within:ring-2 focus-within:ring-primary-500 focus-within:ring-offset-2',
          sizeClasses[size],
          checked ? 'bg-primary-600' : 'bg-gray-300',
          disabled && 'cursor-not-allowed'
        )}
        onClick={handleToggle}
        onKeyDown={handleKeyDown}
        tabIndex={disabled ? -1 : 0}
        role="switch"
        aria-checked={checked}
        aria-disabled={disabled}
      >
        <span
          className={cn(
            'inline-block rounded-full bg-white transition-transform',
            thumbSizeClasses[size],
            checked ? translateClasses[size] : 'translate-x-0.5'
          )}
        />
      </div>
      {label && <span className="text-sm text-gray-700">{label}</span>}
    </label>
  )
}

export default Switch

