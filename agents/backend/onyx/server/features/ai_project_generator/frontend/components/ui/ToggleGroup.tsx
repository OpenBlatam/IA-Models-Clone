'use client'

import { useState, useCallback, ReactNode } from 'react'
import { cn } from '@/lib/utils'
import ToggleButton from './ToggleButton'

interface ToggleOption {
  value: string
  label: ReactNode
  disabled?: boolean
}

interface ToggleGroupProps {
  options: ToggleOption[]
  value?: string | string[]
  defaultValue?: string | string[]
  onChange?: (value: string | string[]) => void
  multiple?: boolean
  className?: string
  variant?: 'default' | 'outlined'
}

const ToggleGroup = ({
  options,
  value: controlledValue,
  defaultValue,
  onChange,
  multiple = false,
  className,
  variant = 'default',
}: ToggleGroupProps) => {
  const [internalValue, setInternalValue] = useState<string | string[]>(
    defaultValue || (multiple ? [] : '')
  )
  const isControlled = controlledValue !== undefined
  const value = isControlled ? controlledValue : internalValue

  const handleToggle = useCallback(
    (optionValue: string) => {
      let newValue: string | string[]

      if (multiple) {
        const currentValues = Array.isArray(value) ? value : []
        if (currentValues.includes(optionValue)) {
          newValue = currentValues.filter((v) => v !== optionValue)
        } else {
          newValue = [...currentValues, optionValue]
        }
      } else {
        newValue = value === optionValue ? '' : optionValue
      }

      if (!isControlled) {
        setInternalValue(newValue)
      }
      onChange?.(newValue)
    },
    [value, multiple, isControlled, onChange]
  )

  const isSelected = (optionValue: string) => {
    if (multiple) {
      return Array.isArray(value) && value.includes(optionValue)
    }
    return value === optionValue
  }

  return (
    <div
      className={cn(
        'inline-flex gap-0 rounded-lg border border-gray-300 overflow-hidden',
        variant === 'outlined' && 'border-2',
        className
      )}
      role="group"
    >
      {options.map((option, index) => (
        <ToggleButton
          key={option.value}
          pressed={isSelected(option.value)}
          onPressedChange={() => handleToggle(option.value)}
          disabled={option.disabled}
          variant="secondary"
          className={cn(
            'rounded-none border-0',
            index === 0 && 'rounded-l-lg',
            index === options.length - 1 && 'rounded-r-lg',
            index > 0 && '-ml-px'
          )}
        >
          {option.label}
        </ToggleButton>
      ))}
    </div>
  )
}

export default ToggleGroup

