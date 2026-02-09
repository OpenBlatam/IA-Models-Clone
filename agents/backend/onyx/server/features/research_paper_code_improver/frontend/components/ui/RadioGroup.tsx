'use client'

import React from 'react'
import { clsx } from 'clsx'

interface RadioOption {
  value: string
  label: string
  description?: string
  disabled?: boolean
}

interface RadioGroupProps {
  options: RadioOption[]
  value?: string
  onChange?: (value: string) => void
  label?: string
  name: string
  className?: string
  orientation?: 'horizontal' | 'vertical'
}

const RadioGroup: React.FC<RadioGroupProps> = ({
  options,
  value,
  onChange,
  label,
  name,
  className,
  orientation = 'vertical',
}) => {
  return (
    <div className={clsx('w-full', className)}>
      {label && (
        <label className="block text-sm font-medium text-gray-700 mb-2">
          {label}
        </label>
      )}
      <div
        className={clsx(
          'space-y-2',
          orientation === 'horizontal' && 'flex flex-wrap gap-4'
        )}
        role="radiogroup"
        aria-label={label}
      >
        {options.map((option) => (
          <label
            key={option.value}
            className={clsx(
              'flex items-start gap-3 cursor-pointer group',
              option.disabled && 'opacity-50 cursor-not-allowed',
              orientation === 'horizontal' && 'flex-col'
            )}
          >
            <div className="relative flex items-center justify-center flex-shrink-0 mt-0.5">
              <input
                type="radio"
                name={name}
                value={option.value}
                checked={value === option.value}
                onChange={() => !option.disabled && onChange?.(option.value)}
                disabled={option.disabled}
                className="sr-only"
              />
              <div
                className={clsx(
                  'w-5 h-5 border-2 rounded-full transition-all flex items-center justify-center',
                  value === option.value
                    ? 'border-primary-600 bg-primary-50'
                    : 'border-gray-300 group-hover:border-primary-400',
                  option.disabled && 'opacity-50'
                )}
              >
                {value === option.value && (
                  <div className="w-2.5 h-2.5 bg-primary-600 rounded-full" />
                )}
              </div>
            </div>
            <div className="flex-1">
              <span
                className={clsx(
                  'text-sm font-medium',
                  value === option.value ? 'text-gray-900' : 'text-gray-700',
                  option.disabled && 'opacity-50'
                )}
              >
                {option.label}
              </span>
              {option.description && (
                <p className="text-xs text-gray-500 mt-0.5">
                  {option.description}
                </p>
              )}
            </div>
          </label>
        ))}
      </div>
    </div>
  )
}

export default RadioGroup




