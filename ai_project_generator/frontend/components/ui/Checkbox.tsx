'use client'

import clsx from 'clsx'
import type { InputHTMLAttributes } from 'react'

interface CheckboxProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type'> {
  label?: string
  error?: string
  helperText?: string
}

const Checkbox = ({ label, error, helperText, className, id, ...props }: CheckboxProps) => {
  const checkboxId = id || `checkbox-${Math.random().toString(36).substr(2, 9)}`

  return (
    <div className="w-full">
      <div className="flex items-start">
        <div className="flex items-center h-5">
          <input
            id={checkboxId}
            type="checkbox"
            className={clsx(
              'rounded border-gray-300 text-primary-600 focus:ring-primary-500',
              error && 'border-red-300',
              className
            )}
            aria-invalid={error ? 'true' : 'false'}
            aria-describedby={
              error ? `${checkboxId}-error` : helperText ? `${checkboxId}-helper` : undefined
            }
            {...props}
          />
        </div>
        {label && (
          <div className="ml-3 text-sm">
            <label htmlFor={checkboxId} className="font-medium text-gray-700">
              {label}
            </label>
            {helperText && !error && (
              <p id={`${checkboxId}-helper`} className="text-gray-500 mt-1">
                {helperText}
              </p>
            )}
            {error && (
              <p id={`${checkboxId}-error`} className="text-red-600 mt-1" role="alert">
                {error}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default Checkbox

