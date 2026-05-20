'use client'

import clsx from 'clsx'
import type { TextareaHTMLAttributes } from 'react'

interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string
  error?: string
  helperText?: string
  showCharCount?: boolean
  maxLength?: number
}

const Textarea = ({
  label,
  error,
  helperText,
  showCharCount = false,
  maxLength,
  className,
  id,
  value,
  ...props
}: TextareaProps) => {
  const inputId = id || `textarea-${Math.random().toString(36).substr(2, 9)}`
  const charCount = typeof value === 'string' ? value.length : 0

  return (
    <div className="w-full">
      {label && (
        <label htmlFor={inputId} className="block text-sm font-medium text-gray-700 mb-2">
          {label}
        </label>
      )}
      <textarea
        id={inputId}
        value={value}
        maxLength={maxLength}
        className={clsx(
          'input w-full',
          error && 'border-red-300 focus:ring-red-500 focus:border-red-500',
          className
        )}
        aria-invalid={error ? 'true' : 'false'}
        aria-describedby={
          error
            ? `${inputId}-error`
            : helperText
            ? `${inputId}-helper`
            : showCharCount
            ? `${inputId}-count`
            : undefined
        }
        {...props}
      />
      <div className="flex items-center justify-between mt-1">
        <div>
          {error && (
            <p id={`${inputId}-error`} className="text-sm text-red-600" role="alert">
              {error}
            </p>
          )}
          {helperText && !error && (
            <p id={`${inputId}-helper`} className="text-sm text-gray-500">
              {helperText}
            </p>
          )}
        </div>
        {showCharCount && maxLength && (
          <p
            id={`${inputId}-count`}
            className={clsx(
              'text-sm',
              charCount > maxLength * 0.9 ? 'text-yellow-600' : 'text-gray-500'
            )}
          >
            {charCount}/{maxLength}
          </p>
        )}
      </div>
    </div>
  )
}

export default Textarea

