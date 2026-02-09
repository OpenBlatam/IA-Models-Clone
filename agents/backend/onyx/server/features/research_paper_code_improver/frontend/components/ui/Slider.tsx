'use client'

import React from 'react'
import { clsx } from 'clsx'

interface SliderProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'type'> {
  label?: string
  helperText?: string
  showValue?: boolean
  min?: number
  max?: number
  step?: number
}

const Slider = React.forwardRef<HTMLInputElement, SliderProps>(
  ({ className, label, helperText, showValue = true, min = 0, max = 100, step = 1, value, ...props }, ref) => {
    const percentage = ((Number(value) - min) / (max - min)) * 100

    return (
      <div className={clsx('w-full', className)}>
        {label && (
          <div className="flex items-center justify-between mb-2">
            <label className="block text-sm font-medium text-gray-700">
              {label}
              {props.required && <span className="text-red-500 ml-1">*</span>}
            </label>
            {showValue && (
              <span className="text-sm font-medium text-gray-900">{value}</span>
            )}
          </div>
        )}
        <div className="relative">
          <input
            ref={ref}
            type="range"
            min={min}
            max={max}
            step={step}
            value={value}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
            style={{
              background: `linear-gradient(to right, #667eea 0%, #667eea ${percentage}%, #e5e7eb ${percentage}%, #e5e7eb 100%)`,
            }}
            {...props}
          />
        </div>
        {helperText && (
          <p className="mt-1 text-xs text-gray-500">{helperText}</p>
        )}
      </div>
    )
  }
)

Slider.displayName = 'Slider'

export default Slider




