'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { cn } from '@/lib/utils'

interface SliderProps {
  value?: number
  defaultValue?: number
  min?: number
  max?: number
  step?: number
  onChange?: (value: number) => void
  className?: string
  disabled?: boolean
  label?: string
  showValue?: boolean
}

const Slider = ({
  value: controlledValue,
  defaultValue = 0,
  min = 0,
  max = 100,
  step = 1,
  onChange,
  className,
  disabled = false,
  label,
  showValue = false,
}: SliderProps) => {
  const [internalValue, setInternalValue] = useState(defaultValue)
  const sliderRef = useRef<HTMLDivElement>(null)
  const isControlled = controlledValue !== undefined
  const value = isControlled ? controlledValue : internalValue

  const percentage = ((value - min) / (max - min)) * 100

  const handleChange = useCallback(
    (newValue: number) => {
      const clampedValue = Math.min(Math.max(newValue, min), max)
      if (!isControlled) {
        setInternalValue(clampedValue)
      }
      onChange?.(clampedValue)
    },
    [min, max, isControlled, onChange]
  )

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (disabled) {
        return
      }

      const handleMouseMove = (moveEvent: MouseEvent) => {
        if (!sliderRef.current) {
          return
        }

        const rect = sliderRef.current.getBoundingClientRect()
        const x = moveEvent.clientX - rect.left
        const percentage = Math.min(Math.max(x / rect.width, 0), 1)
        const newValue = min + percentage * (max - min)
        const steppedValue = Math.round(newValue / step) * step
        handleChange(steppedValue)
      }

      const handleMouseUp = () => {
        document.removeEventListener('mousemove', handleMouseMove)
        document.removeEventListener('mouseup', handleMouseUp)
      }

      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)

      const rect = sliderRef.current?.getBoundingClientRect()
      if (rect) {
        const x = e.clientX - rect.left
        const percentage = Math.min(Math.max(x / rect.width, 0), 1)
        const newValue = min + percentage * (max - min)
        const steppedValue = Math.round(newValue / step) * step
        handleChange(steppedValue)
      }
    },
    [disabled, min, max, step, handleChange]
  )

  return (
    <div className={cn('w-full', className)}>
      {label && (
        <div className="flex items-center justify-between mb-2">
          <label className="text-sm font-medium text-gray-700">{label}</label>
          {showValue && <span className="text-sm text-gray-500">{value}</span>}
        </div>
      )}
      <div
        ref={sliderRef}
        className={cn(
          'relative h-2 w-full rounded-full bg-gray-200 cursor-pointer',
          disabled && 'opacity-50 cursor-not-allowed'
        )}
        onMouseDown={handleMouseDown}
        role="slider"
        aria-valuenow={value}
        aria-valuemin={min}
        aria-valuemax={max}
        aria-disabled={disabled}
        tabIndex={disabled ? -1 : 0}
      >
        <div
          className="absolute h-full bg-primary-600 rounded-full transition-all"
          style={{ width: `${percentage}%` }}
        />
        <div
          className="absolute w-4 h-4 bg-white border-2 border-primary-600 rounded-full top-1/2 -translate-y-1/2 -translate-x-1/2 transition-all hover:scale-110"
          style={{ left: `${percentage}%` }}
        />
      </div>
    </div>
  )
}

export default Slider

