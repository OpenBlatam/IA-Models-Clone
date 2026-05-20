'use client'

import { useState, useCallback } from 'react'
import { cn } from '@/lib/utils'

interface TimePickerProps {
  value?: { hours: number; minutes: number }
  onChange?: (time: { hours: number; minutes: number }) => void
  className?: string
  format?: '12h' | '24h'
}

const TimePicker = ({
  value,
  onChange,
  className,
  format = '24h',
}: TimePickerProps) => {
  const [time, setTime] = useState(
    value || { hours: 0, minutes: 0 }
  )
  const [amPm, setAmPm] = useState<'AM' | 'PM'>('AM')

  const handleChange = useCallback(
    (updates: Partial<{ hours: number; minutes: number }>) => {
      let newHours = updates.hours !== undefined ? updates.hours : time.hours
      let newMinutes = updates.minutes !== undefined ? updates.minutes : time.minutes

      if (format === '12h') {
        if (amPm === 'PM' && newHours < 12) {
          newHours += 12
        } else if (amPm === 'AM' && newHours === 12) {
          newHours = 0
        }
      }

      const newTime = { hours: newHours, minutes: newMinutes }
      setTime(newTime)
      onChange?.(newTime)
    },
    [time, format, amPm, onChange]
  )

  const displayHours = format === '12h' ? (time.hours % 12 || 12) : time.hours
  const displayMinutes = time.minutes

  return (
    <div className={cn('flex items-center gap-2', className)}>
      <div className="flex items-center gap-1">
        <input
          type="number"
          min={format === '12h' ? 1 : 0}
          max={format === '12h' ? 12 : 23}
          value={displayHours}
          onChange={(e) => {
            const hours = parseInt(e.target.value) || 0
            if (format === '12h') {
              handleChange({ hours: amPm === 'PM' ? (hours === 12 ? 12 : hours + 12) : (hours === 12 ? 0 : hours) })
            } else {
              handleChange({ hours })
            }
          }}
          className="w-16 px-2 py-1 border border-gray-300 rounded text-center font-mono"
        />
        <span className="text-gray-500">:</span>
        <input
          type="number"
          min={0}
          max={59}
          value={displayMinutes}
          onChange={(e) => {
            const minutes = parseInt(e.target.value) || 0
            handleChange({ minutes: Math.min(59, Math.max(0, minutes)) })
          }}
          className="w-16 px-2 py-1 border border-gray-300 rounded text-center font-mono"
        />
      </div>
      {format === '12h' && (
        <div className="flex flex-col gap-1">
          <button
            onClick={() => {
              setAmPm('AM')
              if (time.hours >= 12) {
                handleChange({ hours: time.hours - 12 })
              }
            }}
            className={cn(
              'px-3 py-1 text-xs rounded',
              amPm === 'AM' ? 'bg-primary-600 text-white' : 'bg-gray-100 text-gray-700'
            )}
          >
            AM
          </button>
          <button
            onClick={() => {
              setAmPm('PM')
              if (time.hours < 12) {
                handleChange({ hours: time.hours + 12 })
              }
            }}
            className={cn(
              'px-3 py-1 text-xs rounded',
              amPm === 'PM' ? 'bg-primary-600 text-white' : 'bg-gray-100 text-gray-700'
            )}
          >
            PM
          </button>
        </div>
      )}
    </div>
  )
}

export default TimePicker

