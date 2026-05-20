'use client'

import { useState, useCallback } from 'react'
import { Calendar } from '@/components/ui'
import { cn } from '@/lib/utils'
import { formatDate } from '@/lib/utils'

interface DateRangePickerProps {
  startDate?: Date
  endDate?: Date
  onChange?: (startDate: Date | null, endDate: Date | null) => void
  className?: string
  minDate?: Date
  maxDate?: Date
}

const DateRangePicker = ({
  startDate,
  endDate,
  onChange,
  className,
  minDate,
  maxDate,
}: DateRangePickerProps) => {
  const [internalStartDate, setInternalStartDate] = useState<Date | null>(startDate || null)
  const [internalEndDate, setInternalEndDate] = useState<Date | null>(endDate || null)

  const handleStartDateChange = useCallback(
    (date: Date) => {
      const newStartDate = date
      setInternalStartDate(newStartDate)
      if (internalEndDate && newStartDate > internalEndDate) {
        setInternalEndDate(null)
        onChange?.(newStartDate, null)
      } else {
        onChange?.(newStartDate, internalEndDate)
      }
    },
    [internalEndDate, onChange]
  )

  const handleEndDateChange = useCallback(
    (date: Date) => {
      const newEndDate = date
      if (internalStartDate && newEndDate < internalStartDate) {
        return
      }
      setInternalEndDate(newEndDate)
      onChange?.(internalStartDate, newEndDate)
    },
    [internalStartDate, onChange]
  )

  const clearDates = useCallback(() => {
    setInternalStartDate(null)
    setInternalEndDate(null)
    onChange?.(null, null)
  }, [onChange])

  return (
    <div className={cn('space-y-4', className)}>
      <div className="flex items-center gap-4">
        <div className="flex-1">
          <label className="block text-sm font-medium text-gray-700 mb-2">Start Date</label>
          <Calendar
            value={internalStartDate || undefined}
            onChange={handleStartDateChange}
            minDate={minDate}
            maxDate={internalEndDate || maxDate}
          />
        </div>
        <div className="flex-1">
          <label className="block text-sm font-medium text-gray-700 mb-2">End Date</label>
          <Calendar
            value={internalEndDate || undefined}
            onChange={handleEndDateChange}
            minDate={internalStartDate || minDate}
            maxDate={maxDate}
          />
        </div>
      </div>
      {(internalStartDate || internalEndDate) && (
        <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
          <div className="text-sm">
            <span className="font-medium">Range: </span>
            <span className="text-gray-600">
              {internalStartDate ? formatDate(internalStartDate.toISOString()) : 'Not set'} -{' '}
              {internalEndDate ? formatDate(internalEndDate.toISOString()) : 'Not set'}
            </span>
          </div>
          <button
            onClick={clearDates}
            className="text-sm text-primary-600 hover:text-primary-700"
          >
            Clear
          </button>
        </div>
      )}
    </div>
  )
}

export default DateRangePicker

