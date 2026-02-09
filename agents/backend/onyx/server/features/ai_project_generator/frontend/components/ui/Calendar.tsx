'use client'

import { useState, useCallback } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui'

interface CalendarProps {
  value?: Date
  onChange?: (date: Date) => void
  minDate?: Date
  maxDate?: Date
  className?: string
  showTodayButton?: boolean
}

const Calendar = ({
  value,
  onChange,
  minDate,
  maxDate,
  className,
  showTodayButton = true,
}: CalendarProps) => {
  const [currentDate, setCurrentDate] = useState(value || new Date())
  const [selectedDate, setSelectedDate] = useState(value)

  const year = currentDate.getFullYear()
  const month = currentDate.getMonth()

  const firstDayOfMonth = new Date(year, month, 1)
  const lastDayOfMonth = new Date(year, month + 1, 0)
  const daysInMonth = lastDayOfMonth.getDate()
  const startingDayOfWeek = firstDayOfMonth.getDay()

  const monthNames = [
    'January',
    'February',
    'March',
    'April',
    'May',
    'June',
    'July',
    'August',
    'September',
    'October',
    'November',
    'December',
  ]

  const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

  const handleDateClick = useCallback(
    (day: number) => {
      const newDate = new Date(year, month, day)
      if (minDate && newDate < minDate) {
        return
      }
      if (maxDate && newDate > maxDate) {
        return
      }
      setSelectedDate(newDate)
      onChange?.(newDate)
    },
    [year, month, minDate, maxDate, onChange]
  )

  const handlePreviousMonth = useCallback(() => {
    setCurrentDate(new Date(year, month - 1, 1))
  }, [year, month])

  const handleNextMonth = useCallback(() => {
    setCurrentDate(new Date(year, month + 1, 1))
  }, [year, month])

  const handleToday = useCallback(() => {
    const today = new Date()
    setCurrentDate(today)
    setSelectedDate(today)
    onChange?.(today)
  }, [onChange])

  const isToday = (day: number) => {
    const today = new Date()
    return (
      day === today.getDate() &&
      month === today.getMonth() &&
      year === today.getFullYear()
    )
  }

  const isSelected = (day: number) => {
    if (!selectedDate) {
      return false
    }
    return (
      day === selectedDate.getDate() &&
      month === selectedDate.getMonth() &&
      year === selectedDate.getFullYear()
    )
  }

  const isDisabled = (day: number) => {
    const date = new Date(year, month, day)
    if (minDate && date < minDate) {
      return true
    }
    if (maxDate && date > maxDate) {
      return true
    }
    return false
  }

  const days = []
  for (let i = 0; i < startingDayOfWeek; i++) {
    days.push(null)
  }
  for (let i = 1; i <= daysInMonth; i++) {
    days.push(i)
  }

  return (
    <div className={cn('w-full max-w-sm', className)}>
      <div className="flex items-center justify-between mb-4">
        <Button
          variant="secondary"
          size="sm"
          onClick={handlePreviousMonth}
          leftIcon={<ChevronLeft className="w-4 h-4" />}
          aria-label="Previous month"
        />
        <div className="text-center">
          <div className="font-semibold text-gray-900">
            {monthNames[month]} {year}
          </div>
        </div>
        <Button
          variant="secondary"
          size="sm"
          onClick={handleNextMonth}
          rightIcon={<ChevronRight className="w-4 h-4" />}
          aria-label="Next month"
        />
      </div>

      <div className="grid grid-cols-7 gap-1 mb-2">
        {dayNames.map((day) => (
          <div key={day} className="text-center text-sm font-medium text-gray-500 py-2">
            {day}
          </div>
        ))}
      </div>

      <div className="grid grid-cols-7 gap-1">
        {days.map((day, index) => {
          if (day === null) {
            return <div key={index} />
          }

          return (
            <button
              key={day}
              onClick={() => handleDateClick(day)}
              disabled={isDisabled(day)}
              className={cn(
                'aspect-square rounded-md text-sm transition-colors',
                isToday(day) && 'ring-2 ring-primary-500',
                isSelected(day) && 'bg-primary-600 text-white',
                !isSelected(day) && !isToday(day) && 'hover:bg-gray-100',
                isDisabled(day) && 'opacity-50 cursor-not-allowed'
              )}
            >
              {day}
            </button>
          )
        })}
      </div>

      {showTodayButton && (
        <div className="mt-4">
          <Button variant="secondary" size="sm" onClick={handleToday} className="w-full">
            Today
          </Button>
        </div>
      )}
    </div>
  )
}

export default Calendar

