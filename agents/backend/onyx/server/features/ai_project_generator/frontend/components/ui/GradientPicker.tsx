'use client'

import { useState, useCallback } from 'react'
import { cn } from '@/lib/utils'

interface GradientStop {
  color: string
  position: number
}

interface GradientPickerProps {
  value?: string
  onChange?: (gradient: string) => void
  className?: string
  direction?: 'linear' | 'radial'
}

const GradientPicker = ({
  value,
  onChange,
  className,
  direction = 'linear',
}: GradientPickerProps) => {
  const [stops, setStops] = useState<GradientStop[]>([
    { color: '#000000', position: 0 },
    { color: '#FFFFFF', position: 100 },
  ])

  const handleStopChange = useCallback(
    (index: number, updates: Partial<GradientStop>) => {
      const newStops = [...stops]
      newStops[index] = { ...newStops[index], ...updates }
      setStops(newStops)

      const gradient = direction === 'linear'
        ? `linear-gradient(to right, ${newStops.map((s) => `${s.color} ${s.position}%`).join(', ')})`
        : `radial-gradient(circle, ${newStops.map((s) => `${s.color} ${s.position}%`).join(', ')})`

      onChange?.(gradient)
    },
    [stops, direction, onChange]
  )

  const addStop = useCallback(() => {
    const newStop: GradientStop = {
      color: '#808080',
      position: 50,
    }
    setStops([...stops, newStop].sort((a, b) => a.position - b.position))
  }, [stops])

  const removeStop = useCallback(
    (index: number) => {
      if (stops.length > 2) {
        setStops(stops.filter((_, i) => i !== index))
      }
    },
    [stops]
  )

  const gradient = direction === 'linear'
    ? `linear-gradient(to right, ${stops.map((s) => `${s.color} ${s.position}%`).join(', ')})`
    : `radial-gradient(circle, ${stops.map((s) => `${s.color} ${s.position}%`).join(', ')})`

  return (
    <div className={cn('space-y-4', className)}>
      <div
        className="w-full h-32 rounded-lg border border-gray-300"
        style={{ background: gradient }}
      />
      <div className="space-y-2">
        {stops.map((stop, index) => (
          <div key={index} className="flex items-center gap-3">
            <input
              type="color"
              value={stop.color}
              onChange={(e) => handleStopChange(index, { color: e.target.value })}
              className="w-12 h-12 rounded border border-gray-300 cursor-pointer"
            />
            <input
              type="range"
              min="0"
              max="100"
              value={stop.position}
              onChange={(e) => handleStopChange(index, { position: Number(e.target.value) })}
              className="flex-1"
            />
            <input
              type="number"
              min="0"
              max="100"
              value={stop.position}
              onChange={(e) => handleStopChange(index, { position: Number(e.target.value) })}
              className="w-16 px-2 py-1 border border-gray-300 rounded text-sm"
            />
            {stops.length > 2 && (
              <button
                onClick={() => removeStop(index)}
                className="px-2 py-1 text-sm text-red-600 hover:bg-red-50 rounded"
              >
                Remove
              </button>
            )}
          </div>
        ))}
        <button
          onClick={addStop}
          className="w-full px-4 py-2 text-sm text-primary-600 hover:bg-primary-50 rounded border border-primary-200"
        >
          Add Color Stop
        </button>
      </div>
    </div>
  )
}

export default GradientPicker

