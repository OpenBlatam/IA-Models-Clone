'use client'

import { useState, useCallback } from 'react'
import { cn } from '@/lib/utils'

interface ColorPickerProps {
  value?: string
  onChange?: (color: string) => void
  className?: string
  showPresets?: boolean
  presets?: string[]
}

const ColorPicker = ({
  value = '#000000',
  onChange,
  className,
  showPresets = true,
  presets = [
    '#000000',
    '#FFFFFF',
    '#FF0000',
    '#00FF00',
    '#0000FF',
    '#FFFF00',
    '#FF00FF',
    '#00FFFF',
    '#FFA500',
    '#800080',
    '#FFC0CB',
    '#A52A2A',
  ],
}: ColorPickerProps) => {
  const [color, setColor] = useState(value)

  const handleChange = useCallback(
    (newColor: string) => {
      setColor(newColor)
      onChange?.(newColor)
    },
    [onChange]
  )

  return (
    <div className={cn('space-y-3', className)}>
      <div className="flex items-center gap-3">
        <input
          type="color"
          value={color}
          onChange={(e) => handleChange(e.target.value)}
          className="w-16 h-16 rounded border border-gray-300 cursor-pointer"
          aria-label="Color picker"
        />
        <div className="flex-1">
          <input
            type="text"
            value={color}
            onChange={(e) => {
              const newColor = e.target.value
              if (/^#[0-9A-Fa-f]{0,6}$/.test(newColor)) {
                handleChange(newColor)
              }
            }}
            className="w-full px-3 py-2 border border-gray-300 rounded-md font-mono text-sm"
            placeholder="#000000"
            maxLength={7}
          />
        </div>
      </div>
      {showPresets && (
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Presets</label>
          <div className="grid grid-cols-6 gap-2">
            {presets.map((preset) => (
              <button
                key={preset}
                onClick={() => handleChange(preset)}
                className={cn(
                  'w-full h-10 rounded border-2 transition-all',
                  color === preset ? 'border-primary-600 scale-110' : 'border-gray-300 hover:border-gray-400'
                )}
                style={{ backgroundColor: preset }}
                aria-label={`Select color ${preset}`}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default ColorPicker

