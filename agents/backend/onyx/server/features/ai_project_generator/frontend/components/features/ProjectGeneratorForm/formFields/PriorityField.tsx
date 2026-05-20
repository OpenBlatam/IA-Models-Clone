'use client'

import { useCallback } from 'react'
import { Input, HelpTooltip } from '@/components/ui'

interface PriorityFieldProps {
  value: number
  error?: string
  onChange: (value: number) => void
}

const PriorityField = ({ value, error, onChange }: PriorityFieldProps) => {
  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      onChange(parseInt(e.target.value, 10) || 0)
    },
    [onChange]
  )

  return (
    <div className="relative">
      <div className="flex items-center gap-2 mb-2">
        <label htmlFor="priority" className="block text-sm font-medium text-gray-700">
          Priority (-10 to 10)
        </label>
        <HelpTooltip content="Higher priority projects are processed first. Use positive numbers for urgent projects." />
      </div>
      <Input
        id="priority"
        type="number"
        value={value}
        onChange={handleChange}
        min={-10}
        max={10}
        error={error}
        helperText="Higher priority = processed first"
        className="w-full"
      />
    </div>
  )
}

export default PriorityField

