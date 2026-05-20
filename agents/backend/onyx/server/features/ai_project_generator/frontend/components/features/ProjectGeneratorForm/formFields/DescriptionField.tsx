'use client'

import { useCallback } from 'react'
import type { ProjectRequest } from '@/types'
import { Textarea, HelpTooltip } from '@/components/ui'

interface DescriptionFieldProps {
  value: string
  error?: string
  onChange: (value: string) => void
}

const DescriptionField = ({ value, error, onChange }: DescriptionFieldProps) => {
  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      onChange(e.target.value)
    },
    [onChange]
  )

  return (
    <div className="relative">
      <div className="flex items-center gap-2 mb-2">
        <label htmlFor="description" className="block text-sm font-medium text-gray-700">
          Project Description *
        </label>
        <HelpTooltip content="Provide a detailed description of your AI project. Include the type of AI (chat, vision, audio, etc.) and key features you need." />
      </div>
      <Textarea
        id="description"
        value={value}
        onChange={handleChange}
        required
        minLength={10}
        maxLength={2000}
        rows={5}
        placeholder="Describe your AI project in detail..."
        error={error}
        showCharCount
        helperText="Include the type of AI (chat, vision, audio, etc.) and key features you need."
        className="w-full"
      />
    </div>
  )
}

export default DescriptionField

