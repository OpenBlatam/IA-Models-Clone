'use client'

import { useState, useCallback, KeyboardEvent } from 'react'
import { X } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Input, Chip } from '@/components/ui'

interface TagInputProps {
  tags: string[]
  onChange: (tags: string[]) => void
  placeholder?: string
  className?: string
  maxTags?: number
  separator?: string
}

const TagInput = ({
  tags,
  onChange,
  placeholder = 'Add tags...',
  className,
  maxTags,
  separator = ',',
}: TagInputProps) => {
  const [inputValue, setInputValue] = useState('')

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter' || e.key === separator) {
        e.preventDefault()
        const newTag = inputValue.trim()
        if (newTag && !tags.includes(newTag)) {
          if (maxTags && tags.length >= maxTags) {
            return
          }
          onChange([...tags, newTag])
          setInputValue('')
        }
      } else if (e.key === 'Backspace' && inputValue === '' && tags.length > 0) {
        onChange(tags.slice(0, -1))
      }
    },
    [inputValue, tags, onChange, maxTags, separator]
  )

  const handleRemove = useCallback(
    (tagToRemove: string) => {
      onChange(tags.filter((tag) => tag !== tagToRemove))
    },
    [tags, onChange]
  )

  return (
    <div className={cn('space-y-2', className)}>
      <div className="flex flex-wrap gap-2 p-2 border border-gray-300 rounded-md min-h-[42px] focus-within:ring-2 focus-within:ring-primary-500 focus-within:border-primary-500">
        {tags.map((tag) => (
          <Chip
            key={tag}
            label={tag}
            onRemove={() => handleRemove(tag)}
            variant="default"
            size="sm"
          />
        ))}
        <Input
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={tags.length === 0 ? placeholder : ''}
          className="flex-1 min-w-[120px] border-0 focus:ring-0 p-0"
        />
      </div>
      {maxTags && (
        <p className="text-xs text-gray-500">
          {tags.length} / {maxTags} tags
        </p>
      )}
    </div>
  )
}

export default TagInput

