'use client'

import React, { useState, KeyboardEvent } from 'react'
import { X } from 'lucide-react'
import { Input, Badge } from '../ui'
import { clsx } from 'clsx'

interface TagInputProps {
  label?: string
  tags: string[]
  onChange: (tags: string[]) => void
  placeholder?: string
  maxTags?: number
  className?: string
}

const TagInput: React.FC<TagInputProps> = ({
  label,
  tags,
  onChange,
  placeholder = 'Add tags...',
  maxTags,
  className,
}) => {
  const [inputValue, setInputValue] = useState('')

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && inputValue.trim()) {
      e.preventDefault()
      addTag(inputValue.trim())
    } else if (e.key === 'Backspace' && !inputValue && tags.length > 0) {
      removeTag(tags.length - 1)
    }
  }

  const addTag = (tag: string) => {
    if (maxTags && tags.length >= maxTags) {
      return
    }
    if (tag && !tags.includes(tag)) {
      onChange([...tags, tag])
      setInputValue('')
    }
  }

  const removeTag = (index: number) => {
    onChange(tags.filter((_, i) => i !== index))
  }

  return (
    <div className={clsx('w-full', className)}>
      {label && (
        <label className="block text-sm font-medium text-gray-700 mb-1">
          {label}
        </label>
      )}
      <div className="flex flex-wrap gap-2 p-2 border border-gray-300 rounded-lg focus-within:border-primary-500 focus-within:ring-2 focus-within:ring-primary-500">
        {tags.map((tag, index) => (
          <Badge
            key={index}
            variant="default"
            size="sm"
            className="flex items-center gap-1"
          >
            {tag}
            <button
              onClick={() => removeTag(index)}
              className="ml-1 hover:bg-gray-200 rounded-full p-0.5 transition-colors"
              aria-label={`Remove ${tag}`}
            >
              <X className="w-3 h-3" />
            </button>
          </Badge>
        ))}
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={tags.length === 0 ? placeholder : ''}
          className="flex-1 min-w-[120px] outline-none text-sm"
          disabled={maxTags ? tags.length >= maxTags : false}
        />
      </div>
      {maxTags && (
        <p className="mt-1 text-xs text-gray-500">
          {tags.length}/{maxTags} tags
        </p>
      )}
    </div>
  )
}

export default TagInput




