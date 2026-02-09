'use client'

import { useState, useCallback } from 'react'
import { Plus, X } from 'lucide-react'
import { Input, Button, Badge } from '@/components/ui'

interface TagsFieldProps {
  tags: string[]
  onChange: (tags: string[]) => void
}

const TagsField = ({ tags, onChange }: TagsFieldProps) => {
  const [tagInput, setTagInput] = useState('')

  const handleAddTag = useCallback(() => {
    const trimmedTag = tagInput.trim()
    if (trimmedTag && !tags.includes(trimmedTag)) {
      onChange([...tags, trimmedTag])
      setTagInput('')
    }
  }, [tagInput, tags, onChange])

  const handleRemoveTag = useCallback(
    (tagToRemove: string) => {
      onChange(tags.filter((tag) => tag !== tagToRemove))
    },
    [tags, onChange]
  )

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') {
        e.preventDefault()
        handleAddTag()
      }
    },
    [handleAddTag]
  )

  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-2">Tags</label>
      <div className="flex gap-2 mb-2">
        <Input
          value={tagInput}
          onChange={(e) => setTagInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Add a tag..."
          className="flex-1"
        />
        <Button
          type="button"
          variant="secondary"
          size="md"
          leftIcon={<Plus className="w-4 h-4" />}
          onClick={handleAddTag}
          disabled={!tagInput.trim()}
        >
          Add
        </Button>
      </div>
      {tags.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {tags.map((tag) => (
            <Badge key={tag} variant="info" size="md" className="flex items-center gap-1">
              {tag}
              <button
                onClick={() => handleRemoveTag(tag)}
                className="ml-1 hover:text-red-600 focus:outline-none"
                tabIndex={0}
                aria-label={`Remove tag ${tag}`}
              >
                <X className="w-3 h-3" />
              </button>
            </Badge>
          ))}
        </div>
      )}
    </div>
  )
}

export default TagsField

