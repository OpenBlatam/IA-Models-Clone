'use client'

import { useCallback } from 'react'
import { Search, X } from 'lucide-react'
import Input from './Input'
import Button from './Button'

interface SearchInputProps {
  value: string
  onChange: (value: string) => void
  placeholder?: string
  onClear?: () => void
  className?: string
}

const SearchInput = ({ value, onChange, placeholder = 'Search...', onClear, className }: SearchInputProps) => {
  const handleClear = useCallback(() => {
    onChange('')
    onClear?.()
  }, [onChange, onClear])

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Escape') {
        handleClear()
      }
    },
    [handleClear]
  )

  return (
    <div className={className}>
      <div className="relative">
        <Input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          leftIcon={<Search className="w-4 h-4" />}
          className="pr-10"
        />
        {value && (
          <div className="absolute right-3 top-1/2 -translate-y-1/2">
            <Button
              variant="secondary"
              size="sm"
              onClick={handleClear}
              aria-label="Clear search"
              className="!p-1 !min-w-0"
            >
              <X className="w-4 h-4" />
            </Button>
          </div>
        )}
      </div>
    </div>
  )
}

export default SearchInput

