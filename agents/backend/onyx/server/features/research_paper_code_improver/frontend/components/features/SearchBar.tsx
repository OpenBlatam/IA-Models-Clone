'use client'

import React, { useState, useCallback } from 'react'
import { Search, X } from 'lucide-react'
import { Input } from '../ui'
import { useDebounce } from '@/hooks'

interface SearchBarProps {
  onSearch: (query: string) => void
  placeholder?: string
  debounceMs?: number
}

const SearchBar: React.FC<SearchBarProps> = ({
  onSearch,
  placeholder = 'Search...',
  debounceMs = 300,
}) => {
  const [searchQuery, setSearchQuery] = useState('')
  const debouncedSearch = useDebounce(searchQuery, debounceMs)

  React.useEffect(() => {
    onSearch(debouncedSearch)
  }, [debouncedSearch, onSearch])

  const handleClear = useCallback(() => {
    setSearchQuery('')
    onSearch('')
  }, [onSearch])

  return (
    <div className="relative">
      <Input
        type="text"
        placeholder={placeholder}
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        leftIcon={<Search className="w-4 h-4" />}
        rightIcon={
          searchQuery ? (
            <button
              onClick={handleClear}
              className="hover:bg-gray-100 rounded p-1 transition-colors"
              aria-label="Clear search"
            >
              <X className="w-4 h-4" />
            </button>
          ) : undefined
        }
      />
    </div>
  )
}

export default SearchBar

