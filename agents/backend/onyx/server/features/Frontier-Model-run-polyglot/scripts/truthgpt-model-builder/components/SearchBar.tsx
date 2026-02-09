/**
 * Search Bar Component
 * Advanced search functionality for messages and models
 */

'use client'

import { useState, useMemo, useCallback, forwardRef, useImperativeHandle, useRef } from 'react'
import { Search, X, Filter } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

interface SearchBarProps {
  onSearch: (query: string) => void
  onFilter?: (filters: SearchFilters) => void
  placeholder?: string
  className?: string
}

export interface SearchBarRef {
  focus: () => void
  clear: () => void
  getValue: () => string
}

export interface SearchFilters {
  role?: 'user' | 'assistant' | 'system'
  dateRange?: {
    start: Date
    end: Date
  }
  hasAttachments?: boolean
}

const SearchBar = forwardRef<SearchBarRef, SearchBarProps>(({
  onSearch,
  onFilter,
  placeholder = 'Buscar mensajes...',
  className = ''
}, ref) => {
  const [query, setQuery] = useState('')
  const [showFilters, setShowFilters] = useState(false)
  const [filters, setFilters] = useState<SearchFilters>({})
  const inputRef = useRef<HTMLInputElement>(null)

  const debouncedSearch = useMemo(
    () => {
      let timeout: NodeJS.Timeout | null = null
      return (value: string) => {
        if (timeout) clearTimeout(timeout)
        timeout = setTimeout(() => {
          onSearch(value)
        }, 300)
      }
    },
    [onSearch]
  )

  const handleQueryChange = useCallback(
    (value: string) => {
      setQuery(value)
      debouncedSearch(value)
    },
    [debouncedSearch]
  )

  const handleFilterChange = useCallback(
    (newFilters: SearchFilters) => {
      const updatedFilters = { ...filters, ...newFilters }
      setFilters(updatedFilters)
      onFilter?.(updatedFilters)
    },
    [filters, onFilter]
  )

  const clearSearch = useCallback(() => {
    setQuery('')
    onSearch('')
  }, [onSearch])

  useImperativeHandle(ref, () => ({
    focus: () => {
      inputRef.current?.focus()
    },
    clear: () => {
      setQuery('')
      onSearch('')
      if (inputRef.current) {
        inputRef.current.value = ''
      }
    },
    getValue: () => query
  }), [query, onSearch])

  return (
    <div className={`relative ${className}`}>
      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={e => handleQueryChange(e.target.value)}
            placeholder={placeholder}
            className="w-full pl-10 pr-10 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
          />
          {query && (
            <button
              onClick={clearSearch}
              className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
        <button
          onClick={() => setShowFilters(!showFilters)}
          className={`p-2 rounded-lg transition-colors ${
            showFilters
              ? 'bg-purple-600 text-white'
              : 'bg-slate-700/50 text-gray-400 hover:text-white'
          }`}
          title="Filtros avanzados"
        >
          <Filter className="w-4 h-4" />
        </button>
      </div>

      <AnimatePresence>
        {showFilters && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-2 p-4 bg-slate-800/50 rounded-lg border border-slate-700 space-y-3"
          >
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Rol
              </label>
              <select
                value={filters.role || ''}
                onChange={e =>
                  handleFilterChange({
                    role: e.target.value as SearchFilters['role'] || undefined
                  })
                }
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white"
              >
                <option value="">Todos</option>
                <option value="user">Usuario</option>
                <option value="assistant">Asistente</option>
                <option value="system">Sistema</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Con adjuntos
              </label>
              <input
                type="checkbox"
                checked={filters.hasAttachments || false}
                onChange={e =>
                  handleFilterChange({
                    hasAttachments: e.target.checked || undefined
                  })
                }
                className="w-4 h-4 text-purple-600 bg-slate-700 border-slate-600 rounded focus:ring-purple-500"
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
})

SearchBar.displayName = 'SearchBar'

export default SearchBar
