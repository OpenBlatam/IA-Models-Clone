/**
 * Custom hook for search and filtering functionality
 * Handles search queries, filters, and message filtering
 */

import { useState, useCallback, useMemo, useEffect } from 'react'

export interface SearchFilters {
  dateRange?: { start: Date; end: Date }
  minWords?: number
  maxWords?: number
  hasCode?: boolean
  hasLinks?: boolean
  role?: 'all' | 'user' | 'assistant'
}

export interface SearchState {
  searchQuery: string
  currentSearchIndex: number
  filteredMessages: any[]
  searchFilters: SearchFilters
  advancedSearch: boolean
  highlightSearch: boolean
  searchIndex: Map<string, number[]>
}

export interface SearchActions {
  setSearchQuery: (query: string) => void
  setCurrentSearchIndex: (index: number) => void
  setSearchFilters: (filters: SearchFilters) => void
  setAdvancedSearch: (enabled: boolean) => void
  setHighlightSearch: (enabled: boolean) => void
  nextMatch: () => void
  previousMatch: () => void
  clearSearch: () => void
  updateFilter: <K extends keyof SearchFilters>(key: K, value: SearchFilters[K]) => void
}

export function useSearchAndFilters(
  messages: any[] = []
): SearchState & SearchActions {
  const [searchQuery, setSearchQuery] = useState('')
  const [currentSearchIndex, setCurrentSearchIndex] = useState(-1)
  const [searchFilters, setSearchFilters] = useState<SearchFilters>({ role: 'all' })
  const [advancedSearch, setAdvancedSearch] = useState(false)
  const [highlightSearch, setHighlightSearch] = useState(true)
  const [searchIndex, setSearchIndex] = useState<Map<string, number[]>>(new Map())

  // Filter messages based on search query and filters
  const filteredMessages = useMemo(() => {
    if (!Array.isArray(messages)) {
      return []
    }

    let filtered = messages

    // Apply role filter
    if (searchFilters.role && searchFilters.role !== 'all') {
      filtered = filtered.filter(msg => msg.role === searchFilters.role)
    }

    // Apply search query
    if (searchQuery && searchQuery.trim()) {
      const query = searchQuery.toLowerCase().trim()
      filtered = filtered.filter(msg => {
        const content = typeof msg.content === 'string' ? msg.content.toLowerCase() : ''
        return content.includes(query)
      })
    }

    // Apply advanced filters
    if (advancedSearch) {
      if (searchFilters.minWords !== undefined) {
        filtered = filtered.filter(msg => {
          const wordCount = typeof msg.content === 'string' ? msg.content.split(/\s+/).length : 0
          return wordCount >= searchFilters.minWords!
        })
      }

      if (searchFilters.maxWords !== undefined) {
        filtered = filtered.filter(msg => {
          const wordCount = typeof msg.content === 'string' ? msg.content.split(/\s+/).length : 0
          return wordCount <= searchFilters.maxWords!
        })
      }

      if (searchFilters.hasCode !== undefined) {
        filtered = filtered.filter(msg => {
          const content = typeof msg.content === 'string' ? msg.content : ''
          const hasCode = content.includes('```') || content.includes('<code>')
          return searchFilters.hasCode ? hasCode : !hasCode
        })
      }

      if (searchFilters.hasLinks !== undefined) {
        filtered = filtered.filter(msg => {
          const content = typeof msg.content === 'string' ? msg.content : ''
          const hasLinks = /https?:\/\//.test(content)
          return searchFilters.hasLinks ? hasLinks : !hasLinks
        })
      }

      if (searchFilters.dateRange) {
        filtered = filtered.filter(msg => {
          if (!msg.timestamp) return false
          const msgDate = new Date(msg.timestamp)
          return msgDate >= searchFilters.dateRange!.start && msgDate <= searchFilters.dateRange!.end
        })
      }
    }

    return filtered
  }, [messages, searchQuery, searchFilters, advancedSearch])

  // Build search index
  useEffect(() => {
    if (!searchQuery || !Array.isArray(messages)) {
      setSearchIndex(new Map())
      return
    }

    const query = searchQuery.toLowerCase().trim()
    if (query.length === 0) {
      setSearchIndex(new Map())
      return
    }

    const index = new Map<string, number[]>()
    messages.forEach((msg, idx) => {
      const content = typeof msg.content === 'string' ? msg.content.toLowerCase() : ''
      if (content.includes(query)) {
        const positions: number[] = []
        let startIndex = 0
        while ((startIndex = content.indexOf(query, startIndex)) !== -1) {
          positions.push(startIndex)
          startIndex += query.length
        }
        index.set(msg.id || idx.toString(), positions)
      }
    })

    setSearchIndex(index)
    setCurrentSearchIndex(-1)
  }, [searchQuery, messages])

  // Navigation functions
  const nextMatch = useCallback(() => {
    if (filteredMessages.length === 0) return
    
    setCurrentSearchIndex(prev => {
      const next = prev + 1
      return next >= filteredMessages.length ? 0 : next
    })
  }, [filteredMessages.length])

  const previousMatch = useCallback(() => {
    if (filteredMessages.length === 0) return
    
    setCurrentSearchIndex(prev => {
      const next = prev - 1
      return next < 0 ? filteredMessages.length - 1 : next
    })
  }, [filteredMessages.length])

  const clearSearch = useCallback(() => {
    setSearchQuery('')
    setCurrentSearchIndex(-1)
    setSearchFilters({ role: 'all' })
    setAdvancedSearch(false)
  }, [])

  const updateFilter = useCallback(<K extends keyof SearchFilters>(
    key: K,
    value: SearchFilters[K]
  ) => {
    setSearchFilters(prev => ({
      ...prev,
      [key]: value,
    }))
  }, [])

  return {
    // State
    searchQuery,
    currentSearchIndex,
    filteredMessages,
    searchFilters,
    advancedSearch,
    highlightSearch,
    searchIndex,
    // Actions
    setSearchQuery,
    setCurrentSearchIndex,
    setSearchFilters,
    setAdvancedSearch,
    setHighlightSearch,
    nextMatch,
    previousMatch,
    clearSearch,
    updateFilter,
  }
}




