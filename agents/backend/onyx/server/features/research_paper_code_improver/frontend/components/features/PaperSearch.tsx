'use client'

import React, { useState, useMemo } from 'react'
import { Search, Filter, X } from 'lucide-react'
import { Input, Button, Card, Badge } from '../ui'
import { useDebounce } from '@/hooks'
import type { Paper } from '@/lib/api/types'

interface PaperSearchProps {
  papers: Paper[]
  onSearch?: (filteredPapers: Paper[]) => void
  onFilterChange?: (filters: {
    sources: string[]
    authors: string[]
    minSections?: number
  }) => void
}

const PaperSearch: React.FC<PaperSearchProps> = ({
  papers,
  onSearch,
  onFilterChange,
}) => {
  const [searchQuery, setSearchQuery] = useState('')
  const [showFilters, setShowFilters] = useState(false)
  const [selectedSources, setSelectedSources] = useState<string[]>([])
  const [selectedAuthors, setSelectedAuthors] = useState<string[]>([])
  const [minSections, setMinSections] = useState<number | undefined>()

  const debouncedSearch = useDebounce(searchQuery, 300)

  const uniqueSources = useMemo(() => {
    const sources = new Set(papers.map((p) => p.source))
    return Array.from(sources)
  }, [papers])

  const uniqueAuthors = useMemo(() => {
    const authors = new Set(
      papers.flatMap((p) => p.authors || []).filter(Boolean)
    )
    return Array.from(authors).slice(0, 20) // Limit to 20 most common
  }, [papers])

  const filteredPapers = useMemo(() => {
    let filtered = [...papers]

    // Text search
    if (debouncedSearch.trim()) {
      const query = debouncedSearch.toLowerCase()
      filtered = filtered.filter(
        (paper) =>
          paper.title?.toLowerCase().includes(query) ||
          paper.abstract?.toLowerCase().includes(query) ||
          paper.authors?.some((author) =>
            author.toLowerCase().includes(query)
          )
      )
    }

    // Source filter
    if (selectedSources.length > 0) {
      filtered = filtered.filter((paper) =>
        selectedSources.includes(paper.source)
      )
    }

    // Author filter
    if (selectedAuthors.length > 0) {
      filtered = filtered.filter((paper) =>
        paper.authors?.some((author) => selectedAuthors.includes(author))
      )
    }

    // Min sections filter
    if (minSections !== undefined) {
      filtered = filtered.filter(
        (paper) => paper.sections_count >= minSections
      )
    }

    return filtered
  }, [papers, debouncedSearch, selectedSources, selectedAuthors, minSections])

  React.useEffect(() => {
    onSearch?.(filteredPapers)
    onFilterChange?.({
      sources: selectedSources,
      authors: selectedAuthors,
      minSections,
    })
  }, [filteredPapers, selectedSources, selectedAuthors, minSections, onSearch, onFilterChange])

  const handleSourceToggle = (source: string) => {
    setSelectedSources((prev) =>
      prev.includes(source)
        ? prev.filter((s) => s !== source)
        : [...prev, source]
    )
  }

  const handleAuthorToggle = (author: string) => {
    setSelectedAuthors((prev) =>
      prev.includes(author)
        ? prev.filter((a) => a !== author)
        : [...prev, author]
    )
  }

  const clearFilters = () => {
    setSelectedSources([])
    setSelectedAuthors([])
    setMinSections(undefined)
    setSearchQuery('')
  }

  const hasActiveFilters =
    selectedSources.length > 0 ||
    selectedAuthors.length > 0 ||
    minSections !== undefined ||
    searchQuery.trim() !== ''

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <Input
          placeholder="Search papers by title, author, or abstract..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          leftIcon={<Search className="w-4 h-4" />}
          className="flex-1"
        />
        <Button
          variant="outline"
          onClick={() => setShowFilters(!showFilters)}
        >
          <Filter className="w-4 h-4 mr-2" />
          Filters
        </Button>
        {hasActiveFilters && (
          <Button variant="ghost" onClick={clearFilters}>
            <X className="w-4 h-4 mr-2" />
            Clear
          </Button>
        )}
      </div>

      {showFilters && (
        <Card>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-gray-900">Advanced Filters</h3>
              <Button variant="ghost" size="sm" onClick={clearFilters}>
                Clear All
              </Button>
            </div>

            {uniqueSources.length > 0 && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Source
                </label>
                <div className="flex flex-wrap gap-2">
                  {uniqueSources.map((source) => (
                    <Badge
                      key={source}
                      variant={
                        selectedSources.includes(source) ? 'primary' : 'default'
                      }
                      className="cursor-pointer"
                      onClick={() => handleSourceToggle(source)}
                    >
                      {source}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {uniqueAuthors.length > 0 && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Authors
                </label>
                <div className="max-h-32 overflow-y-auto flex flex-wrap gap-2">
                  {uniqueAuthors.map((author) => (
                    <Badge
                      key={author}
                      variant={
                        selectedAuthors.includes(author) ? 'primary' : 'default'
                      }
                      className="cursor-pointer"
                      onClick={() => handleAuthorToggle(author)}
                    >
                      {author}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Minimum Sections
              </label>
              <input
                type="number"
                min="0"
                value={minSections || ''}
                onChange={(e) =>
                  setMinSections(
                    e.target.value ? parseInt(e.target.value) : undefined
                  )
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                placeholder="Any"
              />
            </div>
          </div>
        </Card>
      )}

      {hasActiveFilters && (
        <div className="flex items-center gap-2 text-sm text-gray-600">
          <span>Showing {filteredPapers.length} of {papers.length} papers</span>
        </div>
      )}
    </div>
  )
}

export default PaperSearch



