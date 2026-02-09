'use client'

import React, { useState } from 'react'
import { Filter, X } from 'lucide-react'
import { Card, Button, Checkbox, Select } from '../ui'

interface FilterOption {
  value: string
  label: string
}

interface FilterPanelProps {
  sources?: FilterOption[]
  onFilterChange?: (filters: {
    sources: string[]
    sortBy: string
  }) => void
}

const FilterPanel: React.FC<FilterPanelProps> = ({
  sources = [],
  onFilterChange,
}) => {
  const [selectedSources, setSelectedSources] = useState<string[]>([])
  const [sortBy, setSortBy] = useState('recent')

  const handleSourceToggle = (value: string) => {
    const newSources = selectedSources.includes(value)
      ? selectedSources.filter((s) => s !== value)
      : [...selectedSources, value]
    setSelectedSources(newSources)
    onFilterChange?.({ sources: newSources, sortBy })
  }

  const handleSortChange = (value: string) => {
    setSortBy(value)
    onFilterChange?.({ sources: selectedSources, sortBy: value })
  }

  const handleClearFilters = () => {
    setSelectedSources([])
    setSortBy('recent')
    onFilterChange?.({ sources: [], sortBy: 'recent' })
  }

  const hasActiveFilters = selectedSources.length > 0 || sortBy !== 'recent'

  return (
    <Card>
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Filter className="w-5 h-5 text-gray-600" />
            <h3 className="font-semibold text-gray-900">Filters</h3>
          </div>
          {hasActiveFilters && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClearFilters}
            >
              <X className="w-4 h-4 mr-1" />
              Clear
            </Button>
          )}
        </div>

        {sources.length > 0 && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Source
            </label>
            <div className="space-y-2">
              {sources.map((source) => (
                <Checkbox
                  key={source.value}
                  label={source.label}
                  checked={selectedSources.includes(source.value)}
                  onChange={() => handleSourceToggle(source.value)}
                />
              ))}
            </div>
          </div>
        )}

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Sort By
          </label>
          <Select
            value={sortBy}
            onChange={(e) => handleSortChange(e.target.value)}
            options={[
              { value: 'recent', label: 'Most Recent' },
              { value: 'oldest', label: 'Oldest First' },
              { value: 'title', label: 'Title (A-Z)' },
              { value: 'size', label: 'Size (Largest)' },
            ]}
          />
        </div>
      </div>
    </Card>
  )
}

export default FilterPanel




