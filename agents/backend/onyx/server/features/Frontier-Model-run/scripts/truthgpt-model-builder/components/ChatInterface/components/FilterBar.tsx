/**
 * FilterBar Component
 * Advanced filtering controls
 */

'use client'

import React, { memo } from 'react'
import { Filter, X, Calendar, Code, Link, User, Bot } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { SearchFilters } from '../types'

interface FilterBarProps {
  filters: SearchFilters
  onFiltersChange: (filters: SearchFilters) => void
  onClear: () => void
  isOpen: boolean
  onToggle: () => void
}

export const FilterBar = memo(function FilterBar({
  filters,
  onFiltersChange,
  onClear,
  isOpen,
  onToggle,
}: FilterBarProps) {
  const updateFilter = <K extends keyof SearchFilters>(key: K, value: SearchFilters[K]) => {
    onFiltersChange({
      ...filters,
      [key]: value,
    })
  }

  const hasActiveFilters = Object.values(filters).some(value => {
    if (value === undefined || value === null) return false
    if (typeof value === 'object' && 'start' in value) {
      return value.start || value.end
    }
    return value !== 'all' && value !== false
  })

  return (
    <>
      <button
        type="button"
        onClick={onToggle}
        className={`filter-bar__toggle ${hasActiveFilters ? 'filter-bar__toggle--active' : ''}`}
      >
        <Filter size={18} />
        <span>Filtros</span>
        {hasActiveFilters && <span className="filter-bar__badge" />}
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="filter-bar"
          >
            <div className="filter-bar__header">
              <h3>Filtros</h3>
              <button
                type="button"
                onClick={onClear}
                className="filter-bar__clear"
                disabled={!hasActiveFilters}
              >
                <X size={16} />
                Limpiar
              </button>
            </div>

            <div className="filter-bar__content">
              {/* Role Filter */}
              <div className="filter-bar__group">
                <label>Rol</label>
                <div className="filter-bar__buttons">
                  <button
                    type="button"
                    onClick={() => updateFilter('role', 'all')}
                    className={filters.role === 'all' ? 'active' : ''}
                  >
                    Todos
                  </button>
                  <button
                    type="button"
                    onClick={() => updateFilter('role', 'user')}
                    className={filters.role === 'user' ? 'active' : ''}
                  >
                    <User size={14} />
                    Usuario
                  </button>
                  <button
                    type="button"
                    onClick={() => updateFilter('role', 'assistant')}
                    className={filters.role === 'assistant' ? 'active' : ''}
                  >
                    <Bot size={14} />
                    Asistente
                  </button>
                </div>
              </div>

              {/* Word Count Filter */}
              <div className="filter-bar__group">
                <label>Palabras</label>
                <div className="filter-bar__inputs">
                  <input
                    type="number"
                    placeholder="Mín"
                    value={filters.minWords || ''}
                    onChange={(e) => updateFilter('minWords', e.target.value ? parseInt(e.target.value) : undefined)}
                  />
                  <span>-</span>
                  <input
                    type="number"
                    placeholder="Máx"
                    value={filters.maxWords || ''}
                    onChange={(e) => updateFilter('maxWords', e.target.value ? parseInt(e.target.value) : undefined)}
                  />
                </div>
              </div>

              {/* Content Filters */}
              <div className="filter-bar__group">
                <label>Contenido</label>
                <div className="filter-bar__checkboxes">
                  <label>
                    <input
                      type="checkbox"
                      checked={filters.hasCode || false}
                      onChange={(e) => updateFilter('hasCode', e.target.checked || undefined)}
                    />
                    <Code size={14} />
                    Con código
                  </label>
                  <label>
                    <input
                      type="checkbox"
                      checked={filters.hasLinks || false}
                      onChange={(e) => updateFilter('hasLinks', e.target.checked || undefined)}
                    />
                    <Link size={14} />
                    Con links
                  </label>
                </div>
              </div>

              {/* Date Range Filter */}
              <div className="filter-bar__group">
                <label>
                  <Calendar size={14} />
                  Rango de fechas
                </label>
                <div className="filter-bar__dates">
                  <input
                    type="date"
                    value={filters.dateRange?.start ? filters.dateRange.start.toISOString().split('T')[0] : ''}
                    onChange={(e) => {
                      const start = e.target.value ? new Date(e.target.value) : undefined
                      updateFilter('dateRange', {
                        start: start || filters.dateRange?.start || new Date(),
                        end: filters.dateRange?.end || new Date(),
                      })
                    }}
                  />
                  <span>a</span>
                  <input
                    type="date"
                    value={filters.dateRange?.end ? filters.dateRange.end.toISOString().split('T')[0] : ''}
                    onChange={(e) => {
                      const end = e.target.value ? new Date(e.target.value) : undefined
                      updateFilter('dateRange', {
                        start: filters.dateRange?.start || new Date(),
                        end: end || filters.dateRange?.end || new Date(),
                      })
                    }}
                  />
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
})

export default FilterBar




