/**
 * Toolbar Component
 * Complete toolbar with search, filters, and view controls
 */

'use client'

import React, { memo } from 'react'
import { Search, Filter, Eye, Settings, Download, Upload, Share2, Bell, Zap } from 'lucide-react'
import { motion } from 'framer-motion'
import SearchBar from '../../SearchBar'

interface ToolbarProps {
  searchQuery: string
  onSearchChange: (query: string) => void
  showFilters: boolean
  onToggleFilters: () => void
  viewMode: 'normal' | 'compact' | 'comfortable'
  onViewModeChange: (mode: 'normal' | 'compact' | 'comfortable') => void
  onExport?: () => void
  onImport?: () => void
  onShare?: () => void
  onSettings?: () => void
  onNotifications?: () => void
  showSearch?: boolean
  showFiltersButton?: boolean
  showViewControls?: boolean
  showActions?: boolean
}

export const Toolbar = memo(function Toolbar({
  searchQuery,
  onSearchChange,
  showFilters,
  onToggleFilters,
  viewMode,
  onViewModeChange,
  onExport,
  onImport,
  onShare,
  onSettings,
  onNotifications,
  showSearch = true,
  showFiltersButton = true,
  showViewControls = true,
  showActions = true,
}: ToolbarProps) {
  return (
    <div className="toolbar">
      <div className="toolbar__left">
        {showSearch && (
          <div className="toolbar__search">
            <SearchBar
              value={searchQuery}
              onChange={onSearchChange}
              placeholder="Buscar mensajes..."
            />
          </div>
        )}

        {showFiltersButton && (
          <button
            type="button"
            onClick={onToggleFilters}
            className={`toolbar__button ${showFilters ? 'toolbar__button--active' : ''}`}
            title="Filtros"
          >
            <Filter size={18} />
            <span>Filtros</span>
          </button>
        )}

        {showViewControls && (
          <div className="toolbar__view-controls">
            <button
              type="button"
              onClick={() => onViewModeChange('compact')}
              className={`toolbar__button ${viewMode === 'compact' ? 'toolbar__button--active' : ''}`}
              title="Vista compacta"
            >
              <Eye size={18} />
            </button>
            <button
              type="button"
              onClick={() => onViewModeChange('normal')}
              className={`toolbar__button ${viewMode === 'normal' ? 'toolbar__button--active' : ''}`}
              title="Vista normal"
            >
              <Eye size={18} />
            </button>
            <button
              type="button"
              onClick={() => onViewModeChange('comfortable')}
              className={`toolbar__button ${viewMode === 'comfortable' ? 'toolbar__button--active' : ''}`}
              title="Vista cómoda"
            >
              <Eye size={18} />
            </button>
          </div>
        )}
      </div>

      {showActions && (
        <div className="toolbar__right">
          {onExport && (
            <button
              type="button"
              onClick={onExport}
              className="toolbar__button"
              title="Exportar"
            >
              <Download size={18} />
            </button>
          )}

          {onImport && (
            <button
              type="button"
              onClick={onImport}
              className="toolbar__button"
              title="Importar"
            >
              <Upload size={18} />
            </button>
          )}

          {onShare && (
            <button
              type="button"
              onClick={onShare}
              className="toolbar__button"
              title="Compartir"
            >
              <Share2 size={18} />
            </button>
          )}

          {onNotifications && (
            <button
              type="button"
              onClick={onNotifications}
              className="toolbar__button"
              title="Notificaciones"
            >
              <Bell size={18} />
            </button>
          )}

          {onSettings && (
            <button
              type="button"
              onClick={onSettings}
              className="toolbar__button"
              title="Configuración"
            >
              <Settings size={18} />
            </button>
          )}
        </div>
      )}
    </div>
  )
})

export default Toolbar




