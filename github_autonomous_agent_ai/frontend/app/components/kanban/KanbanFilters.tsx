'use client';

import { TaskStatus } from '../../types/task';
import { KANBAN_COLUMNS } from '../../constants/task-constants';
import { cn } from '../../utils/cn';

interface KanbanFiltersProps {
  searchQuery: string;
  setSearchQuery: (query: string) => void;
  selectedRepository: string;
  setSelectedRepository: (repo: string) => void;
  repositories: string[];
  statusFilter: TaskStatus | 'all';
  setStatusFilter: (status: TaskStatus | 'all') => void;
  dateFilter: 'all' | 'today' | 'week' | 'month' | 'custom';
  setDateFilter: (filter: 'all' | 'today' | 'week' | 'month' | 'custom') => void;
  viewMode: 'kanban' | 'list' | 'timeline' | 'calendar';
  setViewMode: (mode: 'kanban' | 'list' | 'timeline' | 'calendar') => void;
  sortBy: 'date' | 'status' | 'repo' | 'model';
  setSortBy: (sort: 'date' | 'status' | 'repo' | 'model') => void;
  sortOrder: 'asc' | 'desc';
  setSortOrder: (order: 'asc' | 'desc') => void;
  cardSize: 'compact' | 'normal' | 'expanded';
  setCardSize: (size: 'compact' | 'normal' | 'expanded') => void;
  showRepoSummary: boolean;
  setShowRepoSummary: (show: boolean) => void;
  showRecentActivity: boolean;
  setShowRecentActivity: (show: boolean) => void;
  recentActivityCount: number;
  groupByRepo: boolean;
  setGroupByRepo: (group: boolean) => void;
  showStats: boolean;
  setShowStats: (show: boolean) => void;
  savedFilters: Array<{ name: string; filters: any }>;
  saveCurrentFilters: () => void;
  applySavedFilter: (filter: any) => void;
  deleteSavedFilter: (index: number) => void;
  handleExportTasks: () => void;
  onClearFilters: () => void;
  searchSuggestions?: string[];
  showSearchSuggestions?: boolean;
  setShowSearchSuggestions?: (show: boolean) => void;
}

export function KanbanFilters({
  searchQuery,
  setSearchQuery,
  selectedRepository,
  setSelectedRepository,
  repositories,
  statusFilter,
  setStatusFilter,
  dateFilter,
  setDateFilter,
  viewMode,
  setViewMode,
  sortBy,
  setSortBy,
  sortOrder,
  setSortOrder,
  cardSize,
  setCardSize,
  showRepoSummary,
  setShowRepoSummary,
  showRecentActivity,
  setShowRecentActivity,
  recentActivityCount,
  groupByRepo,
  setGroupByRepo,
  showStats,
  setShowStats,
  savedFilters,
  saveCurrentFilters,
  applySavedFilter,
  deleteSavedFilter,
  handleExportTasks,
  onClearFilters,
  searchSuggestions = [],
  showSearchSuggestions = false,
  setShowSearchSuggestions,
}: KanbanFiltersProps) {
  return (
    <div className="bg-white border-b border-gray-200 px-4 py-3 sticky top-[73px] z-40">
      <div className="flex items-center gap-3 flex-wrap">
        {/* Búsqueda */}
        <div className="flex-1 relative min-w-[200px]">
          <svg className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <div className="relative w-full">
              <input
                type="text"
                placeholder="Buscar tareas... (usa 'repo:', 'status:', 'error:' para búsqueda avanzada)"
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  setShowSearchSuggestions?.(true);
                }}
                onFocus={() => setShowSearchSuggestions?.(true)}
                onBlur={() => setTimeout(() => setShowSearchSuggestions?.(false), 200)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                aria-label="Buscar tareas"
                title="Búsqueda avanzada: usa 'repo:nombre', 'status:completed', 'error:texto', 'model:nombre' para filtrar"
              />
              {/* Sugerencias de búsqueda */}
              {showSearchSuggestions && searchSuggestions.length > 0 && (
                <div className="absolute z-50 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg max-h-60 overflow-y-auto">
                  {searchSuggestions.map((suggestion, idx) => (
                    <button
                      key={idx}
                      onClick={() => {
                        setSearchQuery(suggestion);
                        setShowSearchSuggestions?.(false);
                      }}
                      className="w-full text-left px-4 py-2 hover:bg-gray-100 transition-colors text-sm"
                    >
                      <span className="text-gray-600">{suggestion}</span>
                    </button>
                  ))}
                </div>
              )}
            {(searchQuery.startsWith('repo:') || searchQuery.startsWith('status:') || searchQuery.startsWith('error:') || searchQuery.startsWith('model:')) && (
              <div className="absolute right-12 top-1/2 transform -translate-y-1/2">
                <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded font-medium">
                  Avanzada
                </span>
              </div>
            )}
          </div>
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
              aria-label="Limpiar búsqueda"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>

        {/* Filtro por repositorio */}
        {repositories.length > 0 && (
          <select
            value={selectedRepository}
            onChange={(e) => setSelectedRepository(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
          >
            <option value="all">Todos los repositorios</option>
            {repositories.map((repo) => (
              <option key={repo} value={repo}>{repo}</option>
            ))}
          </select>
        )}

        {/* Filtro por estado */}
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value as TaskStatus | 'all')}
          className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
        >
          <option value="all">Todos los estados</option>
          {KANBAN_COLUMNS.map((col) => (
            <option key={col.id} value={col.id}>{col.label}</option>
          ))}
        </select>

        {/* Filtro por fecha */}
        <select
          value={dateFilter}
          onChange={(e) => setDateFilter(e.target.value as 'all' | 'today' | 'week' | 'month')}
          className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
        >
          <option value="all">Todas las fechas</option>
          <option value="today">Hoy</option>
          <option value="week">Esta semana</option>
          <option value="month">Este mes</option>
        </select>

        {/* Ordenamiento (solo en vista lista) */}
        {viewMode === 'list' && (
          <div className="flex items-center gap-2">
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as 'date' | 'status' | 'repo')}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
            >
              <option value="date">Ordenar por fecha</option>
              <option value="status">Ordenar por estado</option>
              <option value="repo">Ordenar por repositorio</option>
            </select>
            <button
              onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
              className="p-2 border border-gray-300 rounded-lg hover:bg-gray-100 transition-colors"
              title={sortOrder === 'asc' ? 'Orden ascendente' : 'Orden descendente'}
            >
              {sortOrder === 'asc' ? (
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                </svg>
              ) : (
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              )}
            </button>
          </div>
        )}

        {/* Toggle vista */}
        <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1 border border-gray-300">
          <button
            onClick={() => setViewMode('kanban')}
            className={cn(
              "px-3 py-1.5 rounded text-sm font-medium transition-colors",
              viewMode === 'kanban' 
                ? "bg-white text-gray-900 shadow-sm" 
                : "text-gray-600 hover:text-gray-900"
            )}
            title="Vista Kanban"
          >
            <svg className="w-4 h-4 inline mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
            </svg>
            Kanban
          </button>
          <button
            onClick={() => setViewMode('list')}
            className={cn(
              "px-3 py-1.5 rounded text-sm font-medium transition-colors",
              viewMode === 'list' 
                ? "bg-white text-gray-900 shadow-sm" 
                : "text-gray-600 hover:text-gray-900"
            )}
            title="Vista Lista"
          >
            <svg className="w-4 h-4 inline mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
            Lista
          </button>
          <button
            onClick={() => setViewMode('timeline')}
            className={cn(
              "px-3 py-1.5 rounded text-sm font-medium transition-colors",
              viewMode === 'timeline' 
                ? "bg-white text-gray-900 shadow-sm" 
                : "text-gray-600 hover:text-gray-900"
            )}
            title="Vista Timeline"
          >
            <svg className="w-4 h-4 inline mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Timeline
          </button>
        </div>

        {/* Resumen por repositorio */}
        <button
          onClick={() => setShowRepoSummary(!showRepoSummary)}
          className={cn(
            "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2",
            showRepoSummary 
              ? "bg-blue-100 text-blue-700 border border-blue-300" 
              : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
          )}
          title="Resumen por repositorio"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          Repos
        </button>

        {/* Actividad reciente */}
        <button
          onClick={() => setShowRecentActivity(!showRecentActivity)}
          className={cn(
            "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 relative",
            showRecentActivity 
              ? "bg-blue-100 text-blue-700 border border-blue-300" 
              : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
          )}
          title="Actividad reciente"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          Actividad
          {recentActivityCount > 0 && (
            <span className="absolute -top-1 -right-1 w-5 h-5 bg-blue-500 text-white text-xs rounded-full flex items-center justify-center">
              {recentActivityCount > 9 ? '9+' : recentActivityCount}
            </span>
          )}
        </button>

        {/* Tamaño de tarjeta (solo en vista kanban) */}
        {viewMode === 'kanban' && (
          <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1 border border-gray-300">
            <button
              onClick={() => setCardSize('compact')}
              className={cn(
                "px-2 py-1 rounded text-xs transition-colors",
                cardSize === 'compact' 
                  ? "bg-white text-gray-900 shadow-sm" 
                  : "text-gray-600 hover:text-gray-900"
              )}
              title="Vista compacta"
            >
              <svg className="w-3 h-3 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            <button
              onClick={() => setCardSize('normal')}
              className={cn(
                "px-2 py-1 rounded text-xs transition-colors",
                cardSize === 'normal' 
                  ? "bg-white text-gray-900 shadow-sm" 
                  : "text-gray-600 hover:text-gray-900"
              )}
              title="Vista normal"
            >
              <svg className="w-3 h-3 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM14 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1h-4a1 1 0 01-1-1V5zM4 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1H5a1 1 0 01-1-1v-3zM14 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1h-4a1 1 0 01-1-1v-3z" />
              </svg>
            </button>
            <button
              onClick={() => setCardSize('expanded')}
              className={cn(
                "px-2 py-1 rounded text-xs transition-colors",
                cardSize === 'expanded' 
                  ? "bg-white text-gray-900 shadow-sm" 
                  : "text-gray-600 hover:text-gray-900"
              )}
              title="Vista expandida"
            >
              <svg className="w-3 h-3 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
              </svg>
            </button>
          </div>
        )}

        {/* Filtros guardados */}
        {savedFilters.length > 0 && (
          <div className="relative group">
            <button
              className="px-3 py-2 rounded-lg text-sm font-medium bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300 transition-colors flex items-center gap-2"
              title="Filtros guardados"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
              </svg>
              Filtros ({savedFilters.length})
            </button>
            <div className="absolute right-0 top-full mt-2 w-64 bg-white border border-gray-200 rounded-lg shadow-lg z-50 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all">
              <div className="p-2 max-h-96 overflow-y-auto">
                {savedFilters.map((filter, index) => (
                  <div key={index} className="flex items-center justify-between p-2 hover:bg-gray-50 rounded">
                    <button
                      onClick={() => applySavedFilter(filter)}
                      className="flex-1 text-left text-sm text-gray-700 hover:text-blue-600"
                    >
                      {filter.name}
                    </button>
                    <button
                      onClick={() => deleteSavedFilter(index)}
                      className="p-1 text-red-600 hover:text-red-800 hover:bg-red-50 rounded"
                      title="Eliminar filtro"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  </div>
                ))}
              </div>
              <div className="border-t border-gray-200 p-2">
                <button
                  onClick={saveCurrentFilters}
                  className="w-full px-3 py-2 text-sm text-blue-600 hover:bg-blue-50 rounded-lg text-center"
                >
                  + Guardar filtros actuales
                </button>
              </div>
            </div>
          </div>
        )}

        {savedFilters.length === 0 && (
          <button
            onClick={saveCurrentFilters}
            className="px-3 py-2 rounded-lg text-sm font-medium bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300 transition-colors flex items-center gap-2"
            title="Guardar filtros actuales"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
            </svg>
            Guardar filtros
          </button>
        )}

        {/* Botón exportar */}
        <button
          onClick={handleExportTasks}
          className="px-3 py-2 rounded-lg text-sm font-medium bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300 transition-colors flex items-center gap-2"
          title="Exportar tareas a JSON"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          Exportar
        </button>

        {/* Botón de agrupar por repositorio */}
        <button
          onClick={() => setGroupByRepo(!groupByRepo)}
          className={cn(
            "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2",
            groupByRepo 
              ? "bg-blue-100 text-blue-700 border border-blue-300" 
              : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
          )}
          title="Agrupar tareas por repositorio"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
          </svg>
          Agrupar
        </button>

        {/* Botón de estadísticas */}
        <button
          onClick={() => setShowStats(!showStats)}
          className={cn(
            "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2",
            showStats 
              ? "bg-blue-100 text-blue-700 border border-blue-300" 
              : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
          )}
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          Estadísticas
        </button>

        {/* Ayuda de atajos */}
        <div className="text-xs text-gray-500 flex items-center gap-1">
          <kbd className="px-1.5 py-0.5 bg-gray-100 border border-gray-300 rounded text-xs">/</kbd>
          <span>para buscar</span>
        </div>
      </div>
    </div>
  );
}

