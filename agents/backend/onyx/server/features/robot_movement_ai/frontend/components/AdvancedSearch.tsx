'use client';

import { useState } from 'react';
import { Search, Filter, X, Tag } from 'lucide-react';

interface SearchFilter {
  type: 'all' | 'tabs' | 'commands' | 'settings' | 'docs';
  dateRange?: 'today' | 'week' | 'month' | 'all';
  tags?: string[];
}

export default function AdvancedSearch() {
  const [searchTerm, setSearchTerm] = useState('');
  const [filters, setFilters] = useState<SearchFilter>({ type: 'all' });
  const [showFilters, setShowFilters] = useState(false);
  const [results, setResults] = useState<any[]>([]);

  const handleSearch = () => {
    // Simulate search results
    const mockResults = [
      { id: '1', title: 'Control del Robot', type: 'tab', match: 'control' },
      { id: '2', title: 'Mover a posición', type: 'command', match: 'move' },
      { id: '3', title: 'Configuración de tema', type: 'setting', match: 'theme' },
    ];
    setResults(mockResults.filter((r) => 
      r.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      r.match.toLowerCase().includes(searchTerm.toLowerCase())
    ));
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Search className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Búsqueda Avanzada</h3>
        </div>

        {/* Search Bar */}
        <div className="mb-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Buscar en toda la aplicación..."
              className="w-full pl-10 pr-20 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
            <div className="absolute right-2 top-1/2 transform -translate-y-1/2 flex gap-2">
              <button
                onClick={() => setShowFilters(!showFilters)}
                className={`p-2 rounded transition-colors ${
                  showFilters ? 'bg-primary-600 text-white' : 'bg-gray-600 text-gray-300 hover:bg-gray-700'
                }`}
              >
                <Filter className="w-4 h-4" />
              </button>
              {searchTerm && (
                <button
                  onClick={() => {
                    setSearchTerm('');
                    setResults([]);
                  }}
                  className="p-2 bg-gray-600 text-gray-300 rounded hover:bg-gray-700 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Filters */}
        {showFilters && (
          <div className="mb-4 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Tipo</label>
                <select
                  value={filters.type}
                  onChange={(e) => setFilters({ ...filters, type: e.target.value as any })}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
                >
                  <option value="all">Todos</option>
                  <option value="tabs">Pestañas</option>
                  <option value="commands">Comandos</option>
                  <option value="settings">Configuración</option>
                  <option value="docs">Documentación</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Rango de Fecha</label>
                <select
                  value={filters.dateRange || 'all'}
                  onChange={(e) => setFilters({ ...filters, dateRange: e.target.value as any })}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
                >
                  <option value="all">Todas</option>
                  <option value="today">Hoy</option>
                  <option value="week">Esta semana</option>
                  <option value="month">Este mes</option>
                </select>
              </div>
            </div>
          </div>
        )}

        {/* Search Button */}
        <button
          onClick={handleSearch}
          className="w-full px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors flex items-center justify-center gap-2"
        >
          <Search className="w-4 h-4" />
          Buscar
        </button>

        {/* Results */}
        {results.length > 0 && (
          <div className="mt-6">
            <h4 className="text-sm font-medium text-gray-300 mb-3">
              Resultados ({results.length})
            </h4>
            <div className="space-y-2">
              {results.map((result) => (
                <div
                  key={result.id}
                  className="p-3 bg-gray-700/50 rounded-lg border border-gray-600 hover:border-primary-500/50 transition-colors cursor-pointer"
                >
                  <div className="flex items-center gap-2">
                    <Tag className="w-4 h-4 text-primary-400" />
                    <span className="text-white font-medium">{result.title}</span>
                    <span className="ml-auto text-xs text-gray-400 capitalize">{result.type}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {searchTerm && results.length === 0 && (
          <div className="mt-6 text-center py-8 text-gray-400">
            <Search className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No se encontraron resultados</p>
          </div>
        )}
      </div>
    </div>
  );
}


