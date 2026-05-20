'use client';

import { useState } from 'react';
import { Filter, X } from 'lucide-react';

interface AdvancedFiltersProps {
  onFilterChange: (filters: any) => void;
}

export function AdvancedFilters({ onFilterChange }: AdvancedFiltersProps) {
  const [filters, setFilters] = useState({
    minPopularity: '',
    maxPopularity: '',
    minEnergy: '',
    maxEnergy: '',
    minDanceability: '',
    maxDanceability: '',
    genre: '',
    year: '',
  });

  const [isOpen, setIsOpen] = useState(false);

  const handleFilterChange = (key: string, value: string) => {
    const newFilters = { ...filters, [key]: value };
    setFilters(newFilters);
    onFilterChange(newFilters);
  };

  const clearFilters = () => {
    const emptyFilters = {
      minPopularity: '',
      maxPopularity: '',
      minEnergy: '',
      maxEnergy: '',
      minDanceability: '',
      maxDanceability: '',
      genre: '',
      year: '',
    };
    setFilters(emptyFilters);
    onFilterChange(emptyFilters);
  };

  const hasActiveFilters = Object.values(filters).some((v) => v !== '');

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Filter className="w-5 h-5 text-purple-300" />
          <h3 className="text-lg font-semibold text-white">Filtros Avanzados</h3>
        </div>
        <div className="flex items-center gap-2">
          {hasActiveFilters && (
            <button
              onClick={clearFilters}
              className="px-3 py-1 text-sm text-red-300 hover:text-red-200 bg-red-500/20 rounded-lg transition-colors"
            >
              Limpiar
            </button>
          )}
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="px-3 py-1 text-sm text-white bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors"
          >
            {isOpen ? 'Ocultar' : 'Mostrar'}
          </button>
        </div>
      </div>

      {isOpen && (
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm text-gray-400 mb-1">Popularidad Mín</label>
            <input
              type="number"
              min="0"
              max="100"
              value={filters.minPopularity}
              onChange={(e) => handleFilterChange('minPopularity', e.target.value)}
              className="w-full px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
              placeholder="0"
            />
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-1">Popularidad Máx</label>
            <input
              type="number"
              min="0"
              max="100"
              value={filters.maxPopularity}
              onChange={(e) => handleFilterChange('maxPopularity', e.target.value)}
              className="w-full px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
              placeholder="100"
            />
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-1">Energía Mín</label>
            <input
              type="number"
              min="0"
              max="1"
              step="0.1"
              value={filters.minEnergy}
              onChange={(e) => handleFilterChange('minEnergy', e.target.value)}
              className="w-full px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
              placeholder="0.0"
            />
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-1">Energía Máx</label>
            <input
              type="number"
              min="0"
              max="1"
              step="0.1"
              value={filters.maxEnergy}
              onChange={(e) => handleFilterChange('maxEnergy', e.target.value)}
              className="w-full px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
              placeholder="1.0"
            />
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-1">Bailabilidad Mín</label>
            <input
              type="number"
              min="0"
              max="1"
              step="0.1"
              value={filters.minDanceability}
              onChange={(e) => handleFilterChange('minDanceability', e.target.value)}
              className="w-full px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
              placeholder="0.0"
            />
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-1">Bailabilidad Máx</label>
            <input
              type="number"
              min="0"
              max="1"
              step="0.1"
              value={filters.maxDanceability}
              onChange={(e) => handleFilterChange('maxDanceability', e.target.value)}
              className="w-full px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
              placeholder="1.0"
            />
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-1">Género</label>
            <input
              type="text"
              value={filters.genre}
              onChange={(e) => handleFilterChange('genre', e.target.value)}
              className="w-full px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
              placeholder="pop, rock, jazz..."
            />
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-1">Año</label>
            <input
              type="number"
              min="1900"
              max="2024"
              value={filters.year}
              onChange={(e) => handleFilterChange('year', e.target.value)}
              className="w-full px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
              placeholder="2024"
            />
          </div>
        </div>
      )}

      {hasActiveFilters && (
        <div className="mt-4 flex flex-wrap gap-2">
          {Object.entries(filters).map(([key, value]) => {
            if (!value) return null;
            return (
              <div
                key={key}
                className="flex items-center gap-1 px-3 py-1 bg-purple-500/30 rounded-full"
              >
                <span className="text-sm text-white">
                  {key}: {value}
                </span>
                <button
                  onClick={() => handleFilterChange(key, '')}
                  className="text-purple-200 hover:text-white"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}


