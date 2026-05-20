'use client';

import { useState } from 'react';
import { Filter, X } from 'lucide-react';

interface FilterPanelProps {
  onFilterChange: (filters: any) => void;
}

export function FilterPanel({ onFilterChange }: FilterPanelProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [filters, setFilters] = useState({
    minPopularity: 0,
    maxPopularity: 100,
    minEnergy: 0,
    maxEnergy: 1,
    minDanceability: 0,
    maxDanceability: 1,
    genre: '',
    year: '',
  });

  const handleFilterChange = (key: string, value: any) => {
    const newFilters = { ...filters, [key]: value };
    setFilters(newFilters);
    onFilterChange(newFilters);
  };

  const clearFilters = () => {
    const cleared = {
      minPopularity: 0,
      maxPopularity: 100,
      minEnergy: 0,
      maxEnergy: 1,
      minDanceability: 0,
      maxDanceability: 1,
      genre: '',
      year: '',
    };
    setFilters(cleared);
    onFilterChange(cleared);
  };

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors"
      >
        <Filter className="w-5 h-5" />
        <span>Filtros</span>
      </button>

      {isOpen && (
        <div className="absolute top-full right-0 mt-2 w-80 bg-white/10 backdrop-blur-lg rounded-xl border border-white/20 p-4 z-50">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">Filtros</h3>
            <button
              onClick={() => setIsOpen(false)}
              className="text-gray-400 hover:text-white"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          <div className="space-y-4">
            {/* Popularity Range */}
            <div>
              <label className="block text-sm text-gray-400 mb-2">Popularidad</label>
              <div className="grid grid-cols-2 gap-2">
                <input
                  type="number"
                  min="0"
                  max="100"
                  value={filters.minPopularity}
                  onChange={(e) => handleFilterChange('minPopularity', parseInt(e.target.value) || 0)}
                  placeholder="Min"
                  className="px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white text-sm"
                />
                <input
                  type="number"
                  min="0"
                  max="100"
                  value={filters.maxPopularity}
                  onChange={(e) => handleFilterChange('maxPopularity', parseInt(e.target.value) || 100)}
                  placeholder="Max"
                  className="px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white text-sm"
                />
              </div>
            </div>

            {/* Energy Range */}
            <div>
              <label className="block text-sm text-gray-400 mb-2">Energía</label>
              <div className="grid grid-cols-2 gap-2">
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.1"
                  value={filters.minEnergy}
                  onChange={(e) => handleFilterChange('minEnergy', parseFloat(e.target.value) || 0)}
                  placeholder="Min"
                  className="px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white text-sm"
                />
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.1"
                  value={filters.maxEnergy}
                  onChange={(e) => handleFilterChange('maxEnergy', parseFloat(e.target.value) || 1)}
                  placeholder="Max"
                  className="px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white text-sm"
                />
              </div>
            </div>

            {/* Danceability Range */}
            <div>
              <label className="block text-sm text-gray-400 mb-2">Bailabilidad</label>
              <div className="grid grid-cols-2 gap-2">
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.1"
                  value={filters.minDanceability}
                  onChange={(e) => handleFilterChange('minDanceability', parseFloat(e.target.value) || 0)}
                  placeholder="Min"
                  className="px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white text-sm"
                />
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.1"
                  value={filters.maxDanceability}
                  onChange={(e) => handleFilterChange('maxDanceability', parseFloat(e.target.value) || 1)}
                  placeholder="Max"
                  className="px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white text-sm"
                />
              </div>
            </div>

            {/* Genre */}
            <div>
              <label className="block text-sm text-gray-400 mb-2">Género</label>
              <input
                type="text"
                value={filters.genre}
                onChange={(e) => handleFilterChange('genre', e.target.value)}
                placeholder="Rock, Pop, Electronic..."
                className="w-full px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white text-sm"
              />
            </div>

            {/* Year */}
            <div>
              <label className="block text-sm text-gray-400 mb-2">Año</label>
              <input
                type="number"
                value={filters.year}
                onChange={(e) => handleFilterChange('year', e.target.value)}
                placeholder="2020"
                className="w-full px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white text-sm"
              />
            </div>

            {/* Clear Button */}
            <button
              onClick={clearFilters}
              className="w-full px-4 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-300 rounded-lg transition-colors"
            >
              Limpiar Filtros
            </button>
          </div>
        </div>
      )}
    </div>
  );
}


