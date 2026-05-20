'use client';

import { useState } from 'react';
import { Filter, X } from 'lucide-react';

interface SearchFiltersProps {
  onFilterChange: (filters: any) => void;
}

export function SearchFilters({ onFilterChange }: SearchFiltersProps) {
  const [filters, setFilters] = useState({
    genre: '',
    year: '',
    popularity: '',
    energy: '',
    danceability: '',
  });

  const handleFilterChange = (key: string, value: string) => {
    const newFilters = { ...filters, [key]: value };
    setFilters(newFilters);
    onFilterChange(newFilters);
  };

  const clearFilters = () => {
    const emptyFilters = {
      genre: '',
      year: '',
      popularity: '',
      energy: '',
      danceability: '',
    };
    setFilters(emptyFilters);
    onFilterChange(emptyFilters);
  };

  const hasActiveFilters = Object.values(filters).some((v) => v !== '');

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-purple-300" />
          <h3 className="text-sm font-semibold text-white">Filtros Rápidos</h3>
        </div>
        {hasActiveFilters && (
          <button
            onClick={clearFilters}
            className="text-xs text-red-300 hover:text-red-200"
          >
            Limpiar
          </button>
        )}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
        <select
          value={filters.genre}
          onChange={(e) => handleFilterChange('genre', e.target.value)}
          className="px-2 py-1 text-sm bg-white/20 border border-white/30 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-400"
        >
          <option value="">Género</option>
          <option value="pop">Pop</option>
          <option value="rock">Rock</option>
          <option value="jazz">Jazz</option>
          <option value="electronic">Electronic</option>
          <option value="hip-hop">Hip-Hop</option>
        </select>

        <input
          type="number"
          placeholder="Año"
          value={filters.year}
          onChange={(e) => handleFilterChange('year', e.target.value)}
          className="px-2 py-1 text-sm bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
        />

        <select
          value={filters.popularity}
          onChange={(e) => handleFilterChange('popularity', e.target.value)}
          className="px-2 py-1 text-sm bg-white/20 border border-white/30 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-400"
        >
          <option value="">Popularidad</option>
          <option value="high">Alta</option>
          <option value="medium">Media</option>
          <option value="low">Baja</option>
        </select>

        <select
          value={filters.energy}
          onChange={(e) => handleFilterChange('energy', e.target.value)}
          className="px-2 py-1 text-sm bg-white/20 border border-white/30 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-400"
        >
          <option value="">Energía</option>
          <option value="high">Alta</option>
          <option value="medium">Media</option>
          <option value="low">Baja</option>
        </select>

        <select
          value={filters.danceability}
          onChange={(e) => handleFilterChange('danceability', e.target.value)}
          className="px-2 py-1 text-sm bg-white/20 border border-white/30 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-400"
        >
          <option value="">Bailabilidad</option>
          <option value="high">Alta</option>
          <option value="medium">Media</option>
          <option value="low">Baja</option>
        </select>
      </div>
    </div>
  );
}


