'use client';

import { useState } from 'react';
import { Filter, X } from 'lucide-react';

interface MusicFiltersProps {
  onFilterChange: (filters: any) => void;
}

export function MusicFilters({ onFilterChange }: MusicFiltersProps) {
  const [filters, setFilters] = useState({
    genre: [] as string[],
    year: { min: '', max: '' },
    popularity: { min: '', max: '' },
    energy: { min: '', max: '' },
    danceability: { min: '', max: '' },
    tempo: { min: '', max: '' },
  });

  const genres = ['Pop', 'Rock', 'Jazz', 'Electronic', 'Hip-Hop', 'Classical', 'Country', 'R&B'];

  const handleGenreToggle = (genre: string) => {
    const newGenres = filters.genre.includes(genre)
      ? filters.genre.filter((g) => g !== genre)
      : [...filters.genre, genre];
    const newFilters = { ...filters, genre: newGenres };
    setFilters(newFilters);
    onFilterChange(newFilters);
  };

  const handleRangeChange = (category: string, field: 'min' | 'max', value: string) => {
    const newFilters = {
      ...filters,
      [category]: { ...filters[category as keyof typeof filters], [field]: value },
    };
    setFilters(newFilters);
    onFilterChange(newFilters);
  };

  const clearFilters = () => {
    const emptyFilters = {
      genre: [],
      year: { min: '', max: '' },
      popularity: { min: '', max: '' },
      energy: { min: '', max: '' },
      danceability: { min: '', max: '' },
      tempo: { min: '', max: '' },
    };
    setFilters(emptyFilters);
    onFilterChange(emptyFilters);
  };

  const hasActiveFilters =
    filters.genre.length > 0 ||
    Object.values(filters).some((v) => typeof v === 'object' && (v.min || v.max));

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Filter className="w-5 h-5 text-purple-300" />
          <h3 className="text-lg font-semibold text-white">Filtros Avanzados</h3>
        </div>
        {hasActiveFilters && (
          <button
            onClick={clearFilters}
            className="px-3 py-1 text-sm text-red-300 hover:text-red-200 bg-red-500/20 rounded-lg transition-colors flex items-center gap-1"
          >
            <X className="w-4 h-4" />
            Limpiar
          </button>
        )}
      </div>

      <div className="space-y-6">
        <div>
          <label className="block text-sm text-gray-400 mb-2">Géneros</label>
          <div className="flex flex-wrap gap-2">
            {genres.map((genre) => (
              <button
                key={genre}
                onClick={() => handleGenreToggle(genre)}
                className={`px-3 py-1 rounded-lg text-sm transition-colors ${
                  filters.genre.includes(genre)
                    ? 'bg-purple-600 text-white'
                    : 'bg-white/10 text-gray-300 hover:bg-white/20'
                }`}
              >
                {genre}
              </button>
            ))}
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm text-gray-400 mb-2">Año</label>
            <div className="flex gap-2">
              <input
                type="number"
                placeholder="Desde"
                value={filters.year.min}
                onChange={(e) => handleRangeChange('year', 'min', e.target.value)}
                className="flex-1 px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
              />
              <input
                type="number"
                placeholder="Hasta"
                value={filters.year.max}
                onChange={(e) => handleRangeChange('year', 'max', e.target.value)}
                className="flex-1 px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-2">Popularidad</label>
            <div className="flex gap-2">
              <input
                type="number"
                min="0"
                max="100"
                placeholder="Min"
                value={filters.popularity.min}
                onChange={(e) => handleRangeChange('popularity', 'min', e.target.value)}
                className="flex-1 px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
              />
              <input
                type="number"
                min="0"
                max="100"
                placeholder="Max"
                value={filters.popularity.max}
                onChange={(e) => handleRangeChange('popularity', 'max', e.target.value)}
                className="flex-1 px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-2">Energía</label>
            <div className="flex gap-2">
              <input
                type="number"
                min="0"
                max="1"
                step="0.1"
                placeholder="Min"
                value={filters.energy.min}
                onChange={(e) => handleRangeChange('energy', 'min', e.target.value)}
                className="flex-1 px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
              />
              <input
                type="number"
                min="0"
                max="1"
                step="0.1"
                placeholder="Max"
                value={filters.energy.max}
                onChange={(e) => handleRangeChange('energy', 'max', e.target.value)}
                className="flex-1 px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-2">Bailabilidad</label>
            <div className="flex gap-2">
              <input
                type="number"
                min="0"
                max="1"
                step="0.1"
                placeholder="Min"
                value={filters.danceability.min}
                onChange={(e) => handleRangeChange('danceability', 'min', e.target.value)}
                className="flex-1 px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
              />
              <input
                type="number"
                min="0"
                max="1"
                step="0.1"
                placeholder="Max"
                value={filters.danceability.max}
                onChange={(e) => handleRangeChange('danceability', 'max', e.target.value)}
                className="flex-1 px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}


