'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { musicApiService, type Track } from '@/lib/api/music-api';
import { Search, Filter, SlidersHorizontal, Music } from 'lucide-react';
import { debounce } from '@/lib/utils';

interface AdvancedSearchProps {
  onTrackSelect: (track: Track) => void;
}

export function AdvancedSearch({ onTrackSelect }: AdvancedSearchProps) {
  const [query, setQuery] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [filters, setFilters] = useState({
    minPopularity: 0,
    maxPopularity: 100,
    genre: '',
    year: '',
  });
  const [showFilters, setShowFilters] = useState(false);

  const debouncedSearch = debounce((value: string) => {
    setSearchQuery(value);
  }, 500);

  const { data, isLoading } = useQuery({
    queryKey: ['music-search', searchQuery],
    queryFn: () => musicApiService.searchTracks(searchQuery, 20),
    enabled: searchQuery.length > 0,
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setQuery(value);
    debouncedSearch(value);
  };

  const filteredResults = data?.results?.filter((track: Track) => {
    if (track.popularity < filters.minPopularity || track.popularity > filters.maxPopularity) {
      return false;
    }
    return true;
  }) || [];

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-semibold text-white flex items-center gap-2">
          <Search className="w-6 h-6" />
          Búsqueda Avanzada
        </h2>
        <button
          onClick={() => setShowFilters(!showFilters)}
          className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors"
        >
          <SlidersHorizontal className="w-5 h-5" />
          Filtros
        </button>
      </div>

      {/* Search Input */}
      <div className="relative mb-4">
        <input
          type="text"
          value={query}
          onChange={handleInputChange}
          placeholder="Busca canciones, artistas o álbumes..."
          className="w-full px-4 py-3 pl-12 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
        />
        <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-300" />
      </div>

      {/* Filters */}
      {showFilters && (
        <div className="mb-4 p-4 bg-white/5 rounded-lg border border-white/10 space-y-3">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Popularidad Mínima</label>
              <input
                type="number"
                min="0"
                max="100"
                value={filters.minPopularity}
                onChange={(e) => setFilters({ ...filters, minPopularity: parseInt(e.target.value) || 0 })}
                className="w-full px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Popularidad Máxima</label>
              <input
                type="number"
                min="0"
                max="100"
                value={filters.maxPopularity}
                onChange={(e) => setFilters({ ...filters, maxPopularity: parseInt(e.target.value) || 100 })}
                className="w-full px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white"
              />
            </div>
          </div>
        </div>
      )}

      {/* Results */}
      {isLoading && (
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-300" />
        </div>
      )}

      {data && filteredResults.length > 0 && (
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {filteredResults.map((track: Track) => (
            <button
              key={track.id}
              onClick={() => onTrackSelect(track)}
              className="w-full flex items-center gap-3 p-3 bg-white/5 hover:bg-white/10 rounded-lg transition-colors text-left"
            >
              {track.images && track.images[0] ? (
                <img
                  src={track.images[0].url}
                  alt={track.name}
                  className="w-12 h-12 rounded"
                />
              ) : (
                <div className="w-12 h-12 rounded bg-purple-500 flex items-center justify-center">
                  <Music className="w-6 h-6 text-white" />
                </div>
              )}
              <div className="flex-1 min-w-0">
                <p className="text-white font-medium truncate">{track.name}</p>
                <p className="text-sm text-gray-300 truncate">
                  {track.artists.join(', ')}
                </p>
              </div>
              <div className="text-right">
                <p className="text-xs text-gray-400">Popularidad</p>
                <p className="text-sm text-white font-medium">{track.popularity}</p>
              </div>
            </button>
          ))}
        </div>
      )}

      {data && filteredResults.length === 0 && searchQuery && (
        <div className="text-center py-8 text-gray-300">
          No se encontraron resultados con los filtros aplicados
        </div>
      )}
    </div>
  );
}

