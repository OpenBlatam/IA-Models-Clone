'use client';

import { useState } from 'react';
import { Search, X } from 'lucide-react';
import { type Track } from '@/lib/api/music-api';

interface QuickSearchProps {
  onTrackSelect: (track: Track) => void;
  placeholder?: string;
}

export function QuickSearch({ onTrackSelect, placeholder = 'Buscar canciones...' }: QuickSearchProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Track[]>([]);
  const [isSearching, setIsSearching] = useState(false);

  const handleSearch = async (searchQuery: string) => {
    if (!searchQuery.trim()) {
      setResults([]);
      return;
    }

    setIsSearching(true);
    try {
      // Simular búsqueda (en producción usaría la API real)
      const mockResults: Track[] = [];
      setResults(mockResults);
    } catch (error) {
      console.error('Error en búsqueda:', error);
    } finally {
      setIsSearching(false);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setQuery(value);
    handleSearch(value);
  };

  const clearSearch = () => {
    setQuery('');
    setResults([]);
  };

  return (
    <div className="relative">
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
        <input
          type="text"
          value={query}
          onChange={handleInputChange}
          placeholder={placeholder}
          className="w-full pl-10 pr-10 py-3 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
        />
        {query && (
          <button
            onClick={clearSearch}
            className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white"
          >
            <X className="w-5 h-5" />
          </button>
        )}
      </div>

      {results.length > 0 && (
        <div className="absolute z-10 w-full mt-2 bg-gray-800 rounded-lg shadow-lg border border-white/20 max-h-96 overflow-y-auto">
          {results.map((track) => (
            <button
              key={track.id}
              onClick={() => {
                onTrackSelect(track);
                clearSearch();
              }}
              className="w-full flex items-center gap-3 p-3 hover:bg-white/10 transition-colors text-left"
            >
              {track.images && track.images[0] && (
                <img
                  src={track.images[0].url}
                  alt={track.name}
                  className="w-12 h-12 rounded"
                />
              )}
              <div className="flex-1 min-w-0">
                <p className="text-white font-medium truncate">{track.name}</p>
                <p className="text-sm text-gray-300 truncate">
                  {Array.isArray(track.artists) ? track.artists.join(', ') : track.artists}
                </p>
              </div>
            </button>
          ))}
        </div>
      )}

      {isSearching && (
        <div className="absolute z-10 w-full mt-2 bg-gray-800 rounded-lg shadow-lg border border-white/20 p-4 text-center">
          <p className="text-gray-400">Buscando...</p>
        </div>
      )}
    </div>
  );
}


