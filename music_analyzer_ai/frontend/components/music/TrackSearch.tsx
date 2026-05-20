'use client';

/**
 * Track search component.
 * Refactored to use advanced utilities: debounce, retry, circuit breaker.
 */

import { useState, useCallback, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { musicApiService, type Track } from '@/lib/api/music-api';
import { QUERY_KEYS, DEBOUNCE_DELAYS } from '@/lib/constants';
import { Search, Loader2 } from 'lucide-react';
import { getErrorMessage } from '@/lib/errors';
import { TrackImage } from '@/components/ui';
import { useDebounceValue, useRetryAdvanced, useCircuitBreaker } from '@/lib/hooks';
import toast from 'react-hot-toast';

interface TrackSearchProps {
  onTrackSelect: (track: Track) => void;
  onSearchResults?: (results: Track[]) => void;
}

/**
 * Track search component with advanced debounce, retry, and circuit breaker.
 */
export function TrackSearch({ onTrackSelect, onSearchResults }: TrackSearchProps) {
  const [query, setQuery] = useState('');
  
  // Use advanced debounce hook
  const debouncedQuery = useDebounceValue(query, {
    delay: DEBOUNCE_DELAYS.SEARCH,
    leading: false,
  });

  // Circuit breaker for API resilience
  const { execute: executeWithCircuitBreaker } = useCircuitBreaker({
    failureThreshold: 5,
    resetTimeout: 60000,
    onStateChange: (state) => {
      if (state === 'open') {
        toast.error('Servicio temporalmente no disponible. Reintentando...');
      }
    },
  });

  // Advanced retry for API calls
  const retryApiCall = useRetryAdvanced({
    maxAttempts: 3,
    strategy: 'exponential',
    initialDelay: 1000,
    backoffMultiplier: 2,
    jitter: true,
    shouldRetry: (error) => {
      const message = getErrorMessage(error);
      return !message.includes('404') && !message.includes('No encontrado');
    },
  });

  // Enhanced query with circuit breaker and retry
  const {
    data,
    isLoading,
    error,
  } = useQuery({
    queryKey: QUERY_KEYS.MUSIC.SEARCH(debouncedQuery),
    queryFn: async () => {
      return executeWithCircuitBreaker(async () => {
        return retryApiCall(async () => {
          return musicApiService.searchTracks(debouncedQuery, 10);
        });
      });
    },
    enabled: debouncedQuery.length > 0,
    staleTime: 30 * 1000, // 30 seconds
    gcTime: 5 * 60 * 1000, // 5 minutes
    retry: false, // We handle retry manually with useRetryAdvanced
    onSuccess: (data) => {
      if (onSearchResults && data.results) {
        onSearchResults(data.results);
      }
    },
    onError: (error) => {
      const errorMessage = getErrorMessage(error);
      toast.error(`Error al buscar: ${errorMessage}`);
    },
  });

  /**
   * Handles input change - debounce is handled by useDebounceValue hook.
   */
  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setQuery(e.target.value);
    },
    []
  );

  /**
   * Handles track selection.
   */
  const handleTrackClick = useCallback(
    (track: Track) => {
      onTrackSelect(track);
      setQuery('');
    },
    [onTrackSelect]
  );

  // Memoize track list to prevent unnecessary re-renders
  const trackList = useMemo(() => {
    if (!data?.results || data.results.length === 0) return null;

    return (
      <div className="space-y-2 max-h-96 overflow-y-auto" role="list">
        {data.results.map((track) => (
          <div key={track.id} role="listitem">
            <button
              onClick={() => handleTrackClick(track)}
              className="w-full flex items-center gap-3 p-3 bg-white/5 hover:bg-white/10 rounded-lg transition-colors text-left"
              type="button"
              aria-label={`Seleccionar ${track.name} de ${track.artists.join(', ')}`}
            >
              <TrackImage
                src={track.images?.[0]?.url}
                alt={track.name}
                width={48}
                height={48}
                className="w-12 h-12"
              />
              <div className="flex-1 min-w-0">
                <p className="text-white font-medium truncate">{track.name}</p>
                <p className="text-sm text-gray-300 truncate">
                  {track.artists.join(', ')}
                </p>
              </div>
            </button>
          </div>
        ))}
      </div>
    );
  }, [data?.results, handleTrackClick]);

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <h2 className="text-2xl font-semibold text-white mb-4 flex items-center gap-2">
        <Search className="w-6 h-6" />
        Buscar Canciones
      </h2>

      <div className="relative mb-4">
        <input
          type="text"
          value={query}
          onChange={handleInputChange}
          placeholder="Busca canciones, artistas o álbumes..."
          className="w-full px-4 py-3 pl-12 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
          aria-label="Buscar canciones"
        />
        <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-300 pointer-events-none" />
      </div>

      {isLoading && (
        <div className="flex items-center justify-center py-8" role="status" aria-label="Buscando">
          <Loader2 className="w-6 h-6 text-purple-300 animate-spin" />
        </div>
      )}

      {error && (
        <div className="text-red-300 text-sm py-4" role="alert">
          Error al buscar canciones. Intenta de nuevo.
        </div>
      )}

      {trackList}

      {data && data.results.length === 0 && debouncedQuery && (
        <div className="text-center py-8 text-gray-300">
          No se encontraron resultados
        </div>
      )}
    </div>
  );
}
