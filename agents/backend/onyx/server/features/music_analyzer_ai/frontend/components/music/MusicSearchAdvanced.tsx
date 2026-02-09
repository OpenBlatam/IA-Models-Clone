'use client';

/**
 * Advanced search component with filters and validation.
 * Refactored with advanced utilities: debounce, retry, circuit breaker, form validation.
 */

import { useState, useCallback, useMemo, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { musicApiService, type Track } from '@/lib/api/music-api';
import { QUERY_KEYS, DEBOUNCE_DELAYS } from '@/lib/constants';
import { searchRequestSchema } from '@/lib/validations/music';
import { Search, SlidersHorizontal } from 'lucide-react';
import { getErrorMessage } from '@/lib/errors';
import { LoadingSpinner, ErrorMessage, TrackImage } from '@/components/ui';
import {
  useDebounceValue,
  useRetryAdvanced,
  useCircuitBreaker,
  useFormValidation,
} from '@/lib/hooks';
import toast from 'react-hot-toast';

interface AdvancedSearchProps {
  onTrackSelect: (track: Track) => void;
  onResults?: (results: Track[]) => void;
}

interface SearchFilters {
  minPopularity: number;
  maxPopularity: number;
  genre: string;
  year: string;
}

/**
 * Default filter values.
 */
const DEFAULT_FILTERS: SearchFilters = {
  minPopularity: 0,
  maxPopularity: 100,
  genre: '',
  year: '',
};

/**
 * Advanced search component with filters.
 *
 * @param props - Component props
 * @returns Advanced search component
 */
export function MusicSearchAdvanced({
  onTrackSelect,
  onResults,
}: AdvancedSearchProps) {
  const [query, setQuery] = useState('');
  const [filters, setFilters] = useState<SearchFilters>(DEFAULT_FILTERS);
  const [showFilters, setShowFilters] = useState(false);

  // Use advanced debounce hook
  const debouncedQuery = useDebounceValue(query, {
    delay: DEBOUNCE_DELAYS.SEARCH,
    leading: false,
  });

  // Form validation hook
  const {
    validate,
    errors: validationErrors,
  } = useFormValidation(searchRequestSchema);

  // Validate debounced query
  const [validatedQuery, setValidatedQuery] = useState('');
  useEffect(() => {
    if (debouncedQuery) {
      const result = validate({ query: debouncedQuery });
      if (result.isValid) {
        setValidatedQuery(debouncedQuery);
      } else {
        setValidatedQuery('');
      }
    } else {
      setValidatedQuery('');
    }
  }, [debouncedQuery, validate]);

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
    queryKey: QUERY_KEYS.MUSIC.SEARCH(validatedQuery),
    queryFn: async () => {
      return executeWithCircuitBreaker(async () => {
        return retryApiCall(async () => {
          return musicApiService.searchTracks(validatedQuery, 20);
        });
      });
    },
    enabled: validatedQuery.length > 0,
    staleTime: 30 * 1000, // 30 seconds
    gcTime: 5 * 60 * 1000, // 5 minutes
    retry: false, // We handle retry manually with useRetryAdvanced
    onSuccess: (data) => {
      if (onResults && data.results) {
        onResults(data.results);
      }
    },
    onError: (error) => {
      const errorMessage = getErrorMessage(error);
      toast.error(`Error en búsqueda: ${errorMessage}`);
    },
  });

  /**
   * Handles input change - debounce and validation are handled by hooks.
   */
  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setQuery(e.target.value);
    },
    []
  );

  /**
   * Handles filter change.
   */
  const handleFilterChange = useCallback(
    (key: keyof SearchFilters, value: string | number) => {
      setFilters((prev) => ({
        ...prev,
        [key]: value,
      }));
    },
    []
  );

  /**
   * Resets filters to default values.
   */
  const resetFilters = useCallback(() => {
    setFilters(DEFAULT_FILTERS);
  }, []);

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

  // Filter results based on filters
  const filteredResults = useMemo(() => {
    if (!data?.results) return [];

    return data.results.filter((track: Track) => {
      if (
        track.popularity < filters.minPopularity ||
        track.popularity > filters.maxPopularity
      ) {
        return false;
      }
      // Add more filter logic here if needed
      return true;
    });
  }, [data?.results, filters]);

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
          type="button"
          aria-label="Toggle filters"
          aria-expanded={showFilters}
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
          aria-label="Buscar canciones"
          aria-invalid={!!validationErrors.query}
          aria-describedby={validationErrors.query ? 'search-error' : undefined}
        />
        <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-300 pointer-events-none" />
      </div>

      {/* Validation Error */}
      {validationErrors.query && (
        <div id="search-error" className="mb-4">
          <ErrorMessage message={validationErrors.query} variant="inline" />
        </div>
      )}

      {/* Filters */}
      {showFilters && (
        <div className="mb-4 p-4 bg-white/5 rounded-lg border border-white/10 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-white font-medium">Filtros</h3>
            <button
              onClick={resetFilters}
              className="text-sm text-gray-300 hover:text-white transition-colors"
              type="button"
            >
              Resetear
            </button>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Popularidad Mínima
              </label>
              <input
                type="number"
                min="0"
                max="100"
                value={filters.minPopularity}
                onChange={(e) =>
                  handleFilterChange('minPopularity', Number(e.target.value))
                }
                className="w-full px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Popularidad Máxima
              </label>
              <input
                type="number"
                min="0"
                max="100"
                value={filters.maxPopularity}
                onChange={(e) =>
                  handleFilterChange('maxPopularity', Number(e.target.value))
                }
                className="w-full px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white"
              />
            </div>
          </div>
        </div>
      )}

      {/* Loading State */}
      {isLoading && (
        <div className="flex items-center justify-center py-8" role="status">
          <LoadingSpinner size="lg" />
        </div>
      )}

      {/* Error State */}
      {error && (
        <ErrorMessage
          message="Error al buscar canciones. Intenta de nuevo."
          variant="banner"
        />
      )}

      {/* Results */}
      {!isLoading && !error && filteredResults.length > 0 && (
        <div className="space-y-2 max-h-96 overflow-y-auto" role="list" aria-label="Resultados de búsqueda">
          {filteredResults.map((track) => (
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
      )}

      {/* Empty State */}
      {!isLoading && !error && validatedQuery && filteredResults.length === 0 && (
        <div className="text-center py-8 text-gray-300">
          No se encontraron resultados
        </div>
      )}
    </div>
  );
}
