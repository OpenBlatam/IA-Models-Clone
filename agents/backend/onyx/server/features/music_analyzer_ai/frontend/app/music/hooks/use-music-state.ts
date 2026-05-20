'use client';

/**
 * Custom hook for managing music page state.
 * Refactored with advanced utilities: circuit breaker, retry, performance monitoring, analytics.
 */

import { useState, useCallback } from 'react';
import { useQuery } from '@tanstack/react-query';
import { type Track } from '@/lib/api/types';
import { type TabType } from '../components/music-tabs';
import { analyzeTrack } from '@/lib/api';
import { QUERY_KEYS } from '@/lib/constants';
import { useMusicStore, useCurrentTrack } from '@/lib/store';
import { getErrorMessage } from '@/lib/errors';
import {
  useCircuitBreaker,
  useRetryAdvanced,
  usePerformanceMonitor,
  useAnalytics,
} from '@/lib/hooks';
import toast from 'react-hot-toast';

interface UseMusicStateReturn {
  activeTab: TabType;
  selectedTrack: Track | null;
  analysisData: unknown;
  searchResults: Track[];
  isLoadingAnalysis: boolean;
  setActiveTab: (tab: TabType) => void;
  setSelectedTrack: (track: Track | null) => void;
  setSearchResults: (results: Track[]) => void;
  handleTrackSelect: (track: Track) => Promise<void>;
}

/**
 * Custom hook for managing music page state.
 * Integrates local UI state with global Zustand store.
 * @returns State and handlers for the music page
 */
export function useMusicState(): UseMusicStateReturn {
  const [activeTab, setActiveTab] = useState<TabType>('search');
  const [searchResults, setSearchResults] = useState<Track[]>([]);

  // Get current track from global store
  const currentTrack = useCurrentTrack();
  const setCurrentTrack = useMusicStore((state) => state.setCurrentTrack);
  const addToHistory = useMusicStore((state) => state.addToHistory);

  // Performance monitoring
  const performance = usePerformanceMonitor();

  // Analytics tracking
  const analytics = useAnalytics();

  // Circuit breaker for API resilience
  const { execute: executeWithCircuitBreaker } = useCircuitBreaker({
    failureThreshold: 5,
    resetTimeout: 60000,
    onStateChange: (state) => {
      if (state === 'open') {
        toast.error('Servicio temporalmente no disponible. Reintentando...');
        analytics.error('Circuit breaker opened', { service: 'track-analysis' });
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
    onRetry: (attempt, error) => {
      analytics.track('track_analysis_retry', {
        attempt,
        error: getErrorMessage(error),
      });
    },
  });

  // Enhanced query with circuit breaker, retry, and performance monitoring
  const {
    data: analysisData,
    isLoading: isLoadingAnalysis,
    refetch: refetchAnalysis,
  } = useQuery({
    queryKey: QUERY_KEYS.MUSIC.ANALYSIS(currentTrack?.id || ''),
    queryFn: async () => {
      if (!currentTrack) return null;

      return performance.measurePerformanceAsync(
        'track-analysis',
        async () => {
          return executeWithCircuitBreaker(async () => {
            return retryApiCall(async () => {
              analytics.track('track_analysis_start', {
                trackId: currentTrack.id,
                trackName: currentTrack.name,
              });

              const result = await analyzeTrack({
                trackId: currentTrack.id,
                includeCoaching: true,
              });

              analytics.track('track_analysis_success', {
                trackId: currentTrack.id,
              });

              return result;
            });
          });
        },
        { trackId: currentTrack.id }
      );
    },
    enabled: !!currentTrack,
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 10 * 60 * 1000, // 10 minutes
    retry: false, // We handle retry manually with useRetryAdvanced
  });

  /**
   * Handles track selection and triggers analysis.
   * Enhanced with analytics and performance monitoring.
   */
  const handleTrackSelect = useCallback(
    async (track: Track) => {
      // Track user action
      analytics.action('track_selected', {
        trackId: track.id,
        trackName: track.name,
        artists: track.artists,
      });

      // Update global store
      setCurrentTrack(track);
      addToHistory(track);

      // Trigger analysis query
      try {
        performance.mark('track-analysis-total');
        await refetchAnalysis();
        const duration = performance.measure('track-analysis-total', {
          trackId: track.id,
        });

        analytics.track('track_analysis_complete', {
          trackId: track.id,
          duration,
        });

        toast.success('Análisis completado');
        setActiveTab('analysis');
      } catch (error) {
        const errorMessage = getErrorMessage(error);
        analytics.error(error as Error, {
          trackId: track.id,
          action: 'track_analysis',
        });
        toast.error(errorMessage);
        console.error('Error analyzing track:', error);
      }
    },
    [
      setCurrentTrack,
      addToHistory,
      refetchAnalysis,
      analytics,
      performance,
    ]
  );

  return {
    activeTab,
    selectedTrack: currentTrack,
    analysisData: analysisData || null,
    searchResults,
    isLoadingAnalysis,
    setActiveTab,
    setSelectedTrack: setCurrentTrack,
    setSearchResults,
    handleTrackSelect,
  };
}

