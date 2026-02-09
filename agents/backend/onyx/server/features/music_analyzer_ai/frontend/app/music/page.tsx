/**
 * Music Analyzer AI main page.
 * Refactored with improved architecture, code splitting, and maintainability.
 * Optimized for performance and accessibility.
 */

'use client';

import { useQuery } from '@tanstack/react-query';
import { useCallback, useMemo } from 'react';
import { musicApiService } from '@/lib/api';
import { QUERY_KEYS } from '@/lib/constants';
import { useMusicState } from './hooks/use-music-state';
import { MusicHeader, MusicTabs, TabContent } from './components';
import { DynamicMusicComponents } from './components/dynamic-imports';
import { ErrorBoundary } from '@/components/error-boundary';
import { ResponsiveContainer } from '@/components/ui';
import { useKeyboardShortcuts, MUSIC_SHORTCUTS } from '@/lib/hooks';

const {
  MusicDashboard,
  KeyboardShortcuts,
  MusicOffline,
  MusicWelcome,
  MusicTutorial,
  MusicFeedback,
  PerformanceOptimizer,
} = DynamicMusicComponents;

/**
 * Music Analyzer AI main page component.
 * Provides the main interface for music analysis functionality.
 *
 * @returns Music page component
 */
export default function MusicPage() {
  const {
    activeTab,
    selectedTrack,
    analysisData,
    searchResults,
    setActiveTab,
    setSearchResults,
    handleTrackSelect,
  } = useMusicState();

  // Fetch analytics data with proper query key
  const { data: analytics, isLoading: isLoadingAnalytics } = useQuery({
    queryKey: QUERY_KEYS.MUSIC.ANALYTICS,
    queryFn: () => musicApiService.getAnalytics(),
    refetchInterval: 30000, // Refetch every 30 seconds
    staleTime: 60000, // Consider data stale after 1 minute
    gcTime: 5 * 60 * 1000, // Keep in cache for 5 minutes
  });

  /**
   * Handles voice commands.
   * Memoized to prevent unnecessary re-renders.
   */
  const handleVoiceCommand = useCallback(
    (command: string): void => {
      const lowerCommand = command.toLowerCase();
      if (lowerCommand.includes('buscar')) {
        setActiveTab('search');
      } else if (lowerCommand.includes('analizar') && selectedTrack) {
        handleTrackSelect(selectedTrack);
      }
    },
    [setActiveTab, selectedTrack, handleTrackSelect]
  );

  /**
   * Memoized keyboard shortcut handlers.
   */
  const keyboardHandlers = useMemo(
    () => ({
      onSearch: () => setActiveTab('search'),
      onAnalyze: () => {
        if (selectedTrack) {
          handleTrackSelect(selectedTrack);
        }
      },
      onCompare: () => setActiveTab('compare'),
    }),
    [setActiveTab, selectedTrack, handleTrackSelect]
  );

  /**
   * Global keyboard shortcuts for the music page.
   */
  useKeyboardShortcuts({
    shortcuts: [
      {
        key: MUSIC_SHORTCUTS.SEARCH,
        handler: () => setActiveTab('search'),
        description: 'Abrir búsqueda',
      },
      {
        key: MUSIC_SHORTCUTS.ANALYZE,
        handler: () => {
          if (selectedTrack) {
            handleTrackSelect(selectedTrack);
          }
        },
        description: 'Analizar canción seleccionada',
      },
      {
        key: MUSIC_SHORTCUTS.COMPARE,
        handler: () => setActiveTab('compare'),
        description: 'Abrir comparación',
      },
    ],
    enabled: true,
  });

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-pink-900 to-purple-900">
        <MusicOffline />
        <MusicWelcome />
        <MusicTutorial />
        <MusicFeedback />

        <ResponsiveContainer className="py-8">
          <MusicHeader onVoiceCommand={handleVoiceCommand} />

          {/* Keyboard Shortcuts */}
          <KeyboardShortcuts {...keyboardHandlers} />

          {/* Analytics Dashboard */}
          {!isLoadingAnalytics && analytics && (
            <div className="mb-6">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="md:col-span-3">
                  <MusicDashboard analytics={analytics} />
                </div>
                <PerformanceOptimizer />
              </div>
            </div>
          )}

          {/* Tabs Navigation */}
          <MusicTabs activeTab={activeTab} onTabChange={setActiveTab} />

          {/* Tab Content */}
          <TabContent
            activeTab={activeTab}
            selectedTrack={selectedTrack}
            searchResults={searchResults}
            analysisData={analysisData}
            onTrackSelect={handleTrackSelect}
            onSearchResults={setSearchResults}
          />
        </ResponsiveContainer>
      </div>
    </ErrorBoundary>
  );
}
