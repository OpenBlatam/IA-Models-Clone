'use client';

/**
 * Tab content component that dynamically loads content based on active tab.
 * Refactored to use a component map for better maintainability.
 */

import { useMemo } from 'react';
import { type TabType } from './music-tabs';
import { type Track } from '@/lib/api/types';
import dynamic from 'next/dynamic';
import { Suspense } from 'react';
import { TabLoadingState } from '@/components/ui';

/**
 * Dynamic import configuration for tabs.
 */
const dynamicImportConfig = {
  loading: () => <TabLoadingState />,
  ssr: false,
} as const;

/**
 * Tab component map for dynamic loading.
 * Maps tab types to their component import paths.
 */
const tabComponentMap: Record<TabType, () => Promise<{ default: React.ComponentType<TabContentProps> }>> = {
  search: () => import('./tabs/search-tab'),
  analysis: () => import('./tabs/analysis-tab'),
  compare: () => import('./tabs/compare-tab'),
  recommendations: () => import('./tabs/recommendations-tab'),
  ml: () => import('./tabs/ml-tab'),
  favorites: () => import('./tabs/favorites-tab'),
  playlists: () => import('./tabs/playlists-tab'),
  history: () => import('./tabs/history-tab'),
  discovery: () => import('./tabs/discovery-tab'),
  trends: () => import('./tabs/trends-tab'),
  quality: () => import('./tabs/quality-tab'),
  collaborations: () => import('./tabs/collaborations-tab'),
  covers: () => import('./tabs/covers-tab'),
  artists: () => import('./tabs/artists-tab'),
  'playlist-analysis': () => import('./tabs/playlist-analysis-tab'),
  batch: () => import('./tabs/batch-tab'),
  statistics: () => import('./tabs/statistics-tab'),
  bookmarks: () => import('./tabs/bookmarks-tab'),
  'playlist-generator': () => import('./tabs/playlist-generator-tab'),
};

/**
 * Creates dynamic components for all tabs.
 */
const createTabComponents = () => {
  const components: Partial<Record<TabType, React.ComponentType<TabContentProps>>> = {};
  
  for (const [tabType, importFn] of Object.entries(tabComponentMap)) {
    components[tabType as TabType] = dynamic(importFn, dynamicImportConfig);
  }
  
  return components as Record<TabType, React.ComponentType<TabContentProps>>;
};

interface TabContentProps {
  selectedTrack: Track | null;
  searchResults: Track[];
  analysisData: unknown;
  onTrackSelect: (track: Track) => void;
  onSearchResults: (results: Track[]) => void;
}

interface TabContentComponentProps {
  activeTab: TabType;
  selectedTrack: Track | null;
  searchResults: Track[];
  analysisData: unknown;
  onTrackSelect: (track: Track) => void;
  onSearchResults: (results: Track[]) => void;
}

/**
 * Tab content component that renders the appropriate tab based on activeTab.
 * Uses memoized component map for better performance.
 */
export function TabContent({
  activeTab,
  selectedTrack,
  searchResults,
  analysisData,
  onTrackSelect,
  onSearchResults,
}: TabContentComponentProps) {
  // Memoize tab components to prevent recreation on each render
  const tabComponents = useMemo(() => createTabComponents(), []);

  const commonProps: TabContentProps = useMemo(
    () => ({
      selectedTrack,
      searchResults,
      analysisData,
      onTrackSelect,
      onSearchResults,
    }),
    [selectedTrack, searchResults, analysisData, onTrackSelect, onSearchResults]
  );

  const ActiveTabComponent = tabComponents[activeTab];

  if (!ActiveTabComponent) {
    return (
      <div className="text-center py-12 text-gray-300">
        Tab no encontrado: {activeTab}
      </div>
    );
  }

  return (
    <Suspense fallback={<TabLoadingState />}>
      <div className="space-y-6">
        <ActiveTabComponent {...commonProps} />
      </div>
    </Suspense>
  );
}
