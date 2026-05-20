'use client';

/**
 * Music tabs navigation component.
 * Provides tab navigation for different sections of the music analyzer.
 * Refactored with better type safety and accessibility.
 */

import {
  Music,
  BarChart3,
  GitCompare,
  Sparkles,
  Brain,
  Heart,
  ListMusic,
  History,
  Compass,
  TrendingUp,
  Award,
  Network,
  Music2,
  User,
  BarChart,
  FileText,
  PieChart,
  Bookmark,
  Wand2,
  type LucideIcon,
} from 'lucide-react';
import { cn } from '@/lib/utils';

/**
 * Tab type definition.
 */
export type TabType =
  | 'search'
  | 'analysis'
  | 'compare'
  | 'recommendations'
  | 'ml'
  | 'favorites'
  | 'playlists'
  | 'history'
  | 'discovery'
  | 'trends'
  | 'quality'
  | 'collaborations'
  | 'covers'
  | 'artists'
  | 'playlist-analysis'
  | 'batch'
  | 'statistics'
  | 'bookmarks'
  | 'playlist-generator';

/**
 * Tab configuration interface.
 */
export interface TabConfig {
  id: TabType;
  label: string;
  icon: LucideIcon;
  description?: string;
}

/**
 * Music tabs configuration.
 * Centralized configuration for all available tabs.
 */
export const MUSIC_TABS: TabConfig[] = [
  { id: 'search', label: 'Buscar', icon: Music, description: 'Buscar canciones' },
  { id: 'analysis', label: 'Análisis', icon: BarChart3, description: 'Análisis de canciones' },
  { id: 'compare', label: 'Comparar', icon: GitCompare, description: 'Comparar canciones' },
  {
    id: 'recommendations',
    label: 'Recomendaciones',
    icon: Sparkles,
    description: 'Recomendaciones personalizadas',
  },
  { id: 'ml', label: 'ML', icon: Brain, description: 'Análisis con ML' },
  { id: 'favorites', label: 'Favoritos', icon: Heart, description: 'Canciones favoritas' },
  { id: 'playlists', label: 'Playlists', icon: ListMusic, description: 'Gestionar playlists' },
  { id: 'history', label: 'Historial', icon: History, description: 'Historial de búsquedas' },
  { id: 'discovery', label: 'Descubrir', icon: Compass, description: 'Descubrir música' },
  { id: 'trends', label: 'Tendencias', icon: TrendingUp, description: 'Tendencias musicales' },
  { id: 'quality', label: 'Calidad', icon: Award, description: 'Análisis de calidad' },
  {
    id: 'collaborations',
    label: 'Colaboraciones',
    icon: Network,
    description: 'Colaboraciones',
  },
  { id: 'covers', label: 'Covers/Remixes', icon: Music2, description: 'Covers y remixes' },
  { id: 'artists', label: 'Artistas', icon: User, description: 'Análisis de artistas' },
  {
    id: 'playlist-analysis',
    label: 'Análisis Playlist',
    icon: BarChart,
    description: 'Análisis de playlists',
  },
  { id: 'batch', label: 'Análisis Lote', icon: FileText, description: 'Análisis en lote' },
  { id: 'statistics', label: 'Estadísticas', icon: PieChart, description: 'Estadísticas' },
  { id: 'bookmarks', label: 'Marcadores', icon: Bookmark, description: 'Marcadores' },
  {
    id: 'playlist-generator',
    label: 'Generar Playlist',
    icon: Wand2,
    description: 'Generar playlist automática',
  },
] as const;

interface MusicTabsProps {
  activeTab: TabType;
  onTabChange: (tab: TabType) => void;
}

/**
 * Music tabs navigation component.
 * Provides accessible tab navigation with keyboard support.
 *
 * @param props - Component props
 * @returns Music tabs component
 */
export function MusicTabs({ activeTab, onTabChange }: MusicTabsProps) {
  /**
   * Handles keyboard navigation.
   */
  const handleKeyDown = (
    e: React.KeyboardEvent<HTMLButtonElement>,
    tabId: TabType
  ) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onTabChange(tabId);
    }
  };

  return (
    <nav className="mb-6" aria-label="Music analyzer tabs">
      <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-hide">
        {MUSIC_TABS.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;

          return (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              onKeyDown={(e) => handleKeyDown(e, tab.id)}
              className={cn(
                'flex items-center gap-2 px-6 py-3 rounded-lg transition-colors whitespace-nowrap',
                'focus:outline-none focus:ring-2 focus:ring-purple-400 focus:ring-offset-2 focus:ring-offset-purple-900',
                isActive
                  ? 'bg-purple-600 text-white'
                  : 'bg-white/10 text-gray-300 hover:bg-white/20'
              )}
              aria-label={tab.description || tab.label}
              aria-selected={isActive}
              role="tab"
              type="button"
            >
              <Icon className="w-5 h-5" aria-hidden="true" />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>
    </nav>
  );
}
