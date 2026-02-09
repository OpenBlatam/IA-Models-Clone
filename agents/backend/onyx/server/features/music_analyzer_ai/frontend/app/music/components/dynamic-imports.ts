/**
 * Dynamic imports configuration.
 * Centralized dynamic imports for better code splitting and performance.
 */

import dynamic from 'next/dynamic';
import { LoadingState } from '@/components/ui';

/**
 * Dynamic import configuration.
 */
const dynamicConfig = {
  ssr: false,
  loading: () => <LoadingState message="Cargando..." />,
} as const;

/**
 * Dynamically imported components for the music page.
 * All components are lazy-loaded for better performance.
 */
export const DynamicMusicComponents = {
  MusicDashboard: dynamic(
    () =>
      import('@/components/music/MusicDashboard').then((mod) => ({
        default: mod.MusicDashboard,
      })),
    dynamicConfig
  ),

  KeyboardShortcuts: dynamic(
    () =>
      import('@/components/music/KeyboardShortcuts').then((mod) => ({
        default: mod.KeyboardShortcuts,
      })),
    dynamicConfig
  ),

  MusicOffline: dynamic(
    () =>
      import('@/components/music/MusicOffline').then((mod) => ({
        default: mod.MusicOffline,
      })),
    dynamicConfig
  ),

  MusicWelcome: dynamic(
    () =>
      import('@/components/music/MusicWelcome').then((mod) => ({
        default: mod.MusicWelcome,
      })),
    dynamicConfig
  ),

  MusicTutorial: dynamic(
    () =>
      import('@/components/music/MusicTutorial').then((mod) => ({
        default: mod.MusicTutorial,
      })),
    dynamicConfig
  ),

  MusicFeedback: dynamic(
    () =>
      import('@/components/music/MusicFeedback').then((mod) => ({
        default: mod.MusicFeedback,
      })),
    dynamicConfig
  ),

  PerformanceOptimizer: dynamic(
    () =>
      import('@/components/music/PerformanceOptimizer').then((mod) => ({
        default: mod.PerformanceOptimizer,
      })),
    dynamicConfig
  ),
} as const;

