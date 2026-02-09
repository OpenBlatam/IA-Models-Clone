'use client';

/**
 * Music page header component.
 * Refactored to optimize dynamic imports and improve maintainability.
 */

import { useMemo } from 'react';
import { Music } from 'lucide-react';
import dynamic from 'next/dynamic';

/**
 * Dynamic import configuration for header components.
 */
const dynamicImportConfig = {
  ssr: false,
} as const;

/**
 * Header component map for dynamic loading.
 */
const headerComponentMap = {
  MusicKeyboardShortcuts: () =>
    import('@/components/music/MusicKeyboardShortcuts').then(
      (mod) => ({ default: mod.MusicKeyboardShortcuts })
    ),
  MusicAccessibility: () =>
    import('@/components/music/MusicAccessibility').then(
      (mod) => ({ default: mod.MusicAccessibility })
    ),
  MusicSettings: () =>
    import('@/components/music/MusicSettings').then(
      (mod) => ({ default: mod.MusicSettings })
    ),
  MusicSync: () =>
    import('@/components/music/MusicSync').then(
      (mod) => ({ default: mod.MusicSync })
    ),
  RealTimeUpdates: () =>
    import('@/components/music/RealTimeUpdates').then(
      (mod) => ({ default: mod.RealTimeUpdates })
    ),
  VoiceCommands: () =>
    import('@/components/music/VoiceCommands').then(
      (mod) => ({ default: mod.VoiceCommands })
    ),
  MusicNotifications: () =>
    import('@/components/music/MusicNotifications').then(
      (mod) => ({ default: mod.MusicNotifications })
    ),
  MusicTour: () =>
    import('@/components/music/MusicTour').then(
      (mod) => ({ default: mod.MusicTour })
    ),
  MusicTips: () =>
    import('@/components/music/MusicTips').then(
      (mod) => ({ default: mod.MusicTips })
    ),
  NotificationCenter: () =>
    import('@/components/music/NotificationCenter').then(
      (mod) => ({ default: mod.NotificationCenter })
    ),
  ThemeToggle: () =>
    import('@/components/music/ThemeToggle').then(
      (mod) => ({ default: mod.ThemeToggle })
    ),
} as const;

/**
 * Creates dynamic components for header.
 */
const createHeaderComponents = () => {
  const components: Record<
    keyof typeof headerComponentMap,
    React.ComponentType<any>
  > = {} as any;

  for (const [key, importFn] of Object.entries(headerComponentMap)) {
    components[key as keyof typeof headerComponentMap] = dynamic(
      importFn,
      dynamicImportConfig
    );
  }

  return components;
};

interface MusicHeaderProps {
  onVoiceCommand?: (command: string) => void;
}

/**
 * Music page header component.
 * Optimized with memoized dynamic components.
 */
export function MusicHeader({ onVoiceCommand }: MusicHeaderProps) {
  // Memoize header components to prevent recreation on each render
  const headerComponents = useMemo(() => createHeaderComponents(), []);

  const {
    MusicKeyboardShortcuts,
    MusicAccessibility,
    MusicSettings,
    MusicSync,
    RealTimeUpdates,
    VoiceCommands,
    MusicNotifications,
    MusicTour,
    MusicTips,
    NotificationCenter,
    ThemeToggle,
  } = headerComponents;

  return (
    <div className="mb-8">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <Music className="w-10 h-10 text-purple-300" />
          <div>
            <h1 className="text-4xl font-bold text-white">Music Analyzer AI</h1>
            <p className="text-gray-300">
              Analiza canciones, obtén insights musicales y coaching personalizado
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <MusicKeyboardShortcuts />
          <MusicAccessibility />
          <MusicSettings />
          <MusicSync />
          <RealTimeUpdates />
          {onVoiceCommand && (
            <VoiceCommands onCommand={onVoiceCommand} />
          )}
          <MusicNotifications />
          <MusicTour />
          <MusicTips />
          <NotificationCenter />
          <ThemeToggle />
        </div>
      </div>
    </div>
  );
}
