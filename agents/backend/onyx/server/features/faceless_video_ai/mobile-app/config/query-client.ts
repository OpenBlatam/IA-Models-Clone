import { QueryClient } from '@tanstack/react-query';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: (failureCount, error: any) => {
        // Don't retry on 4xx errors
        if (error?.status_code >= 400 && error?.status_code < 500) {
          return false;
        }
        return failureCount < 3;
      },
      staleTime: 5 * 60 * 1000, // 5 minutes
      gcTime: 10 * 60 * 1000, // 10 minutes (formerly cacheTime)
      refetchOnWindowFocus: false,
      refetchOnReconnect: true,
    },
    mutations: {
      retry: 1,
    },
  },
});

// Query key factories for type safety
export const queryKeys = {
  auth: {
    me: ['auth', 'me'] as const,
    apiKey: ['auth', 'api-key'] as const,
  },
  videos: {
    all: ['videos'] as const,
    detail: (id: string) => ['videos', id] as const,
    status: (id: string) => ['videos', id, 'status'] as const,
    feedback: (id: string) => ['videos', id, 'feedback'] as const,
    versions: (id: string) => ['videos', id, 'versions'] as const,
    version: (id: string, version: number) =>
      ['videos', id, 'version', version] as const,
  },
  templates: {
    all: ['templates'] as const,
    detail: (name: string) => ['templates', name] as const,
    custom: (userOnly?: boolean) =>
      ['templates', 'custom', userOnly] as const,
  },
  search: {
    videos: (query: string, filters?: unknown) =>
      ['search', 'videos', query, filters] as const,
    suggestions: (query: string) => ['search', 'suggestions', query] as const,
  },
  batch: {
    status: (ids: string[]) => ['batch', 'status', ids.sort().join(',')] as const,
  },
  music: {
    tracks: (style?: string) => ['music', 'tracks', style] as const,
  },
  analytics: {
    all: ['analytics'] as const,
    recommendations: (
      script: string,
      language: string,
      platform?: string,
      contentType?: string
    ) => ['analytics', 'recommendations', script, language, platform, contentType] as const,
    quota: ['analytics', 'quota'] as const,
  },
  scheduled: {
    all: (videoId?: string, status?: string) =>
      ['scheduled', videoId, status] as const,
  },
} as const;


