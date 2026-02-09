/**
 * Test Utilities
 * Reusable utilities and helpers for testing
 */

import { render, RenderOptions } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactElement, ReactNode } from 'react';
import { useMusicStore } from '@/lib/store/music-store';

/**
 * Creates a test QueryClient with default options
 */
export function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        cacheTime: 0,
        staleTime: 0,
      },
      mutations: {
        retry: false,
      },
    },
  });
}

/**
 * Wrapper component for React Query
 */
function QueryClientWrapper({ children }: { children: ReactNode }) {
  const queryClient = createTestQueryClient();
  return (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

/**
 * Renders component with QueryClient provider
 */
export function renderWithQueryClient(
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
) {
  return render(ui, { wrapper: QueryClientWrapper, ...options });
}

/**
 * Resets the music store to initial state
 */
export function resetMusicStore() {
  useMusicStore.getState().clearQueue();
  useMusicStore.getState().clearRecentSearches();
  useMusicStore.getState().clearFilters();
  useMusicStore.getState().clearSelection();
  useMusicStore.getState().clearHistory();
  useMusicStore.getState().setCurrentTrack(null);
  useMusicStore.getState().resetPlayback();
  useMusicStore.getState().resetViewPreferences();
}

/**
 * Waits for async operations to complete
 */
export async function waitForAsync() {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Creates a mock track for testing
 */
export function createMockTrack(overrides: Partial<any> = {}) {
  return {
    id: 'track-123',
    name: 'Test Track',
    artists: ['Test Artist'],
    album: 'Test Album',
    duration_ms: 200000,
    preview_url: 'https://example.com/preview.mp3',
    popularity: 80,
    images: [{ url: 'https://example.com/image.jpg' }],
    ...overrides,
  };
}

/**
 * Creates multiple mock tracks
 */
export function createMockTracks(count: number) {
  return Array.from({ length: count }, (_, i) =>
    createMockTrack({
      id: `track-${i + 1}`,
      name: `Track ${i + 1}`,
      artists: [`Artist ${i + 1}`],
    })
  );
}

/**
 * Creates a mock API response
 */
export function createMockApiResponse<T>(data: T, success = true) {
  return {
    success,
    ...(success ? { data } : { error: 'Error message' }),
  };
}

/**
 * Creates a mock paginated response
 */
export function createMockPaginatedResponse<T>(
  items: T[],
  page = 1,
  pageSize = 20
) {
  return {
    data: items,
    meta: {
      page,
      limit: pageSize,
      total: items.length,
      totalPages: Math.ceil(items.length / pageSize),
      hasNext: page * pageSize < items.length,
      hasPrev: page > 1,
    },
  };
}

/**
 * Mocks localStorage
 */
export function mockLocalStorage() {
  const store: Record<string, string> = {};

  return {
    getItem: jest.fn((key: string) => store[key] || null),
    setItem: jest.fn((key: string, value: string) => {
      store[key] = value.toString();
    }),
    removeItem: jest.fn((key: string) => {
      delete store[key];
    }),
    clear: jest.fn(() => {
      Object.keys(store).forEach((key) => delete store[key]);
    }),
  };
}

/**
 * Mocks window.matchMedia
 */
export function mockMatchMedia(matches = false) {
  return jest.fn().mockImplementation((query) => ({
    matches,
    media: query,
    onchange: null,
    addListener: jest.fn(),
    removeListener: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  }));
}

/**
 * Mocks IntersectionObserver
 */
export function mockIntersectionObserver() {
  return class IntersectionObserver {
    constructor() {}
    disconnect() {}
    observe() {}
    takeRecords() {
      return [];
    }
    unobserve() {}
  };
}

/**
 * Mocks Audio API
 */
export function mockAudio() {
  return jest.fn().mockImplementation(() => ({
    play: jest.fn().mockResolvedValue(undefined),
    pause: jest.fn(),
    load: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    currentTime: 0,
    duration: 100,
    volume: 1,
    muted: false,
    paused: true,
  }));
}

/**
 * Waits for a specific amount of time
 */
export function wait(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Creates a mock error
 */
export function createMockError(message = 'Test error') {
  return new Error(message);
}

/**
 * Creates a mock network error
 */
export function createMockNetworkError() {
  const error = new Error('Network Error');
  (error as any).code = 'NETWORK_ERROR';
  return error;
}

/**
 * Asserts that a function throws an error
 */
export async function expectToThrow(
  fn: () => Promise<any> | any,
  errorMessage?: string
) {
  try {
    await fn();
    throw new Error('Expected function to throw');
  } catch (error: any) {
    if (errorMessage) {
      expect(error.message).toContain(errorMessage);
    }
  }
}

