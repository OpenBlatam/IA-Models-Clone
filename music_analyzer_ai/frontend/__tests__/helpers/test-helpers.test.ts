/**
 * Test Helpers
 * Utility functions for testing
 */

import { render, RenderOptions } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactElement } from 'react';

/**
 * Creates a QueryClient wrapper for tests
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
 * Renders component with QueryClient provider
 */
export function renderWithQueryClient(
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
) {
  const queryClient = createTestQueryClient();
  const Wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

  return render(ui, { wrapper: Wrapper, ...options });
}

/**
 * Waits for async operations to complete
 */
export async function waitForAsync() {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Creates mock track for testing
 */
export function createMockTrack(overrides = {}) {
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
 * Creates mock API response
 */
export function createMockApiResponse<T>(data: T, success = true) {
  return {
    success,
    ...(success ? { data } : { error: 'Error message' }),
  };
}

describe('Test Helpers', () => {
  describe('createTestQueryClient', () => {
    it('should create QueryClient with test defaults', () => {
      const client = createTestQueryClient();
      expect(client).toBeInstanceOf(QueryClient);
    });
  });

  describe('createMockTrack', () => {
    it('should create mock track with defaults', () => {
      const track = createMockTrack();
      expect(track.id).toBe('track-123');
      expect(track.name).toBe('Test Track');
    });

    it('should allow overrides', () => {
      const track = createMockTrack({ name: 'Custom Track' });
      expect(track.name).toBe('Custom Track');
      expect(track.id).toBe('track-123');
    });
  });

  describe('createMockApiResponse', () => {
    it('should create success response', () => {
      const response = createMockApiResponse({ data: 'test' });
      expect(response.success).toBe(true);
      expect(response.data).toEqual({ data: 'test' });
    });

    it('should create error response', () => {
      const response = createMockApiResponse(null, false);
      expect(response.success).toBe(false);
      expect(response.error).toBeDefined();
    });
  });
});

