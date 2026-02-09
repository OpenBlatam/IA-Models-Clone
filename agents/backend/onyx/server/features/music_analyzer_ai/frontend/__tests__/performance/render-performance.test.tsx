/**
 * Render Performance Tests
 * Tests to ensure components render efficiently
 */

import { render } from '@testing-library/react';
import { Navigation } from '@/components/Navigation';
import { TrackSearch } from '@/components/music/TrackSearch';
import { AudioPlayer } from '@/components/music/AudioPlayer';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { type Track } from '@/lib/api/music-api';

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, cacheTime: 0 },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

const mockTrack: Track = {
  id: 'track-123',
  name: 'Test Track',
  artists: ['Test Artist'],
  album: 'Test Album',
  duration_ms: 200000,
  preview_url: 'https://example.com/preview.mp3',
  popularity: 80,
  images: [{ url: 'https://example.com/image.jpg' }],
};

describe('Render Performance', () => {
  describe('Component Render Time', () => {
    it('should render Navigation quickly', () => {
      const start = performance.now();
      render(<Navigation />);
      const end = performance.now();
      const duration = end - start;

      // Should render in less than 100ms
      expect(duration).toBeLessThan(100);
    });

    it('should render TrackSearch quickly', () => {
      const start = performance.now();
      render(<TrackSearch onTrackSelect={() => {}} />, {
        wrapper: createWrapper(),
      });
      const end = performance.now();
      const duration = end - start;

      expect(duration).toBeLessThan(100);
    });

    it('should render AudioPlayer quickly', () => {
      const start = performance.now();
      render(<AudioPlayer track={mockTrack} />);
      const end = performance.now();
      const duration = end - start;

      expect(duration).toBeLessThan(100);
    });
  });

  describe('Multiple Renders', () => {
    it('should handle multiple component renders efficiently', () => {
      const start = performance.now();

      for (let i = 0; i < 100; i++) {
        render(<Navigation />);
      }

      const end = performance.now();
      const duration = end - start;

      // 100 renders should complete in less than 1 second
      expect(duration).toBeLessThan(1000);
    });
  });

  describe('Re-render Performance', () => {
    it('should re-render efficiently', () => {
      const { rerender } = render(<Navigation />);
      const start = performance.now();

      for (let i = 0; i < 50; i++) {
        rerender(<Navigation />);
      }

      const end = performance.now();
      const duration = end - start;

      // 50 re-renders should complete quickly
      expect(duration).toBeLessThan(500);
    });
  });
});

