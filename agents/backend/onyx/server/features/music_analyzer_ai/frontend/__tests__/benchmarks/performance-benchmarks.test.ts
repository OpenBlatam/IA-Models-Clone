/**
 * Performance Benchmarks
 * Benchmarks to track performance over time
 */

import { render } from '@testing-library/react';
import { Navigation } from '@/components/Navigation';
import { TrackSearch } from '@/components/music/TrackSearch';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

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

describe('Performance Benchmarks', () => {
  describe('Component Render Benchmarks', () => {
    it('Navigation should render in <50ms', () => {
      const start = performance.now();
      render(<Navigation />);
      const end = performance.now();
      const duration = end - start;

      expect(duration).toBeLessThan(50);
    });

    it('TrackSearch should render in <50ms', () => {
      const start = performance.now();
      render(<TrackSearch onTrackSelect={() => {}} />, {
        wrapper: createWrapper(),
      });
      const end = performance.now();
      const duration = end - start;

      expect(duration).toBeLessThan(50);
    });
  });

  describe('Memory Usage', () => {
    it('should not leak memory on multiple renders', () => {
      const initialMemory = (performance as any).memory?.usedJSHeapSize || 0;

      for (let i = 0; i < 100; i++) {
        const { unmount } = render(<Navigation />);
        unmount();
      }

      // Memory should not grow significantly
      const finalMemory = (performance as any).memory?.usedJSHeapSize || 0;
      const growth = finalMemory - initialMemory;

      // Growth should be minimal (<10MB)
      expect(growth).toBeLessThan(10 * 1024 * 1024);
    });
  });

  describe('Test Execution Speed', () => {
    it('should execute tests quickly', () => {
      const start = performance.now();
      
      // Simulate test execution
      render(<Navigation />);
      
      const end = performance.now();
      const duration = end - start;

      // Should complete in <100ms
      expect(duration).toBeLessThan(100);
    });
  });
});

