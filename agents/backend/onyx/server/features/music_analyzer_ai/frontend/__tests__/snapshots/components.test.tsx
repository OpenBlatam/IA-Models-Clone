/**
 * Snapshot Tests
 * Tests to ensure UI components don't change unexpectedly
 */

import { render } from '@testing-library/react';
import { Navigation } from '@/components/Navigation';
import { ApiStatus } from '@/components/api-status';
import { LoadingSkeleton, TrackCardSkeleton } from '@/components/music/LoadingSkeleton';
import { StatsCard } from '@/components/music/StatsCard';
import { ProgressIndicator } from '@/components/music/ProgressIndicator';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Music } from 'lucide-react';

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        cacheTime: 0,
      },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('Component Snapshots', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Navigation', () => {
    it('should match snapshot', () => {
      const { container } = render(<Navigation />);
      expect(container.firstChild).toMatchSnapshot();
    });
  });

  describe('LoadingSkeleton', () => {
    it('should match snapshot', () => {
      const { container } = render(<LoadingSkeleton />);
      expect(container.firstChild).toMatchSnapshot();
    });
  });

  describe('TrackCardSkeleton', () => {
    it('should match snapshot', () => {
      const { container } = render(<TrackCardSkeleton />);
      expect(container.firstChild).toMatchSnapshot();
    });
  });

  describe('StatsCard', () => {
    it('should match snapshot with all props', () => {
      const { container } = render(
        <StatsCard
          title="Test Stat"
          value="100"
          change={10}
          trend="up"
          icon={<Music />}
        />
      );
      expect(container.firstChild).toMatchSnapshot();
    });

    it('should match snapshot with minimal props', () => {
      const { container } = render(<StatsCard title="Test" value="50" />);
      expect(container.firstChild).toMatchSnapshot();
    });
  });

  describe('ProgressIndicator', () => {
    it('should match snapshot with multiple steps', () => {
      const steps = [
        { id: '1', label: 'Step 1', status: 'completed' as const },
        { id: '2', label: 'Step 2', status: 'loading' as const },
        { id: '3', label: 'Step 3', status: 'pending' as const },
      ];

      const { container } = render(<ProgressIndicator steps={steps} />);
      expect(container.firstChild).toMatchSnapshot();
    });
  });

  describe('ApiStatus', () => {
    it('should match snapshot when healthy', async () => {
      jest.mock('@/lib/api/client', () => ({
        checkApiHealth: jest.fn().mockResolvedValue({
          status: 'healthy',
          message: 'API is reachable',
          timestamp: Date.now(),
        }),
      }));

      const { container } = render(<ApiStatus />, { wrapper: createWrapper() });
      expect(container.firstChild).toMatchSnapshot();
    });
  });
});

