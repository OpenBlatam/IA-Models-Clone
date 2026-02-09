/**
 * Router Integration Tests
 * Tests for Next.js router integration with components and store
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useRouter, usePathname, useSearchParams } from 'next/navigation';
import { useMusicStore } from '@/lib/store/music-store';
import { type Track } from '@/lib/api/types';

// Mock Next.js router
const mockPush = jest.fn();
const mockReplace = jest.fn();
const mockBack = jest.fn();
const mockRefresh = jest.fn();

jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: mockPush,
    replace: mockReplace,
    back: mockBack,
    refresh: mockRefresh,
  }),
  usePathname: () => '/music',
  useSearchParams: () => new URLSearchParams(),
}));

// Component that uses router
const RouterComponent = () => {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  
  return (
    <div>
      <div data-testid="pathname">{pathname}</div>
      <button onClick={() => router.push('/music')} data-testid="push-music">
        Go to Music
      </button>
      <button onClick={() => router.push('/analyze')} data-testid="push-analyze">
        Go to Analyze
      </button>
      <button onClick={() => router.back()} data-testid="back">
        Back
      </button>
      <button onClick={() => router.refresh()} data-testid="refresh">
        Refresh
      </button>
    </div>
  );
};

// Component that uses router and store
const RouterStoreComponent = () => {
  const router = useRouter();
  const { setCurrentTrack, currentTrack } = useMusicStore();
  
  const handleNavigateWithTrack = (track: Track) => {
    setCurrentTrack(track);
    router.push('/music');
  };
  
  return (
    <div>
      {currentTrack && (
        <div data-testid="current-track">{currentTrack.name}</div>
      )}
      <button
        onClick={() => handleNavigateWithTrack({
          id: '1',
          name: 'Test Track',
          artists: ['Artist'],
          duration_ms: 200000,
          preview_url: 'https://example.com/preview.mp3',
          images: [],
        })}
        data-testid="navigate-with-track"
      >
        Navigate with Track
      </button>
    </div>
  );
};

// Component that uses search params
const SearchParamsComponent = () => {
  const searchParams = useSearchParams();
  const query = searchParams.get('q') || '';
  
  return (
    <div>
      <div data-testid="query">{query}</div>
    </div>
  );
};

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, cacheTime: 0 },
    },
  });
  
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
};

describe('Router Integration', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    useMusicStore.setState({
      currentTrack: null,
      playlistQueue: [],
      isPlaying: false,
    });
  });

  describe('Basic Router Navigation', () => {
    it('should navigate to different routes', async () => {
      const user = userEvent.setup();
      const Wrapper = createWrapper();
      
      render(
        <Wrapper>
          <RouterComponent />
        </Wrapper>
      );
      
      const pushMusicButton = screen.getByTestId('push-music');
      await user.click(pushMusicButton);
      
      expect(mockPush).toHaveBeenCalledWith('/music');
    });

    it('should navigate to analyze route', async () => {
      const user = userEvent.setup();
      const Wrapper = createWrapper();
      
      render(
        <Wrapper>
          <RouterComponent />
        </Wrapper>
      );
      
      const pushAnalyzeButton = screen.getByTestId('push-analyze');
      await user.click(pushAnalyzeButton);
      
      expect(mockPush).toHaveBeenCalledWith('/analyze');
    });

    it('should handle back navigation', async () => {
      const user = userEvent.setup();
      const Wrapper = createWrapper();
      
      render(
        <Wrapper>
          <RouterComponent />
        </Wrapper>
      );
      
      const backButton = screen.getByTestId('back');
      await user.click(backButton);
      
      expect(mockBack).toHaveBeenCalled();
    });

    it('should handle refresh', async () => {
      const user = userEvent.setup();
      const Wrapper = createWrapper();
      
      render(
        <Wrapper>
          <RouterComponent />
        </Wrapper>
      );
      
      const refreshButton = screen.getByTestId('refresh');
      await user.click(refreshButton);
      
      expect(mockRefresh).toHaveBeenCalled();
    });

    it('should display current pathname', () => {
      const Wrapper = createWrapper();
      
      render(
        <Wrapper>
          <RouterComponent />
        </Wrapper>
      );
      
      expect(screen.getByTestId('pathname')).toHaveTextContent('/music');
    });
  });

  describe('Router with Store Integration', () => {
    it('should navigate with track in store', async () => {
      const user = userEvent.setup();
      const Wrapper = createWrapper();
      
      render(
        <Wrapper>
          <RouterStoreComponent />
        </Wrapper>
      );
      
      const navigateButton = screen.getByTestId('navigate-with-track');
      await user.click(navigateButton);
      
      expect(mockPush).toHaveBeenCalledWith('/music');
      
      await waitFor(() => {
        const { currentTrack } = useMusicStore.getState();
        expect(currentTrack).toBeTruthy();
        expect(currentTrack?.name).toBe('Test Track');
      });
    });

    it('should preserve track when navigating', async () => {
      const user = userEvent.setup();
      const Wrapper = createWrapper();
      
      // Set track first
      const track: Track = {
        id: '1',
        name: 'Preserved Track',
        artists: ['Artist'],
        duration_ms: 200000,
        preview_url: 'https://example.com/preview.mp3',
        images: [],
      };
      
      useMusicStore.setState({ currentTrack: track });
      
      render(
        <Wrapper>
          <RouterStoreComponent />
        </Wrapper>
      );
      
      expect(screen.getByTestId('current-track')).toHaveTextContent('Preserved Track');
      
      const navigateButton = screen.getByTestId('navigate-with-track');
      await user.click(navigateButton);
      
      // Track should still be in store
      const { currentTrack: updatedTrack } = useMusicStore.getState();
      expect(updatedTrack).toBeTruthy();
    });
  });

  describe('Search Params Integration', () => {
    it('should read search params', () => {
      const Wrapper = createWrapper();
      
      // Mock search params with query
      jest.spyOn(require('next/navigation'), 'useSearchParams').mockReturnValue(
        new URLSearchParams('q=test')
      );
      
      render(
        <Wrapper>
          <SearchParamsComponent />
        </Wrapper>
      );
      
      expect(screen.getByTestId('query')).toHaveTextContent('test');
    });

    it('should handle empty search params', () => {
      const Wrapper = createWrapper();
      
      render(
        <Wrapper>
          <SearchParamsComponent />
        </Wrapper>
      );
      
      expect(screen.getByTestId('query')).toHaveTextContent('');
    });
  });

  describe('Router Error Handling', () => {
    it('should handle navigation errors gracefully', async () => {
      const user = userEvent.setup();
      mockPush.mockImplementation(() => {
        throw new Error('Navigation error');
      });
      
      const Wrapper = createWrapper();
      
      render(
        <Wrapper>
          <RouterComponent />
        </Wrapper>
      );
      
      const pushButton = screen.getByTestId('push-music');
      
      // Should not crash
      await expect(user.click(pushButton)).resolves.not.toThrow();
    });
  });

  describe('Router State Persistence', () => {
    it('should maintain router state across re-renders', () => {
      const Wrapper = createWrapper();
      
      const { rerender } = render(
        <Wrapper>
          <RouterComponent />
        </Wrapper>
      );
      
      const pathname1 = screen.getByTestId('pathname').textContent;
      
      rerender(
        <Wrapper>
          <RouterComponent />
        </Wrapper>
      );
      
      const pathname2 = screen.getByTestId('pathname').textContent;
      
      expect(pathname1).toBe(pathname2);
    });
  });
});

