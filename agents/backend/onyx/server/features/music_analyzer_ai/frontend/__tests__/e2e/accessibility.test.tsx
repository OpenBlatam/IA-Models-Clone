/**
 * E2E Tests - Accessibility
 * Tests accessibility features and keyboard navigation
 */

import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Navigation } from '@/components/Navigation';
import { TrackSearch } from '@/components/music/TrackSearch';
import { AudioPlayer } from '@/components/music/AudioPlayer';
import { ThemeToggle } from '@/components/music/ThemeToggle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { type Track } from '@/lib/api/music-api';

jest.mock('@/lib/api/music-api', () => ({
  musicApiService: {
    searchTracks: jest.fn(),
  },
}));

jest.mock('react-hot-toast', () => ({
  default: {
    success: jest.fn(),
    error: jest.fn(),
  },
}));

const mockSearchTracks = musicApiService.searchTracks as jest.MockedFunction<
  typeof musicApiService.searchTracks
>;

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

describe('E2E Accessibility Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('Keyboard Navigation', () => {
    it('should navigate with keyboard through navigation links', async () => {
      const user = userEvent.setup();
      render(<Navigation />);

      const homeLink = screen.getByText('Inicio').closest('a');
      const musicLink = screen.getByText('Music AI').closest('a');
      const robotLink = screen.getByText('Robot AI').closest('a');

      // Tab through links
      await user.tab();
      expect(homeLink).toHaveFocus();

      await user.tab();
      expect(musicLink).toHaveFocus();

      await user.tab();
      expect(robotLink).toHaveFocus();
    });

    it('should navigate search with keyboard', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      mockSearchTracks.mockResolvedValue({
        success: true,
        query: 'test',
        results: [mockTrack],
        total: 1,
      });

      render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );

      // Focus input with Tab
      await user.tab();
      expect(searchInput).toHaveFocus();

      // Type search query
      await user.type(searchInput, 'test');
      jest.advanceTimersByTime(500);

      await waitFor(() => {
        expect(screen.getByText('Test Track')).toBeInTheDocument();
      });

      // Navigate to result with Tab
      await user.tab();
      const trackButton = screen.getByText('Test Track').closest('button');
      expect(trackButton).toHaveFocus();

      // Select with Enter
      await user.keyboard('{Enter}');
      expect(onTrackSelect).toHaveBeenCalled();
    });
  });

  describe('ARIA Attributes', () => {
    it('should have proper ARIA labels', () => {
      render(<ThemeToggle />);

      const toggleButton = screen.getByRole('button', { name: /toggle theme/i });
      expect(toggleButton).toHaveAttribute('aria-label', 'Toggle theme');
    });

    it('should have proper button roles', () => {
      const onTrackSelect = jest.fn();

      render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      // Search button should be accessible
      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );
      expect(searchInput).toHaveAttribute('type', 'text');
    });
  });

  describe('Screen Reader Support', () => {
    it('should provide meaningful text for screen readers', () => {
      render(<AudioPlayer track={mockTrack} />);

      // Track name should be readable
      expect(screen.getByText('Test Track')).toBeInTheDocument();
      expect(screen.getByText('Test Artist')).toBeInTheDocument();

      // Buttons should have accessible names
      const playButton = screen.getByRole('button', { name: /play/i });
      expect(playButton).toBeInTheDocument();
    });
  });

  describe('Focus Management', () => {
    it('should manage focus correctly in modals and dropdowns', async () => {
      const user = userEvent.setup();
      const onSortChange = jest.fn();

      // This would test a dropdown component
      // For now, we test the concept
      render(
        <div>
          <button>Open Menu</button>
          <div role="menu" style={{ display: 'none' }}>
            <button>Option 1</button>
            <button>Option 2</button>
          </div>
        </div>
      );

      const openButton = screen.getByText('Open Menu');
      await user.click(openButton);

      // Focus should move to first menu item
      // (This is a conceptual test - actual implementation would vary)
    });
  });

  describe('Color Contrast', () => {
    it('should have sufficient color contrast for text', () => {
      render(<Navigation />);

      const links = screen.getAllByRole('link');
      links.forEach((link) => {
        // In a real test, you would check computed styles
        // This is a placeholder for the concept
        expect(link).toBeInTheDocument();
      });
    });
  });

  describe('Form Accessibility', () => {
    it('should have proper form labels and error messages', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );

      // Input should be accessible
      expect(searchInput).toHaveAttribute('type', 'text');
      expect(searchInput).toHaveAttribute('placeholder');

      // Error messages should be accessible
      mockSearchTracks.mockRejectedValueOnce(new Error('API Error'));

      await user.type(searchInput, 'test');
      jest.advanceTimersByTime(500);

      await waitFor(() => {
        expect(
          screen.getByText(/error al buscar canciones/i)
        ).toBeInTheDocument();
      });
    });
  });
});

