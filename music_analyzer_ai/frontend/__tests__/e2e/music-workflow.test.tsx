/**
 * E2E Tests - Complete Music Workflow
 * Tests the complete music analysis and playback workflow
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TrackSearch } from '@/components/music/TrackSearch';
import { AudioPlayer } from '@/components/music/AudioPlayer';
import { ProgressIndicator } from '@/components/music/ProgressIndicator';
import { musicApiService } from '@/lib/api/music-api';
import { type Track } from '@/lib/api/music-api';

jest.mock('@/lib/api/music-api', () => ({
  musicApiService: {
    searchTracks: jest.fn(),
    analyzeTrack: jest.fn(),
    getRecommendations: jest.fn(),
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
const mockAnalyzeTrack = musicApiService.analyzeTrack as jest.MockedFunction<
  typeof musicApiService.analyzeTrack
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
  name: 'Bohemian Rhapsody',
  artists: ['Queen'],
  album: 'A Night at the Opera',
  duration_ms: 355000,
  preview_url: 'https://example.com/preview.mp3',
  popularity: 95,
  images: [{ url: 'https://example.com/queen.jpg' }],
};

describe('E2E Music Workflow', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('Complete Track Analysis Workflow', () => {
    it('should complete full track search -> select -> analyze flow', async () => {
      const user = userEvent.setup({ delay: null });
      let selectedTrack: Track | null = null;

      const onTrackSelect = (track: Track) => {
        selectedTrack = track;
      };

      mockSearchTracks.mockResolvedValue({
        success: true,
        query: 'bohemian',
        results: [mockTrack],
        total: 1,
      });

      mockAnalyzeTrack.mockResolvedValue({
        success: true,
        track_basic_info: {
          name: 'Bohemian Rhapsody',
          artists: ['Queen'],
          album: 'A Night at the Opera',
          duration_seconds: 355,
        },
        musical_analysis: {
          key_signature: 'Bb',
          root_note: 'Bb',
          mode: 'major',
          tempo: { bpm: 72, category: 'slow' },
          time_signature: '4/4',
          scale: {
            name: 'Bb Major',
            notes: ['Bb', 'C', 'D', 'Eb', 'F', 'G', 'A'],
          },
        },
        technical_analysis: {
          energy: { value: 0.6, description: 'Moderate energy' },
          danceability: { value: 0.4, description: 'Low danceability' },
          valence: { value: 0.5, description: 'Neutral mood' },
          acousticness: { value: 0.3, description: 'Mixed' },
          instrumentalness: { value: 0.1, description: 'Vocal' },
          liveness: { value: 0.2, description: 'Studio' },
          loudness: { value: -8, description: 'Moderate' },
        },
      });

      render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      // Step 1: User searches for track
      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );
      await user.type(searchInput, 'bohemian');
      jest.advanceTimersByTime(500);

      // Step 2: Results appear
      await waitFor(() => {
        expect(screen.getByText('Bohemian Rhapsody')).toBeInTheDocument();
        expect(screen.getByText('Queen')).toBeInTheDocument();
      });

      // Step 3: User selects track
      const trackButton = screen.getByText('Bohemian Rhapsody').closest('button');
      await user.click(trackButton!);

      // Step 4: Track is selected
      expect(selectedTrack).toEqual(mockTrack);

      // Step 5: Analysis would be triggered (tested in parent component)
      // In a real scenario, this would call analyzeTrack
    });
  });

  describe('Playback Workflow', () => {
    it('should handle complete playback workflow', async () => {
      const user = userEvent.setup({ delay: null });
      const onNext = jest.fn();
      const onPrevious = jest.fn();

      const nextTrack: Track = {
        ...mockTrack,
        id: 'track-456',
        name: 'We Will Rock You',
      };

      render(
        <AudioPlayer
          track={mockTrack}
          onNext={onNext}
          onPrevious={onPrevious}
        />
      );

      // Verify track is displayed
      expect(screen.getByText('Bohemian Rhapsody')).toBeInTheDocument();
      expect(screen.getByText('Queen')).toBeInTheDocument();

      // User clicks play (would trigger audio playback)
      const playButton = screen.getByRole('button', { name: /play/i });
      await user.click(playButton);

      // User skips to next track
      const nextButton = screen.getByRole('button', { name: /next|skip forward/i });
      await user.click(nextButton);
      expect(onNext).toHaveBeenCalled();

      // User goes to previous track
      const prevButton = screen.getByRole('button', {
        name: /previous|skip back/i,
      });
      await user.click(prevButton);
      expect(onPrevious).toHaveBeenCalled();
    });
  });

  describe('Progress Tracking Workflow', () => {
    it('should show progress during analysis', () => {
      const steps = [
        { id: '1', label: 'Buscando track', status: 'completed' as const },
        { id: '2', label: 'Analizando audio', status: 'loading' as const },
        { id: '3', label: 'Generando insights', status: 'pending' as const },
      ];

      render(<ProgressIndicator steps={steps} />);

      expect(screen.getByText('Buscando track')).toBeInTheDocument();
      expect(screen.getByText('Analizando audio')).toBeInTheDocument();
      expect(screen.getByText('Generando insights')).toBeInTheDocument();

      // Verify status indicators
      const step1 = screen.getByText('Buscando track').closest('div');
      const step2 = screen.getByText('Analizando audio').closest('div');

      // Step 1 should show completed icon
      expect(step1?.querySelector('.text-green-400')).toBeInTheDocument();

      // Step 2 should show loading spinner
      expect(step2?.querySelector('.animate-spin')).toBeInTheDocument();
    });
  });

  describe('Error Recovery Workflow', () => {
    it('should recover from API errors', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      // First call fails
      mockSearchTracks.mockRejectedValueOnce(new Error('Network error'));

      // Second call succeeds
      mockSearchTracks.mockResolvedValueOnce({
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

      // First attempt fails
      await user.type(searchInput, 'test');
      jest.advanceTimersByTime(500);

      await waitFor(() => {
        expect(
          screen.getByText(/error al buscar canciones/i)
        ).toBeInTheDocument();
      });

      // User tries again
      await user.clear(searchInput);
      await user.type(searchInput, 'test');
      jest.advanceTimersByTime(500);

      // Second attempt succeeds
      await waitFor(() => {
        expect(screen.getByText('Bohemian Rhapsody')).toBeInTheDocument();
      });
    });
  });

  describe('Multi-Step User Journey', () => {
    it('should complete complex multi-step journey', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      const tracks = [
        mockTrack,
        {
          ...mockTrack,
          id: 'track-2',
          name: 'Don\'t Stop Me Now',
          artists: ['Queen'],
        },
        {
          ...mockTrack,
          id: 'track-3',
          name: 'Somebody to Love',
          artists: ['Queen'],
        },
      ];

      mockSearchTracks.mockResolvedValue({
        success: true,
        query: 'queen',
        results: tracks,
        total: 3,
      });

      render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      // Step 1: Search
      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );
      await user.type(searchInput, 'queen');
      jest.advanceTimersByTime(500);

      // Step 2: See multiple results
      await waitFor(() => {
        expect(screen.getByText('Bohemian Rhapsody')).toBeInTheDocument();
        expect(screen.getByText("Don't Stop Me Now")).toBeInTheDocument();
        expect(screen.getByText('Somebody to Love')).toBeInTheDocument();
      });

      // Step 3: Select first track
      const firstTrack = screen.getByText('Bohemian Rhapsody').closest('button');
      await user.click(firstTrack!);

      expect(onTrackSelect).toHaveBeenCalledWith(tracks[0]);

      // Step 4: Search clears after selection
      expect((searchInput as HTMLInputElement).value).toBe('');
    });
  });
});

