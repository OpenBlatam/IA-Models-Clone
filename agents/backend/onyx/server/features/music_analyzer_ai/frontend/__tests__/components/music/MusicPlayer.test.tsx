import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MusicPlayer } from '@/components/music/MusicPlayer';
import { type Track } from '@/lib/api/music-api';

const createTestQueryClient = () => new QueryClient({
  defaultOptions: {
    queries: { retry: false },
  },
});

const renderWithQueryClient = (component: React.ReactElement) => {
  const queryClient = createTestQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      {component}
    </QueryClientProvider>
  );
};

describe('MusicPlayer', () => {
  const mockTrack: Track = {
    id: '1',
    name: 'Test Track',
    artists: ['Test Artist'],
    album: 'Test Album',
    duration_ms: 200000,
    popularity: 80,
    preview_url: 'https://example.com/preview.mp3',
  };

  const mockQueue: Track[] = [mockTrack];

  const mockHandlers = {
    onNext: jest.fn(),
    onPrevious: jest.fn(),
    onShuffle: jest.fn(),
    onRepeat: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders player controls', () => {
    renderWithQueryClient(
      <MusicPlayer
        track={mockTrack}
        queue={mockQueue}
        currentIndex={0}
        {...mockHandlers}
      />
    );

    expect(screen.getByRole('button', { name: /play|pause/i })).toBeInTheDocument();
  });

  it('displays track information', () => {
    renderWithQueryClient(
      <MusicPlayer
        track={mockTrack}
        queue={mockQueue}
        currentIndex={0}
        {...mockHandlers}
      />
    );

    expect(screen.getByText('Test Track')).toBeInTheDocument();
    expect(screen.getByText('Test Artist')).toBeInTheDocument();
  });

  it('handles play/pause button click', () => {
    renderWithQueryClient(
      <MusicPlayer
        track={mockTrack}
        queue={mockQueue}
        currentIndex={0}
        {...mockHandlers}
      />
    );

    const playButton = screen.getByRole('button', { name: /play|pause/i });
    fireEvent.click(playButton);
    // The component should toggle play/pause state
  });

  it('handles next button click', () => {
    renderWithQueryClient(
      <MusicPlayer
        track={mockTrack}
        queue={mockQueue}
        currentIndex={0}
        {...mockHandlers}
      />
    );

    const nextButton = screen.getByRole('button', { name: /next|skip/i });
    fireEvent.click(nextButton);
    expect(mockHandlers.onNext).toHaveBeenCalled();
  });

  it('handles previous button click', () => {
    renderWithQueryClient(
      <MusicPlayer
        track={mockTrack}
        queue={mockQueue}
        currentIndex={0}
        {...mockHandlers}
      />
    );

    const prevButton = screen.getByRole('button', { name: /previous|back/i });
    fireEvent.click(prevButton);
    expect(mockHandlers.onPrevious).toHaveBeenCalled();
  });

  it('handles shuffle button click', () => {
    renderWithQueryClient(
      <MusicPlayer
        track={mockTrack}
        queue={mockQueue}
        currentIndex={0}
        {...mockHandlers}
      />
    );

    const shuffleButton = screen.getByRole('button', { name: /shuffle/i });
    fireEvent.click(shuffleButton);
    expect(mockHandlers.onShuffle).toHaveBeenCalled();
  });

  it('handles repeat button click', () => {
    renderWithQueryClient(
      <MusicPlayer
        track={mockTrack}
        queue={mockQueue}
        currentIndex={0}
        {...mockHandlers}
      />
    );

    const repeatButton = screen.getByRole('button', { name: /repeat/i });
    fireEvent.click(repeatButton);
    expect(mockHandlers.onRepeat).toHaveBeenCalled();
  });

  it('displays empty state when no track', () => {
    renderWithQueryClient(
      <MusicPlayer
        track={null}
        queue={[]}
        currentIndex={0}
        {...mockHandlers}
      />
    );

    // Player should still render but may show empty state
    expect(screen.queryByText('Test Track')).not.toBeInTheDocument();
  });
});

