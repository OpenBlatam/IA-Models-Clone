import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TrackSearch } from '@/components/music/TrackSearch';
import { musicApiService } from '@/lib/api/music-api';

// Mock the API service
jest.mock('@/lib/api/music-api', () => ({
  musicApiService: {
    searchTracks: jest.fn(),
  },
}));

// Mock react-hot-toast
jest.mock('react-hot-toast', () => ({
  default: {
    success: jest.fn(),
    error: jest.fn(),
  },
}));

const mockSearchTracks = musicApiService.searchTracks as jest.MockedFunction<
  typeof musicApiService.searchTracks
>;

const mockTracks = [
  {
    id: '1',
    name: 'Test Track 1',
    artists: ['Artist 1'],
    preview_url: 'https://example.com/preview1.mp3',
    images: [{ url: 'https://example.com/image1.jpg' }],
    duration_ms: 200000,
    album: { name: 'Test Album 1' },
  },
  {
    id: '2',
    name: 'Test Track 2',
    artists: ['Artist 2'],
    preview_url: 'https://example.com/preview2.mp3',
    images: [{ url: 'https://example.com/image2.jpg' }],
    duration_ms: 180000,
    album: { name: 'Test Album 2' },
  },
];

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

describe('TrackSearch', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('should render search input', () => {
    const onTrackSelect = jest.fn();
    render(<TrackSearch onTrackSelect={onTrackSelect} />, {
      wrapper: createWrapper(),
    });

    expect(
      screen.getByPlaceholderText(/busca canciones, artistas o álbumes/i)
    ).toBeInTheDocument();
    expect(screen.getByText(/buscar canciones/i)).toBeInTheDocument();
  });

  it('should debounce search input', async () => {
    const user = userEvent.setup({ delay: null });
    const onTrackSelect = jest.fn();

    mockSearchTracks.mockResolvedValue({
      results: mockTracks,
      total: 2,
    });

    render(<TrackSearch onTrackSelect={onTrackSelect} />, {
      wrapper: createWrapper(),
    });

    const input = screen.getByPlaceholderText(
      /busca canciones, artistas o álbumes/i
    );

    await user.type(input, 'test');

    // Should not call API immediately
    expect(mockSearchTracks).not.toHaveBeenCalled();

    // Fast-forward debounce time
    jest.advanceTimersByTime(500);

    await waitFor(() => {
      expect(mockSearchTracks).toHaveBeenCalledWith('test', 10);
    });
  });

  it('should display loading state', async () => {
    const user = userEvent.setup({ delay: null });
    const onTrackSelect = jest.fn();

    mockSearchTracks.mockImplementation(
      () =>
        new Promise((resolve) => {
          setTimeout(
            () =>
              resolve({
                results: mockTracks,
                total: 2,
              }),
            100
          );
        })
    );

    render(<TrackSearch onTrackSelect={onTrackSelect} />, {
      wrapper: createWrapper(),
    });

    const input = screen.getByPlaceholderText(
      /busca canciones, artistas o álbumes/i
    );

    await user.type(input, 'test');
    jest.advanceTimersByTime(500);

    await waitFor(() => {
      expect(screen.getByRole('status')).toBeInTheDocument();
    });
  });

  it('should display search results', async () => {
    const user = userEvent.setup({ delay: null });
    const onTrackSelect = jest.fn();

    mockSearchTracks.mockResolvedValue({
      results: mockTracks,
      total: 2,
    });

    render(<TrackSearch onTrackSelect={onTrackSelect} />, {
      wrapper: createWrapper(),
    });

    const input = screen.getByPlaceholderText(
      /busca canciones, artistas o álbumes/i
    );

    await user.type(input, 'test');
    jest.advanceTimersByTime(500);

    await waitFor(() => {
      expect(screen.getByText('Test Track 1')).toBeInTheDocument();
      expect(screen.getByText('Test Track 2')).toBeInTheDocument();
    });
  });

  it('should call onTrackSelect when track is clicked', async () => {
    const user = userEvent.setup({ delay: null });
    const onTrackSelect = jest.fn();

    mockSearchTracks.mockResolvedValue({
      results: mockTracks,
      total: 2,
    });

    render(<TrackSearch onTrackSelect={onTrackSelect} />, {
      wrapper: createWrapper(),
    });

    const input = screen.getByPlaceholderText(
      /busca canciones, artistas o álbumes/i
    );

    await user.type(input, 'test');
    jest.advanceTimersByTime(500);

    await waitFor(() => {
      expect(screen.getByText('Test Track 1')).toBeInTheDocument();
    });

    const trackButton = screen.getByText('Test Track 1').closest('button');
    await user.click(trackButton!);

    expect(onTrackSelect).toHaveBeenCalledWith(mockTracks[0]);
  });

  it('should clear search after selecting track', async () => {
    const user = userEvent.setup({ delay: null });
    const onTrackSelect = jest.fn();

    mockSearchTracks.mockResolvedValue({
      results: mockTracks,
      total: 2,
    });

    render(<TrackSearch onTrackSelect={onTrackSelect} />, {
      wrapper: createWrapper(),
    });

    const input = screen.getByPlaceholderText(
      /busca canciones, artistas o álbumes/i
    ) as HTMLInputElement;

    await user.type(input, 'test');
    jest.advanceTimersByTime(500);

    await waitFor(() => {
      expect(screen.getByText('Test Track 1')).toBeInTheDocument();
    });

    const trackButton = screen.getByText('Test Track 1').closest('button');
    await user.click(trackButton!);

    expect(input.value).toBe('');
  });

  it('should call onSearchResults callback', async () => {
    const user = userEvent.setup({ delay: null });
    const onTrackSelect = jest.fn();
    const onSearchResults = jest.fn();

    mockSearchTracks.mockResolvedValue({
      results: mockTracks,
      total: 2,
    });

    render(
      <TrackSearch
        onTrackSelect={onTrackSelect}
        onSearchResults={onSearchResults}
      />,
      {
        wrapper: createWrapper(),
      }
    );

    const input = screen.getByPlaceholderText(
      /busca canciones, artistas o álbumes/i
    );

    await user.type(input, 'test');
    jest.advanceTimersByTime(500);

    await waitFor(() => {
      expect(onSearchResults).toHaveBeenCalledWith(mockTracks);
    });
  });

  it('should display error message on error', async () => {
    const user = userEvent.setup({ delay: null });
    const onTrackSelect = jest.fn();

    mockSearchTracks.mockRejectedValue(new Error('API Error'));

    render(<TrackSearch onTrackSelect={onTrackSelect} />, {
      wrapper: createWrapper(),
    });

    const input = screen.getByPlaceholderText(
      /busca canciones, artistas o álbumes/i
    );

    await user.type(input, 'test');
    jest.advanceTimersByTime(500);

    await waitFor(() => {
      expect(
        screen.getByText(/error al buscar canciones/i)
      ).toBeInTheDocument();
    });
  });

  it('should not search when query is empty', async () => {
    const user = userEvent.setup({ delay: null });
    const onTrackSelect = jest.fn();

    render(<TrackSearch onTrackSelect={onTrackSelect} />, {
      wrapper: createWrapper(),
    });

    const input = screen.getByPlaceholderText(
      /busca canciones, artistas o álbumes/i
    );

    await user.type(input, 'test');
    jest.advanceTimersByTime(500);

    await waitFor(() => {
      expect(mockSearchTracks).toHaveBeenCalled();
    });

    jest.clearAllMocks();

    await user.clear(input);
    jest.advanceTimersByTime(500);

    // Should not call API with empty query
    expect(mockSearchTracks).not.toHaveBeenCalled();
  });

  it('should display track images when available', async () => {
    const user = userEvent.setup({ delay: null });
    const onTrackSelect = jest.fn();

    mockSearchTracks.mockResolvedValue({
      results: mockTracks,
      total: 2,
    });

    render(<TrackSearch onTrackSelect={onTrackSelect} />, {
      wrapper: createWrapper(),
    });

    const input = screen.getByPlaceholderText(
      /busca canciones, artistas o álbumes/i
    );

    await user.type(input, 'test');
    jest.advanceTimersByTime(500);

    await waitFor(() => {
      const image = screen.getByAltText('Test Track 1');
      expect(image).toHaveAttribute(
        'src',
        'https://example.com/image1.jpg'
      );
    });
  });

  it('should display placeholder icon when track has no image', async () => {
    const user = userEvent.setup({ delay: null });
    const onTrackSelect = jest.fn();

    const tracksWithoutImages = [
      {
        ...mockTracks[0],
        images: [],
      },
    ];

    mockSearchTracks.mockResolvedValue({
      results: tracksWithoutImages,
      total: 1,
    });

    render(<TrackSearch onTrackSelect={onTrackSelect} />, {
      wrapper: createWrapper(),
    });

    const input = screen.getByPlaceholderText(
      /busca canciones, artistas o álbumes/i
    );

    await user.type(input, 'test');
    jest.advanceTimersByTime(500);

    await waitFor(() => {
      // Should show Music icon placeholder
      expect(screen.getByText('Test Track 1')).toBeInTheDocument();
    });
  });
});

