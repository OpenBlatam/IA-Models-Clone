import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TrackSearch } from '@/components/music/TrackSearch';
import { musicApiService } from '@/lib/api/music-api';

// Mock the API service
jest.mock('@/lib/api/music-api', () => ({
  musicApiService: {
    searchTracks: jest.fn(),
  },
}));

const mockedMusicApiService = musicApiService as jest.Mocked<typeof musicApiService>;

// Create a test query client
const createTestQueryClient = () => new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
    },
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

describe('TrackSearch', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders search input', () => {
    renderWithQueryClient(<TrackSearch onTrackSelect={() => {}} />);
    
    const searchInput = screen.getByPlaceholderText(/buscar|search/i);
    expect(searchInput).toBeInTheDocument();
  });

  it('handles search input change', () => {
    renderWithQueryClient(<TrackSearch onTrackSelect={() => {}} />);
    
    const searchInput = screen.getByPlaceholderText(/buscar|search/i);
    fireEvent.change(searchInput, { target: { value: 'test query' } });
    
    expect(searchInput).toHaveValue('test query');
  });

  it('calls API on search', async () => {
    const mockResults = {
      success: true,
      query: 'test',
      results: [
        {
          id: '1',
          name: 'Test Track',
          artists: ['Test Artist'],
          album: 'Test Album',
          duration_ms: 200000,
          popularity: 80,
        },
      ],
      total: 1,
    };

    mockedMusicApiService.searchTracks.mockResolvedValue(mockResults);

    renderWithQueryClient(<TrackSearch onTrackSelect={() => {}} />);
    
    const searchInput = screen.getByPlaceholderText(/buscar|search/i);
    fireEvent.change(searchInput, { target: { value: 'test' } });
    
    // Wait for debounce (500ms) and query execution
    await waitFor(() => {
      expect(mockedMusicApiService.searchTracks).toHaveBeenCalledWith('test', 10);
    }, { timeout: 2000 });
  });

  it('displays search results', async () => {
    const mockResults = {
      success: true,
      query: 'test',
      results: [
        {
          id: '1',
          name: 'Test Track',
          artists: ['Test Artist'],
          album: 'Test Album',
          duration_ms: 200000,
          popularity: 80,
        },
      ],
      total: 1,
    };

    mockedMusicApiService.searchTracks.mockResolvedValue(mockResults);

    renderWithQueryClient(<TrackSearch onTrackSelect={() => {}} />);
    
    const searchInput = screen.getByPlaceholderText(/buscar|search/i);
    fireEvent.change(searchInput, { target: { value: 'test' } });
    
    await waitFor(() => {
      expect(screen.getByText('Test Track')).toBeInTheDocument();
    }, { timeout: 2000 });
  });

  it('calls onTrackSelect when track is clicked', async () => {
    const handleTrackSelect = jest.fn();
    const mockResults = {
      success: true,
      query: 'test',
      results: [
        {
          id: '1',
          name: 'Test Track',
          artists: ['Test Artist'],
          album: 'Test Album',
          duration_ms: 200000,
          popularity: 80,
        },
      ],
      total: 1,
    };

    mockedMusicApiService.searchTracks.mockResolvedValue(mockResults);

    renderWithQueryClient(<TrackSearch onTrackSelect={handleTrackSelect} />);
    
    const searchInput = screen.getByPlaceholderText(/buscar|search/i);
    fireEvent.change(searchInput, { target: { value: 'test' } });
    
    await waitFor(() => {
      const trackElement = screen.getByText('Test Track');
      fireEvent.click(trackElement);
      expect(handleTrackSelect).toHaveBeenCalled();
    }, { timeout: 2000 });
  });
});

