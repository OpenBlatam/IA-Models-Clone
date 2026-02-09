/**
 * API-Component Integration Tests
 * Tests for integration between API calls and React components
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';

// Mock the API service
jest.mock('@/lib/api/music-api', () => ({
  musicApiService: {
    searchTracks: jest.fn(),
    analyzeTrack: jest.fn(),
    getTrackInfo: jest.fn(),
  },
}));

const SearchComponent = () => {
  const [query, setQuery] = React.useState('');
  const [results, setResults] = React.useState([]);
  const [loading, setLoading] = React.useState(false);

  const handleSearch = async () => {
    setLoading(true);
    try {
      const data = await musicApiService.searchTracks(query);
      setResults(data.tracks || []);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        data-testid="search-input"
      />
      <button onClick={handleSearch} data-testid="search-button">
        Search
      </button>
      {loading && <div data-testid="loading">Loading...</div>}
      <div data-testid="results">
        {results.map((track: any) => (
          <div key={track.id} data-testid={`track-${track.id}`}>
            {track.name}
          </div>
        ))}
      </div>
    </div>
  );
};

describe('API-Component Integration', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    jest.clearAllMocks();
  });

  it('should fetch and display search results', async () => {
    const mockTracks = [
      { id: '1', name: 'Track 1' },
      { id: '2', name: 'Track 2' },
    ];

    (musicApiService.searchTracks as jest.Mock).mockResolvedValue({
      tracks: mockTracks,
    });

    const user = userEvent.setup();
    render(
      <QueryClientProvider client={queryClient}>
        <SearchComponent />
      </QueryClientProvider>
    );

    const input = screen.getByTestId('search-input');
    const button = screen.getByTestId('search-button');

    await user.type(input, 'test query');
    await user.click(button);

    await waitFor(() => {
      expect(screen.getByTestId('loading')).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByTestId('track-1')).toHaveTextContent('Track 1');
      expect(screen.getByTestId('track-2')).toHaveTextContent('Track 2');
    });

    expect(musicApiService.searchTracks).toHaveBeenCalledWith('test query');
  });

  it('should handle API errors gracefully', async () => {
    (musicApiService.searchTracks as jest.Mock).mockRejectedValue(
      new Error('API Error')
    );

    const user = userEvent.setup();
    render(
      <QueryClientProvider client={queryClient}>
        <SearchComponent />
      </QueryClientProvider>
    );

    const input = screen.getByTestId('search-input');
    const button = screen.getByTestId('search-button');

    await user.type(input, 'test');
    await user.click(button);

    await waitFor(() => {
      expect(screen.queryByTestId('loading')).not.toBeInTheDocument();
    });

    // Component should not crash
    expect(screen.getByTestId('results')).toBeInTheDocument();
  });

  it('should show loading state during API call', async () => {
    let resolvePromise: (value: any) => void;
    const promise = new Promise((resolve) => {
      resolvePromise = resolve;
    });

    (musicApiService.searchTracks as jest.Mock).mockReturnValue(promise);

    const user = userEvent.setup();
    render(
      <QueryClientProvider client={queryClient}>
        <SearchComponent />
      </QueryClientProvider>
    );

    const input = screen.getByTestId('search-input');
    const button = screen.getByTestId('search-button');

    await user.type(input, 'test');
    await user.click(button);

    await waitFor(() => {
      expect(screen.getByTestId('loading')).toBeInTheDocument();
    });

    resolvePromise!({ tracks: [] });

    await waitFor(() => {
      expect(screen.queryByTestId('loading')).not.toBeInTheDocument();
    });
  });

  it('should handle empty search results', async () => {
    (musicApiService.searchTracks as jest.Mock).mockResolvedValue({
      tracks: [],
    });

    const user = userEvent.setup();
    render(
      <QueryClientProvider client={queryClient}>
        <SearchComponent />
      </QueryClientProvider>
    );

    const input = screen.getByTestId('search-input');
    const button = screen.getByTestId('search-button');

    await user.type(input, 'test');
    await user.click(button);

    await waitFor(() => {
      expect(screen.queryByTestId('loading')).not.toBeInTheDocument();
    });

    const results = screen.getByTestId('results');
    expect(results.children.length).toBe(0);
  });
});

