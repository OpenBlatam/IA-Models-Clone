import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TrackComparison } from '@/components/music/TrackComparison';
import { musicApiService } from '@/lib/api/music-api';
import { type Track } from '@/lib/api/music-api';
import * as toast from 'react-hot-toast';

jest.mock('@/lib/api/music-api', () => ({
  musicApiService: {
    compareTracks: jest.fn(),
  },
}));

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: {
    success: jest.fn(),
    error: jest.fn(),
  },
}));

const mockedMusicApiService = musicApiService as jest.Mocked<typeof musicApiService>;

const createTestQueryClient = () => new QueryClient({
  defaultOptions: {
    queries: { retry: false },
    mutations: { retry: false },
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

describe('TrackComparison', () => {
  const mockTracks: Track[] = [
    {
      id: '1',
      name: 'Track 1',
      artists: ['Artist 1'],
      album: 'Album 1',
      duration_ms: 200000,
      popularity: 80,
    },
    {
      id: '2',
      name: 'Track 2',
      artists: ['Artist 2'],
      album: 'Album 2',
      duration_ms: 180000,
      popularity: 75,
    },
    {
      id: '3',
      name: 'Track 3',
      artists: ['Artist 3'],
      album: 'Album 3',
      duration_ms: 220000,
      popularity: 85,
    },
  ];

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders comparison component', () => {
    renderWithQueryClient(<TrackComparison tracks={mockTracks} />);
    
    expect(screen.getByText(/comparar|compare/i)).toBeInTheDocument();
  });

  it('displays available tracks', () => {
    renderWithQueryClient(<TrackComparison tracks={mockTracks} />);
    
    expect(screen.getByText('Track 1')).toBeInTheDocument();
    expect(screen.getByText('Track 2')).toBeInTheDocument();
    expect(screen.getByText('Track 3')).toBeInTheDocument();
  });

  it('allows selecting tracks for comparison', () => {
    renderWithQueryClient(<TrackComparison tracks={mockTracks} />);
    
    const track1 = screen.getByText('Track 1');
    fireEvent.click(track1);
    
    // Track should be selected
    expect(screen.getByText(/seleccionado|selected/i)).toBeInTheDocument();
  });

  it('prevents selecting more than 5 tracks', () => {
    const toastSpy = jest.spyOn(toast, 'default');
    const manyTracks: Track[] = Array.from({ length: 10 }, (_, i) => ({
      id: `${i}`,
      name: `Track ${i}`,
      artists: [`Artist ${i}`],
      album: `Album ${i}`,
      duration_ms: 200000,
      popularity: 80,
    }));
    
    renderWithQueryClient(<TrackComparison tracks={manyTracks} />);
    
    // Select 5 tracks
    for (let i = 0; i < 5; i++) {
      const track = screen.getByText(`Track ${i}`);
      fireEvent.click(track);
    }
    
    // Try to select 6th track
    const track6 = screen.getByText('Track 6');
    fireEvent.click(track6);
    
    expect(toastSpy.error).toHaveBeenCalledWith('Máximo 5 canciones para comparar');
  });

  it('requires at least 2 tracks to compare', async () => {
    const toastSpy = jest.spyOn(toast, 'default');
    renderWithQueryClient(<TrackComparison tracks={mockTracks} />);
    
    // Select only 1 track
    const track1 = screen.getByText('Track 1');
    fireEvent.click(track1);
    
    // Try to compare
    const compareButton = screen.getByText(/comparar|compare/i);
    fireEvent.click(compareButton);
    
    expect(toastSpy.error).toHaveBeenCalledWith('Selecciona al menos 2 canciones para comparar');
  });

  it('compares tracks successfully', async () => {
    const toastSpy = jest.spyOn(toast, 'default');
    const mockComparison = {
      similarities: { tempo: 0.9, key: 0.8 },
      differences: { energy: 0.3 },
    };
    
    mockedMusicApiService.compareTracks.mockResolvedValue(mockComparison);
    
    renderWithQueryClient(<TrackComparison tracks={mockTracks} />);
    
    // Select 2 tracks
    const track1 = screen.getByText('Track 1');
    const track2 = screen.getByText('Track 2');
    fireEvent.click(track1);
    fireEvent.click(track2);
    
    // Compare
    const compareButton = screen.getByText(/comparar|compare/i);
    fireEvent.click(compareButton);
    
    await waitFor(() => {
      expect(mockedMusicApiService.compareTracks).toHaveBeenCalledWith(['1', '2']);
      expect(toastSpy.success).toHaveBeenCalled();
    });
  });
});

