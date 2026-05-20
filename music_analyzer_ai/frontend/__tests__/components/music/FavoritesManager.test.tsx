import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { FavoritesManager } from '@/components/music/FavoritesManager';
import { musicApiService } from '@/lib/api/music-api';
import * as toast from 'react-hot-toast';

jest.mock('@/lib/api/music-api', () => ({
  musicApiService: {
    getFavorites: jest.fn(),
    addToFavorites: jest.fn(),
    removeFromFavorites: jest.fn(),
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

describe('FavoritesManager', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders favorites manager', () => {
    mockedMusicApiService.getFavorites.mockResolvedValue({ tracks: [] });
    
    renderWithQueryClient(<FavoritesManager />);
    
    expect(screen.getByText(/favoritos|favorites/i)).toBeInTheDocument();
  });

  it('displays loading state', () => {
    mockedMusicApiService.getFavorites.mockImplementation(() => new Promise(() => {}));
    
    renderWithQueryClient(<FavoritesManager />);
    
    expect(screen.getByText(/cargando|loading/i)).toBeInTheDocument();
  });

  it('displays favorites list', async () => {
    const mockFavorites = {
      tracks: [
        {
          id: '1',
          name: 'Favorite Track 1',
          artists: ['Artist 1'],
          album: 'Album 1',
          duration_ms: 200000,
          popularity: 90,
        },
        {
          id: '2',
          name: 'Favorite Track 2',
          artists: ['Artist 2'],
          album: 'Album 2',
          duration_ms: 180000,
          popularity: 85,
        },
      ],
    };
    
    mockedMusicApiService.getFavorites.mockResolvedValue(mockFavorites);
    
    renderWithQueryClient(<FavoritesManager />);
    
    await waitFor(() => {
      expect(screen.getByText('Favorite Track 1')).toBeInTheDocument();
      expect(screen.getByText('Favorite Track 2')).toBeInTheDocument();
    });
  });

  it('removes favorite successfully', async () => {
    const toastSpy = jest.spyOn(toast, 'default');
    const mockFavorites = {
      tracks: [
        {
          id: '1',
          name: 'Favorite Track 1',
          artists: ['Artist 1'],
          album: 'Album 1',
          duration_ms: 200000,
          popularity: 90,
        },
      ],
    };
    
    mockedMusicApiService.getFavorites.mockResolvedValue(mockFavorites);
    mockedMusicApiService.removeFromFavorites.mockResolvedValue({ success: true });
    
    renderWithQueryClient(<FavoritesManager />);
    
    await waitFor(() => {
      expect(screen.getByText('Favorite Track 1')).toBeInTheDocument();
    });
    
    const removeButton = screen.getByRole('button', { name: /eliminar|remove|delete/i });
    fireEvent.click(removeButton);
    
    await waitFor(() => {
      expect(mockedMusicApiService.removeFromFavorites).toHaveBeenCalled();
      expect(toastSpy.success).toHaveBeenCalled();
    });
  });

  it('displays empty state when no favorites', async () => {
    mockedMusicApiService.getFavorites.mockResolvedValue({ tracks: [] });
    
    renderWithQueryClient(<FavoritesManager />);
    
    await waitFor(() => {
      expect(screen.getByText(/no hay|empty|sin favoritos/i)).toBeInTheDocument();
    });
  });
});

