import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { PlaylistManager } from '@/components/music/PlaylistManager';
import { musicApiService } from '@/lib/api/music-api';
import * as toast from 'react-hot-toast';

jest.mock('@/lib/api/music-api', () => ({
  musicApiService: {
    getPlaylists: jest.fn(),
    createPlaylist: jest.fn(),
    deletePlaylist: jest.fn(),
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

describe('PlaylistManager', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders playlist manager', () => {
    mockedMusicApiService.getPlaylists.mockResolvedValue({ playlists: [] });
    
    renderWithQueryClient(<PlaylistManager />);
    
    expect(screen.getByText(/playlists|mis playlists/i)).toBeInTheDocument();
  });

  it('displays create playlist button', () => {
    mockedMusicApiService.getPlaylists.mockResolvedValue({ playlists: [] });
    
    renderWithQueryClient(<PlaylistManager />);
    
    const createButton = screen.getByText(/crear|nueva/i);
    expect(createButton).toBeInTheDocument();
  });

  it('shows create form when button is clicked', () => {
    mockedMusicApiService.getPlaylists.mockResolvedValue({ playlists: [] });
    
    renderWithQueryClient(<PlaylistManager />);
    
    const createButton = screen.getByText(/crear|nueva/i);
    fireEvent.click(createButton);
    
    expect(screen.getByPlaceholderText(/nombre|name/i)).toBeInTheDocument();
  });

  it('creates playlist successfully', async () => {
    const toastSpy = jest.spyOn(toast, 'default');
    mockedMusicApiService.getPlaylists.mockResolvedValue({ playlists: [] });
    mockedMusicApiService.createPlaylist.mockResolvedValue({ success: true });
    
    renderWithQueryClient(<PlaylistManager />);
    
    const createButton = screen.getByText(/crear|nueva/i);
    fireEvent.click(createButton);
    
    const input = screen.getByPlaceholderText(/nombre|name/i);
    fireEvent.change(input, { target: { value: 'My Playlist' } });
    
    const submitButton = screen.getByText(/crear|guardar/i);
    fireEvent.click(submitButton);
    
    await waitFor(() => {
      expect(mockedMusicApiService.createPlaylist).toHaveBeenCalled();
      expect(toastSpy.success).toHaveBeenCalled();
    });
  });

  it('displays error when playlist name is empty', async () => {
    const toastSpy = jest.spyOn(toast, 'default');
    mockedMusicApiService.getPlaylists.mockResolvedValue({ playlists: [] });
    
    renderWithQueryClient(<PlaylistManager />);
    
    const createButton = screen.getByText(/crear|nueva/i);
    fireEvent.click(createButton);
    
    const submitButton = screen.getByText(/crear|guardar/i);
    fireEvent.click(submitButton);
    
    await waitFor(() => {
      expect(toastSpy.error).toHaveBeenCalled();
    });
  });

  it('displays playlists list', async () => {
    const mockPlaylists = {
      playlists: [
        { id: '1', name: 'Playlist 1', track_count: 10 },
        { id: '2', name: 'Playlist 2', track_count: 5 },
      ],
    };
    
    mockedMusicApiService.getPlaylists.mockResolvedValue(mockPlaylists);
    
    renderWithQueryClient(<PlaylistManager />);
    
    await waitFor(() => {
      expect(screen.getByText('Playlist 1')).toBeInTheDocument();
      expect(screen.getByText('Playlist 2')).toBeInTheDocument();
    });
  });
});

