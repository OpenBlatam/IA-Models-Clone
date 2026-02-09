import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { PlaylistManager } from '@/components/music/PlaylistManager';
import { musicApiService } from '@/lib/api/music-api';

jest.mock('@/lib/api/music-api', () => ({
  musicApiService: {
    getPlaylists: jest.fn(),
    createPlaylist: jest.fn(),
  },
}));

jest.mock('react-hot-toast', () => ({
  default: {
    success: jest.fn(),
    error: jest.fn(),
  },
}));

const mockGetPlaylists = musicApiService.getPlaylists as jest.MockedFunction<
  typeof musicApiService.getPlaylists
>;
const mockCreatePlaylist = musicApiService.createPlaylist as jest.MockedFunction<
  typeof musicApiService.createPlaylist
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

describe('PlaylistManager', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should render playlist manager', () => {
    mockGetPlaylists.mockResolvedValue({
      playlists: [],
    } as any);

    render(<PlaylistManager />, { wrapper: createWrapper() });

    expect(screen.getByText(/mis playlists/i)).toBeInTheDocument();
    expect(screen.getByText(/nueva playlist/i)).toBeInTheDocument();
  });

  it('should show loading state', () => {
    mockGetPlaylists.mockImplementation(
      () =>
        new Promise(() => {
          // Never resolves
        })
    );

    render(<PlaylistManager />, { wrapper: createWrapper() });

    expect(screen.getByText(/cargando playlists/i)).toBeInTheDocument();
  });

  it('should show empty state when no playlists', async () => {
    mockGetPlaylists.mockResolvedValue({
      playlists: [],
    } as any);

    render(<PlaylistManager />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText(/no tienes playlists aún/i)).toBeInTheDocument();
    });
  });

  it('should display playlists when available', async () => {
    const mockPlaylists = [
      {
        id: '1',
        name: 'My Playlist',
        track_count: 10,
        is_public: true,
      },
      {
        id: '2',
        name: 'Private Playlist',
        track_count: 5,
        is_public: false,
      },
    ];

    mockGetPlaylists.mockResolvedValue({
      playlists: mockPlaylists,
    } as any);

    render(<PlaylistManager />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('My Playlist')).toBeInTheDocument();
      expect(screen.getByText('Private Playlist')).toBeInTheDocument();
    });
  });

  it('should show create form when button is clicked', async () => {
    const user = userEvent.setup();
    mockGetPlaylists.mockResolvedValue({
      playlists: [],
    } as any);

    render(<PlaylistManager />, { wrapper: createWrapper() });

    const createButton = screen.getByText(/nueva playlist/i);
    await user.click(createButton);

    expect(
      screen.getByPlaceholderText(/nombre de la playlist/i)
    ).toBeInTheDocument();
  });

  it('should create playlist when form is submitted', async () => {
    const user = userEvent.setup();
    mockGetPlaylists.mockResolvedValue({
      playlists: [],
    } as any);

    mockCreatePlaylist.mockResolvedValue({
      success: true,
      playlist: { id: '1', name: 'New Playlist' },
    } as any);

    render(<PlaylistManager />, { wrapper: createWrapper() });

    const createButton = screen.getByText(/nueva playlist/i);
    await user.click(createButton);

    const nameInput = screen.getByPlaceholderText(/nombre de la playlist/i);
    await user.type(nameInput, 'New Playlist');

    const submitButton = screen.getByText('Crear');
    await user.click(submitButton);

    await waitFor(() => {
      expect(mockCreatePlaylist).toHaveBeenCalledWith(
        'user123',
        'New Playlist',
        false
      );
    });
  });

  it('should show error when playlist name is empty', async () => {
    const user = userEvent.setup();
    mockGetPlaylists.mockResolvedValue({
      playlists: [],
    } as any);

    render(<PlaylistManager />, { wrapper: createWrapper() });

    const createButton = screen.getByText(/nueva playlist/i);
    await user.click(createButton);

    const submitButton = screen.getByText('Crear');
    await user.click(submitButton);

    // Should show error (via toast)
    expect(mockCreatePlaylist).not.toHaveBeenCalled();
  });

  it('should handle public checkbox', async () => {
    const user = userEvent.setup();
    mockGetPlaylists.mockResolvedValue({
      playlists: [],
    } as any);

    mockCreatePlaylist.mockResolvedValue({
      success: true,
      playlist: { id: '1', name: 'Public Playlist' },
    } as any);

    render(<PlaylistManager />, { wrapper: createWrapper() });

    const createButton = screen.getByText(/nueva playlist/i);
    await user.click(createButton);

    const nameInput = screen.getByPlaceholderText(/nombre de la playlist/i);
    await user.type(nameInput, 'Public Playlist');

    const publicCheckbox = screen.getByLabelText(/pública/i);
    await user.click(publicCheckbox);

    const submitButton = screen.getByText('Crear');
    await user.click(submitButton);

    await waitFor(() => {
      expect(mockCreatePlaylist).toHaveBeenCalledWith(
        'user123',
        'Public Playlist',
        true
      );
    });
  });

  it('should cancel form creation', async () => {
    const user = userEvent.setup();
    mockGetPlaylists.mockResolvedValue({
      playlists: [],
    } as any);

    render(<PlaylistManager />, { wrapper: createWrapper() });

    const createButton = screen.getByText(/nueva playlist/i);
    await user.click(createButton);

    expect(
      screen.getByPlaceholderText(/nombre de la playlist/i)
    ).toBeInTheDocument();

    const cancelButton = screen.getByText('Cancelar');
    await user.click(cancelButton);

    expect(
      screen.queryByPlaceholderText(/nombre de la playlist/i)
    ).not.toBeInTheDocument();
  });

  it('should display playlist track count', async () => {
    const mockPlaylists = [
      {
        id: '1',
        name: 'Playlist 1',
        track_count: 15,
        is_public: true,
      },
    ];

    mockGetPlaylists.mockResolvedValue({
      playlists: mockPlaylists,
    } as any);

    render(<PlaylistManager />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('15 canciones')).toBeInTheDocument();
    });
  });

  it('should display playlist privacy status', async () => {
    const mockPlaylists = [
      {
        id: '1',
        name: 'Public Playlist',
        track_count: 10,
        is_public: true,
      },
      {
        id: '2',
        name: 'Private Playlist',
        track_count: 5,
        is_public: false,
      },
    ];

    mockGetPlaylists.mockResolvedValue({
      playlists: mockPlaylists,
    } as any);

    render(<PlaylistManager />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Pública')).toBeInTheDocument();
      expect(screen.getByText('Privada')).toBeInTheDocument();
    });
  });
});

