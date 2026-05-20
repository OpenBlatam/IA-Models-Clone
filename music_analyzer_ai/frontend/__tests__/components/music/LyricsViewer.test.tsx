import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { LyricsViewer } from '@/components/music/LyricsViewer';
import { musicApiService } from '@/lib/api/music-api';

jest.mock('@/lib/api/music-api', () => ({
  musicApiService: {
    getLyrics: jest.fn(),
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

describe('LyricsViewer', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders lyrics viewer', () => {
    mockedMusicApiService.getLyrics.mockResolvedValue({ lyrics: 'Test lyrics' });
    
    renderWithQueryClient(
      <LyricsViewer trackId="1" trackName="Test Track" artists={['Test Artist']} />
    );
    
    expect(screen.getByText(/letra|lyrics/i)).toBeInTheDocument();
  });

  it('displays lyrics when loaded', async () => {
    const mockLyrics = {
      lyrics: 'This is a test lyric line\nAnother line here',
      synced: false,
    };
    
    mockedMusicApiService.getLyrics.mockResolvedValue(mockLyrics);
    
    renderWithQueryClient(
      <LyricsViewer trackId="1" trackName="Test Track" artists={['Test Artist']} />
    );
    
    await waitFor(() => {
      expect(screen.getByText('This is a test lyric line')).toBeInTheDocument();
    });
  });

  it('displays loading state', () => {
    mockedMusicApiService.getLyrics.mockImplementation(() => new Promise(() => {}));
    
    renderWithQueryClient(
      <LyricsViewer trackId="1" trackName="Test Track" artists={['Test Artist']} />
    );
    
    expect(screen.getByText(/cargando|loading/i)).toBeInTheDocument();
  });

  it('displays error when lyrics not found', async () => {
    mockedMusicApiService.getLyrics.mockRejectedValue(new Error('Lyrics not found'));
    
    renderWithQueryClient(
      <LyricsViewer trackId="1" trackName="Test Track" artists={['Test Artist']} />
    );
    
    await waitFor(() => {
      expect(screen.getByText(/no disponible|not available/i)).toBeInTheDocument();
    });
  });

  it('displays track information', () => {
    mockedMusicApiService.getLyrics.mockResolvedValue({ lyrics: 'Test lyrics' });
    
    renderWithQueryClient(
      <LyricsViewer trackId="1" trackName="Test Track" artists={['Test Artist']} />
    );
    
    expect(screen.getByText('Test Track')).toBeInTheDocument();
    expect(screen.getByText('Test Artist')).toBeInTheDocument();
  });
});

