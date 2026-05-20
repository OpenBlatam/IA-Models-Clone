import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TopArtists } from '@/components/music/TopArtists';
import { musicApiService } from '@/lib/api/music-api';

jest.mock('@/lib/api/music-api', () => ({
  musicApiService: {
    getAnalytics: jest.fn(),
  },
}));

const mockGetAnalytics = musicApiService.getAnalytics as jest.MockedFunction<
  typeof musicApiService.getAnalytics
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

describe('TopArtists', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should render top artists component', () => {
    mockGetAnalytics.mockResolvedValue({} as any);

    render(<TopArtists />, { wrapper: createWrapper() });

    expect(screen.getByText(/top artistas/i)).toBeInTheDocument();
  });

  it('should display top 5 artists', () => {
    mockGetAnalytics.mockResolvedValue({} as any);

    render(<TopArtists />, { wrapper: createWrapper() });

    expect(screen.getByText('Artista 1')).toBeInTheDocument();
    expect(screen.getByText('Artista 2')).toBeInTheDocument();
    expect(screen.getByText('Artista 3')).toBeInTheDocument();
    expect(screen.getByText('Artista 4')).toBeInTheDocument();
    expect(screen.getByText('Artista 5')).toBeInTheDocument();
  });

  it('should display artist rankings', () => {
    mockGetAnalytics.mockResolvedValue({} as any);

    render(<TopArtists />, { wrapper: createWrapper() });

    // Should show ranking numbers
    expect(screen.getByText('1')).toBeInTheDocument();
    expect(screen.getByText('2')).toBeInTheDocument();
    expect(screen.getByText('3')).toBeInTheDocument();
  });

  it('should display track counts for each artist', () => {
    mockGetAnalytics.mockResolvedValue({} as any);

    render(<TopArtists />, { wrapper: createWrapper() });

    expect(screen.getByText('45 canciones')).toBeInTheDocument();
    expect(screen.getByText('38 canciones')).toBeInTheDocument();
  });

  it('should display popularity percentages', () => {
    mockGetAnalytics.mockResolvedValue({} as any);

    render(<TopArtists />, { wrapper: createWrapper() });

    expect(screen.getByText('95%')).toBeInTheDocument();
    expect(screen.getByText('88%')).toBeInTheDocument();
  });

  it('should show crown icon for top artist', () => {
    mockGetAnalytics.mockResolvedValue({} as any);

    render(<TopArtists />, { wrapper: createWrapper() });

    // First artist should have crown
    const artist1 = screen.getByText('Artista 1').closest('div');
    const crownIcon = artist1?.querySelector('.lucide-crown');
    expect(crownIcon).toBeInTheDocument();
  });

  it('should show user icon for other artists', () => {
    mockGetAnalytics.mockResolvedValue({} as any);

    render(<TopArtists />, { wrapper: createWrapper() });

    // Second artist should have user icon
    const artist2 = screen.getByText('Artista 2').closest('div');
    const userIcon = artist2?.querySelector('.lucide-user');
    expect(userIcon).toBeInTheDocument();
  });

  it('should display popularity bars', () => {
    mockGetAnalytics.mockResolvedValue({} as any);

    const { container } = render(<TopArtists />, { wrapper: createWrapper() });

    // Should have progress bars
    const progressBars = container.querySelectorAll('.bg-purple-500');
    expect(progressBars.length).toBeGreaterThan(0);
  });
});

