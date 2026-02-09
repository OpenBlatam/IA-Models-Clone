import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TrackRating } from '@/components/music/TrackRating';
import { musicApiService } from '@/lib/api/music-api';
import * as toast from 'react-hot-toast';

jest.mock('@/lib/api/music-api', () => ({
  musicApiService: {
    getRating: jest.fn(),
    addRating: jest.fn(),
    updateRating: jest.fn(),
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

describe('TrackRating', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders rating component', () => {
    mockedMusicApiService.getRating.mockResolvedValue({ rating: 0 });
    
    renderWithQueryClient(<TrackRating trackId="1" />);
    
    expect(screen.getByText(/calificación|rating/i)).toBeInTheDocument();
  });

  it('displays current rating', async () => {
    mockedMusicApiService.getRating.mockResolvedValue({ rating: 4 });
    
    renderWithQueryClient(<TrackRating trackId="1" />);
    
    await waitFor(() => {
      const stars = screen.getAllByRole('button', { name: /star/i });
      expect(stars.length).toBeGreaterThan(0);
    });
  });

  it('allows rating a track', async () => {
    const toastSpy = jest.spyOn(toast, 'default');
    mockedMusicApiService.getRating.mockResolvedValue({ rating: 0 });
    mockedMusicApiService.addRating.mockResolvedValue({ success: true });
    
    renderWithQueryClient(<TrackRating trackId="1" />);
    
    await waitFor(() => {
      const stars = screen.getAllByRole('button', { name: /star/i });
      if (stars.length > 0) {
        fireEvent.click(stars[3]); // Click 4th star (rating 4)
        
        expect(mockedMusicApiService.addRating).toHaveBeenCalled();
        expect(toastSpy.success).toHaveBeenCalled();
      }
    });
  });

  it('allows updating rating', async () => {
    const toastSpy = jest.spyOn(toast, 'default');
    mockedMusicApiService.getRating.mockResolvedValue({ rating: 3 });
    mockedMusicApiService.updateRating.mockResolvedValue({ success: true });
    
    renderWithQueryClient(<TrackRating trackId="1" />);
    
    await waitFor(() => {
      const stars = screen.getAllByRole('button', { name: /star/i });
      if (stars.length > 0) {
        fireEvent.click(stars[4]); // Click 5th star (rating 5)
        
        expect(mockedMusicApiService.updateRating).toHaveBeenCalled();
        expect(toastSpy.success).toHaveBeenCalled();
      }
    });
  });
});

