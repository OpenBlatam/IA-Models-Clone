import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Recommendations } from '@/components/music/Recommendations';
import { musicApiService } from '@/lib/api/music-api';
import { type Track } from '@/lib/api/music-api';
import * as toast from 'react-hot-toast';

jest.mock('@/lib/api/music-api', () => ({
  musicApiService: {
    getRecommendations: jest.fn(),
    getRecommendationsByMood: jest.fn(),
    getRecommendationsByActivity: jest.fn(),
    getRecommendationsByTimeOfDay: jest.fn(),
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

describe('Recommendations', () => {
  const mockTrack: Track = {
    id: '1',
    name: 'Test Track',
    artists: ['Test Artist'],
    album: 'Test Album',
    duration_ms: 200000,
    popularity: 80,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders recommendations component', () => {
    renderWithQueryClient(<Recommendations trackId="1" track={mockTrack} />);
    
    expect(screen.getByText(/recomendaciones|recommendations/i)).toBeInTheDocument();
  });

  it('displays track information', () => {
    renderWithQueryClient(<Recommendations trackId="1" track={mockTrack} />);
    
    expect(screen.getByText('Test Track')).toBeInTheDocument();
    expect(screen.getByText('Test Artist')).toBeInTheDocument();
  });

  it('allows selecting recommendation type', () => {
    renderWithQueryClient(<Recommendations trackId="1" track={mockTrack} />);
    
    const similarButton = screen.getByText(/similar/i);
    expect(similarButton).toBeInTheDocument();
  });

  it('gets similar recommendations', async () => {
    const toastSpy = jest.spyOn(toast, 'default');
    const mockRecommendations = {
      tracks: [
        {
          id: '2',
          name: 'Recommended Track',
          artists: ['Recommended Artist'],
          album: 'Recommended Album',
          duration_ms: 200000,
          popularity: 75,
        },
      ],
    };
    
    mockedMusicApiService.getRecommendations.mockResolvedValue(mockRecommendations);
    
    renderWithQueryClient(<Recommendations trackId="1" track={mockTrack} />);
    
    const getButton = screen.getByText(/obtener|get/i);
    fireEvent.click(getButton);
    
    await waitFor(() => {
      expect(mockedMusicApiService.getRecommendations).toHaveBeenCalledWith('1', 20);
      expect(toastSpy.success).toHaveBeenCalled();
    });
  });

  it('gets mood-based recommendations', async () => {
    const toastSpy = jest.spyOn(toast, 'default');
    const mockRecommendations = {
      tracks: [],
    };
    
    mockedMusicApiService.getRecommendationsByMood.mockResolvedValue(mockRecommendations);
    
    renderWithQueryClient(<Recommendations trackId="1" track={mockTrack} />);
    
    // Select mood type
    const moodButton = screen.getByText(/mood|estado de ánimo/i);
    fireEvent.click(moodButton);
    
    // Get recommendations
    const getButton = screen.getByText(/obtener|get/i);
    fireEvent.click(getButton);
    
    await waitFor(() => {
      expect(mockedMusicApiService.getRecommendationsByMood).toHaveBeenCalled();
      expect(toastSpy.success).toHaveBeenCalled();
    });
  });

  it('gets activity-based recommendations', async () => {
    const toastSpy = jest.spyOn(toast, 'default');
    const mockRecommendations = {
      tracks: [],
    };
    
    mockedMusicApiService.getRecommendationsByActivity.mockResolvedValue(mockRecommendations);
    
    renderWithQueryClient(<Recommendations trackId="1" track={mockTrack} />);
    
    // Select activity type
    const activityButton = screen.getByText(/activity|actividad/i);
    fireEvent.click(activityButton);
    
    // Get recommendations
    const getButton = screen.getByText(/obtener|get/i);
    fireEvent.click(getButton);
    
    await waitFor(() => {
      expect(mockedMusicApiService.getRecommendationsByActivity).toHaveBeenCalled();
      expect(toastSpy.success).toHaveBeenCalled();
    });
  });

  it('handles recommendation errors', async () => {
    const toastSpy = jest.spyOn(toast, 'default');
    const error = new Error('API Error');
    
    mockedMusicApiService.getRecommendations.mockRejectedValue(error);
    
    renderWithQueryClient(<Recommendations trackId="1" track={mockTrack} />);
    
    const getButton = screen.getByText(/obtener|get/i);
    fireEvent.click(getButton);
    
    await waitFor(() => {
      expect(toastSpy.error).toHaveBeenCalled();
    });
  });
});

