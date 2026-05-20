import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TagManager } from '@/components/music/TagManager';
import { musicApiService } from '@/lib/api/music-api';
import * as toast from 'react-hot-toast';

jest.mock('@/lib/api/music-api', () => ({
  musicApiService: {
    getTags: jest.fn(),
    addTag: jest.fn(),
    removeTag: jest.fn(),
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

describe('TagManager', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders tag manager', () => {
    mockedMusicApiService.getTags.mockResolvedValue({ tags: [] });
    
    renderWithQueryClient(<TagManager resourceId="1" resourceType="track" />);
    
    expect(screen.getByText(/tags|etiquetas/i)).toBeInTheDocument();
  });

  it('displays existing tags', async () => {
    const mockTags = {
      tags: ['rock', 'energetic', 'favorite'],
    };
    
    mockedMusicApiService.getTags.mockResolvedValue(mockTags);
    
    renderWithQueryClient(<TagManager resourceId="1" resourceType="track" />);
    
    await waitFor(() => {
      expect(screen.getByText('rock')).toBeInTheDocument();
      expect(screen.getByText('energetic')).toBeInTheDocument();
      expect(screen.getByText('favorite')).toBeInTheDocument();
    });
  });

  it('allows adding new tag', async () => {
    const toastSpy = jest.spyOn(toast, 'default');
    mockedMusicApiService.getTags.mockResolvedValue({ tags: [] });
    mockedMusicApiService.addTag.mockResolvedValue({ success: true });
    
    renderWithQueryClient(<TagManager resourceId="1" resourceType="track" />);
    
    const input = screen.getByPlaceholderText(/agregar|add/i);
    fireEvent.change(input, { target: { value: 'new-tag' } });
    
    const addButton = screen.getByRole('button', { name: /agregar|add/i });
    fireEvent.click(addButton);
    
    await waitFor(() => {
      expect(mockedMusicApiService.addTag).toHaveBeenCalledWith('1', 'track', ['new-tag']);
      expect(toastSpy.success).toHaveBeenCalled();
    });
  });

  it('allows removing tag', async () => {
    const toastSpy = jest.spyOn(toast, 'default');
    const mockTags = {
      tags: ['rock', 'energetic'],
    };
    
    mockedMusicApiService.getTags.mockResolvedValue(mockTags);
    mockedMusicApiService.removeTag.mockResolvedValue({ success: true });
    
    renderWithQueryClient(<TagManager resourceId="1" resourceType="track" />);
    
    await waitFor(() => {
      expect(screen.getByText('rock')).toBeInTheDocument();
    });
    
    const removeButtons = screen.getAllByRole('button', { name: /remove|eliminar/i });
    if (removeButtons.length > 0) {
      fireEvent.click(removeButtons[0]);
      
      await waitFor(() => {
        expect(mockedMusicApiService.removeTag).toHaveBeenCalled();
        expect(toastSpy.success).toHaveBeenCalled();
      });
    }
  });

  it('prevents adding empty tag', () => {
    mockedMusicApiService.getTags.mockResolvedValue({ tags: [] });
    
    renderWithQueryClient(<TagManager resourceId="1" resourceType="track" />);
    
    const addButton = screen.getByRole('button', { name: /agregar|add/i });
    fireEvent.click(addButton);
    
    expect(mockedMusicApiService.addTag).not.toHaveBeenCalled();
  });
});

