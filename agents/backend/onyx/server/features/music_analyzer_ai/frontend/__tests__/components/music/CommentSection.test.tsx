import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { CommentSection } from '@/components/music/CommentSection';
import { musicApiService } from '@/lib/api/music-api';
import * as toast from 'react-hot-toast';

jest.mock('@/lib/api/music-api', () => ({
  musicApiService: {
    getComments: jest.fn(),
    addComment: jest.fn(),
    updateComment: jest.fn(),
    deleteComment: jest.fn(),
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

describe('CommentSection', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders comment section', () => {
    mockedMusicApiService.getComments.mockResolvedValue({ comments: [] });
    
    renderWithQueryClient(<CommentSection resourceId="1" resourceType="track" />);
    
    expect(screen.getByText(/comentarios|comments/i)).toBeInTheDocument();
  });

  it('displays existing comments', async () => {
    const mockComments = {
      comments: [
        {
          id: '1',
          content: 'Great track!',
          user: 'User1',
          created_at: '2024-01-01',
        },
        {
          id: '2',
          content: 'Love it!',
          user: 'User2',
          created_at: '2024-01-02',
        },
      ],
    };
    
    mockedMusicApiService.getComments.mockResolvedValue(mockComments);
    
    renderWithQueryClient(<CommentSection resourceId="1" resourceType="track" />);
    
    await waitFor(() => {
      expect(screen.getByText('Great track!')).toBeInTheDocument();
      expect(screen.getByText('Love it!')).toBeInTheDocument();
    });
  });

  it('allows adding new comment', async () => {
    const toastSpy = jest.spyOn(toast, 'default');
    mockedMusicApiService.getComments.mockResolvedValue({ comments: [] });
    mockedMusicApiService.addComment.mockResolvedValue({ success: true });
    
    renderWithQueryClient(<CommentSection resourceId="1" resourceType="track" />);
    
    const textarea = screen.getByPlaceholderText(/comentar|comment/i);
    fireEvent.change(textarea, { target: { value: 'New comment' } });
    
    const submitButton = screen.getByRole('button', { name: /enviar|submit/i });
    fireEvent.click(submitButton);
    
    await waitFor(() => {
      expect(mockedMusicApiService.addComment).toHaveBeenCalled();
      expect(toastSpy.success).toHaveBeenCalled();
    });
  });

  it('allows editing own comment', async () => {
    const toastSpy = jest.spyOn(toast, 'default');
    const mockComments = {
      comments: [
        {
          id: '1',
          content: 'Original comment',
          user: 'CurrentUser',
          created_at: '2024-01-01',
        },
      ],
    };
    
    mockedMusicApiService.getComments.mockResolvedValue(mockComments);
    mockedMusicApiService.updateComment.mockResolvedValue({ success: true });
    
    renderWithQueryClient(<CommentSection resourceId="1" resourceType="track" />);
    
    await waitFor(() => {
      expect(screen.getByText('Original comment')).toBeInTheDocument();
    });
    
    const editButton = screen.getByRole('button', { name: /editar|edit/i });
    fireEvent.click(editButton);
    
    const textarea = screen.getByDisplayValue('Original comment');
    fireEvent.change(textarea, { target: { value: 'Updated comment' } });
    
    const saveButton = screen.getByRole('button', { name: /guardar|save/i });
    fireEvent.click(saveButton);
    
    await waitFor(() => {
      expect(mockedMusicApiService.updateComment).toHaveBeenCalled();
      expect(toastSpy.success).toHaveBeenCalled();
    });
  });

  it('allows deleting own comment', async () => {
    const toastSpy = jest.spyOn(toast, 'default');
    const mockComments = {
      comments: [
        {
          id: '1',
          content: 'Comment to delete',
          user: 'CurrentUser',
          created_at: '2024-01-01',
        },
      ],
    };
    
    mockedMusicApiService.getComments.mockResolvedValue(mockComments);
    mockedMusicApiService.deleteComment.mockResolvedValue({ success: true });
    
    renderWithQueryClient(<CommentSection resourceId="1" resourceType="track" />);
    
    await waitFor(() => {
      expect(screen.getByText('Comment to delete')).toBeInTheDocument();
    });
    
    const deleteButton = screen.getByRole('button', { name: /eliminar|delete/i });
    fireEvent.click(deleteButton);
    
    await waitFor(() => {
      expect(mockedMusicApiService.deleteComment).toHaveBeenCalled();
      expect(toastSpy.success).toHaveBeenCalled();
    });
  });
});

