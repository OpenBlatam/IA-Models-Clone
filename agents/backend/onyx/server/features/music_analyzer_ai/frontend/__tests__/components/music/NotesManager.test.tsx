import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { NotesManager } from '@/components/music/NotesManager';
import { musicApiService } from '@/lib/api/music-api';
import * as toast from 'react-hot-toast';

jest.mock('@/lib/api/music-api', () => ({
  musicApiService: {
    getNotes: jest.fn(),
    addNote: jest.fn(),
    updateNote: jest.fn(),
    deleteNote: jest.fn(),
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

describe('NotesManager', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders notes manager', () => {
    mockedMusicApiService.getNotes.mockResolvedValue({ notes: [] });
    
    renderWithQueryClient(<NotesManager resourceId="1" resourceType="track" />);
    
    expect(screen.getByText(/notas|notes/i)).toBeInTheDocument();
  });

  it('displays existing notes', async () => {
    const mockNotes = {
      notes: [
        { id: '1', content: 'Great track!', created_at: '2024-01-01' },
        { id: '2', content: 'Love the bass', created_at: '2024-01-02' },
      ],
    };
    
    mockedMusicApiService.getNotes.mockResolvedValue(mockNotes);
    
    renderWithQueryClient(<NotesManager resourceId="1" resourceType="track" />);
    
    await waitFor(() => {
      expect(screen.getByText('Great track!')).toBeInTheDocument();
      expect(screen.getByText('Love the bass')).toBeInTheDocument();
    });
  });

  it('allows adding new note', async () => {
    const toastSpy = jest.spyOn(toast, 'default');
    mockedMusicApiService.getNotes.mockResolvedValue({ notes: [] });
    mockedMusicApiService.addNote.mockResolvedValue({ success: true });
    
    renderWithQueryClient(<NotesManager resourceId="1" resourceType="track" />);
    
    const addButton = screen.getByRole('button', { name: /agregar|add/i });
    fireEvent.click(addButton);
    
    const textarea = screen.getByPlaceholderText(/escribir|write/i);
    fireEvent.change(textarea, { target: { value: 'New note content' } });
    
    const saveButton = screen.getByRole('button', { name: /guardar|save/i });
    fireEvent.click(saveButton);
    
    await waitFor(() => {
      expect(mockedMusicApiService.addNote).toHaveBeenCalledWith('1', 'track', 'New note content');
      expect(toastSpy.success).toHaveBeenCalled();
    });
  });

  it('allows editing note', async () => {
    const toastSpy = jest.spyOn(toast, 'default');
    const mockNotes = {
      notes: [
        { id: '1', content: 'Original note', created_at: '2024-01-01' },
      ],
    };
    
    mockedMusicApiService.getNotes.mockResolvedValue(mockNotes);
    mockedMusicApiService.updateNote.mockResolvedValue({ success: true });
    
    renderWithQueryClient(<NotesManager resourceId="1" resourceType="track" />);
    
    await waitFor(() => {
      expect(screen.getByText('Original note')).toBeInTheDocument();
    });
    
    const editButton = screen.getByRole('button', { name: /editar|edit/i });
    fireEvent.click(editButton);
    
    const textarea = screen.getByDisplayValue('Original note');
    fireEvent.change(textarea, { target: { value: 'Updated note' } });
    
    const saveButton = screen.getByRole('button', { name: /guardar|save/i });
    fireEvent.click(saveButton);
    
    await waitFor(() => {
      expect(mockedMusicApiService.updateNote).toHaveBeenCalledWith({
        noteId: '1',
        content: 'Updated note',
      });
      expect(toastSpy.success).toHaveBeenCalled();
    });
  });

  it('allows deleting note', async () => {
    const toastSpy = jest.spyOn(toast, 'default');
    const mockNotes = {
      notes: [
        { id: '1', content: 'Note to delete', created_at: '2024-01-01' },
      ],
    };
    
    mockedMusicApiService.getNotes.mockResolvedValue(mockNotes);
    mockedMusicApiService.deleteNote.mockResolvedValue({ success: true });
    
    renderWithQueryClient(<NotesManager resourceId="1" resourceType="track" />);
    
    await waitFor(() => {
      expect(screen.getByText('Note to delete')).toBeInTheDocument();
    });
    
    const deleteButton = screen.getByRole('button', { name: /eliminar|delete/i });
    fireEvent.click(deleteButton);
    
    await waitFor(() => {
      expect(mockedMusicApiService.deleteNote).toHaveBeenCalledWith('1');
      expect(toastSpy.success).toHaveBeenCalled();
    });
  });
});

