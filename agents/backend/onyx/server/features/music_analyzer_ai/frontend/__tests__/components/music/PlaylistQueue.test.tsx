import { render, screen, fireEvent } from '@testing-library/react';
import { PlaylistQueue } from '@/components/music/PlaylistQueue';
import { type Track } from '@/lib/api/music-api';

describe('PlaylistQueue', () => {
  const mockTracks: Track[] = [
    {
      id: '1',
      name: 'Track 1',
      artists: ['Artist 1'],
      album: 'Album 1',
      duration_ms: 200000,
      popularity: 80,
    },
    {
      id: '2',
      name: 'Track 2',
      artists: ['Artist 2'],
      album: 'Album 2',
      duration_ms: 180000,
      popularity: 75,
    },
  ];

  const mockHandlers = {
    onTrackSelect: jest.fn(),
    onRemove: jest.fn(),
    onReorder: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders playlist queue', () => {
    render(<PlaylistQueue tracks={mockTracks} />);
    
    expect(screen.getByText('Track 1')).toBeInTheDocument();
    expect(screen.getByText('Track 2')).toBeInTheDocument();
  });

  it('displays empty state when no tracks', () => {
    render(<PlaylistQueue tracks={[]} />);
    
    expect(screen.getByText(/vacía|empty/i)).toBeInTheDocument();
  });

  it('calls onTrackSelect when track is clicked', () => {
    render(<PlaylistQueue tracks={mockTracks} {...mockHandlers} />);
    
    const track1 = screen.getByText('Track 1');
    fireEvent.click(track1);
    
    expect(mockHandlers.onTrackSelect).toHaveBeenCalledWith(mockTracks[0]);
  });

  it('calls onRemove when remove button is clicked', () => {
    render(<PlaylistQueue tracks={mockTracks} {...mockHandlers} />);
    
    const removeButtons = screen.getAllByRole('button', { name: /remove|eliminar/i });
    if (removeButtons.length > 0) {
      fireEvent.click(removeButtons[0]);
      expect(mockHandlers.onRemove).toHaveBeenCalledWith('1');
    }
  });

  it('highlights current track', () => {
    render(<PlaylistQueue tracks={mockTracks} currentTrackId="1" />);
    
    const track1 = screen.getByText('Track 1');
    expect(track1.closest('div')).toHaveClass(/current|active/i);
  });

  it('handles drag and drop reordering', () => {
    render(<PlaylistQueue tracks={mockTracks} {...mockHandlers} />);
    
    const track1 = screen.getByText('Track 1');
    const track2 = screen.getByText('Track 2');
    
    // Simulate drag and drop
    fireEvent.dragStart(track1);
    fireEvent.dragOver(track2);
    fireEvent.drop(track2);
    
    // onReorder should be called
    expect(mockHandlers.onReorder).toHaveBeenCalled();
  });
});

