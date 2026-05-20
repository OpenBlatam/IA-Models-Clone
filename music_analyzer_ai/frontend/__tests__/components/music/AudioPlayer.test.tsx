import { render, screen, fireEvent } from '@testing-library/react';
import { AudioPlayer } from '@/components/music/AudioPlayer';
import { type Track } from '@/lib/api/music-api';

describe('AudioPlayer', () => {
  const mockTrack: Track = {
    id: '1',
    name: 'Test Track',
    artists: ['Test Artist'],
    album: 'Test Album',
    duration_ms: 200000,
    popularity: 80,
    preview_url: 'https://example.com/preview.mp3',
  };

  it('renders audio player', () => {
    render(<AudioPlayer track={mockTrack} />);
    
    expect(screen.getByText('Test Track')).toBeInTheDocument();
  });

  it('displays track information', () => {
    render(<AudioPlayer track={mockTrack} />);
    
    expect(screen.getByText('Test Track')).toBeInTheDocument();
    expect(screen.getByText('Test Artist')).toBeInTheDocument();
  });

  it('renders play button', () => {
    render(<AudioPlayer track={mockTrack} />);
    
    const playButton = screen.getByRole('button', { name: /play/i });
    expect(playButton).toBeInTheDocument();
  });

  it('handles play/pause toggle', () => {
    render(<AudioPlayer track={mockTrack} />);
    
    const playButton = screen.getByRole('button', { name: /play/i });
    fireEvent.click(playButton);
    
    // Button should change to pause
    expect(screen.getByRole('button', { name: /pause/i })).toBeInTheDocument();
  });

  it('displays audio element', () => {
    render(<AudioPlayer track={mockTrack} />);
    
    const audioElement = screen.getByTestId('audio-element') || document.querySelector('audio');
    expect(audioElement).toBeInTheDocument();
  });

  it('handles volume change', () => {
    render(<AudioPlayer track={mockTrack} />);
    
    const volumeSlider = screen.getByRole('slider', { name: /volume/i });
    if (volumeSlider) {
      fireEvent.change(volumeSlider, { target: { value: '0.5' } });
      expect(volumeSlider).toHaveValue(0.5);
    }
  });
});

