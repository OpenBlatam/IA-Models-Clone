import { render, screen } from '@testing-library/react';
import { WaveformVisualizer } from '@/components/music/WaveformVisualizer';
import { type Track } from '@/lib/api/music-api';

describe('WaveformVisualizer', () => {
  const mockTrack: Track = {
    id: '1',
    name: 'Test Track',
    artists: ['Test Artist'],
    album: 'Test Album',
    duration_ms: 200000,
    popularity: 80,
    preview_url: 'https://example.com/preview.mp3',
  };

  it('renders waveform visualizer', () => {
    render(<WaveformVisualizer track={mockTrack} audioUrl={mockTrack.preview_url} />);
    
    expect(screen.getByText('Test Track')).toBeInTheDocument();
  });

  it('displays canvas element', () => {
    render(<WaveformVisualizer track={mockTrack} audioUrl={mockTrack.preview_url} />);
    
    const canvas = document.querySelector('canvas');
    expect(canvas).toBeInTheDocument();
  });

  it('displays empty state when no audio URL', () => {
    render(<WaveformVisualizer track={mockTrack} />);
    
    expect(screen.getByText(/no disponible|not available/i)).toBeInTheDocument();
  });

  it('displays track information', () => {
    render(<WaveformVisualizer track={mockTrack} audioUrl={mockTrack.preview_url} />);
    
    expect(screen.getByText('Test Track')).toBeInTheDocument();
    expect(screen.getByText('Test Artist')).toBeInTheDocument();
  });
});

