import { render, screen } from '@testing-library/react';
import { TrackAnalysis } from '@/components/music/TrackAnalysis';
import { type Track } from '@/lib/api/music-api';

describe('TrackAnalysis', () => {
  const mockTrack: Track = {
    id: '1',
    name: 'Test Track',
    artists: ['Test Artist'],
    album: 'Test Album',
    duration_ms: 200000,
    popularity: 80,
  };

  const mockAnalysis = {
    musical: {
      key_signature: 'C',
      root_note: 'C',
      mode: 'major',
      tempo: {
        bpm: 120,
        category: 'moderate',
      },
      time_signature: '4/4',
      scale: {
        name: 'C Major',
        notes: ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
      },
    },
    technical: {
      energy: { value: 0.8, description: 'High energy' },
      danceability: { value: 0.7, description: 'Moderately danceable' },
      valence: { value: 0.6, description: 'Positive mood' },
    },
  };

  it('renders track analysis', () => {
    render(<TrackAnalysis track={mockTrack} analysis={mockAnalysis} />);
    
    expect(screen.getByText('Test Track')).toBeInTheDocument();
    expect(screen.getByText('Test Artist')).toBeInTheDocument();
  });

  it('displays musical analysis', () => {
    render(<TrackAnalysis track={mockTrack} analysis={mockAnalysis} />);
    
    expect(screen.getByText(/key|tonalidad/i)).toBeInTheDocument();
    expect(screen.getByText(/tempo|bpm/i)).toBeInTheDocument();
  });

  it('displays technical analysis', () => {
    render(<TrackAnalysis track={mockTrack} analysis={mockAnalysis} />);
    
    expect(screen.getByText(/energy|energía/i)).toBeInTheDocument();
    expect(screen.getByText(/danceability|bailabilidad/i)).toBeInTheDocument();
  });

  it('handles missing analysis data gracefully', () => {
    render(<TrackAnalysis track={mockTrack} analysis={null} />);
    
    expect(screen.getByText('Test Track')).toBeInTheDocument();
  });
});

