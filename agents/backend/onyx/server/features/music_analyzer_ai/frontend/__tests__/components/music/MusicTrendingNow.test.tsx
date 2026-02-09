import { render, screen, fireEvent } from '@testing-library/react';
import { MusicTrendingNow } from '@/components/music/MusicTrendingNow';
import { type Track } from '@/lib/api/music-api';

describe('MusicTrendingNow', () => {
  const mockTrack: Track = {
    id: '1',
    name: 'Test Track',
    artists: ['Test Artist'],
    album: 'Test Album',
    duration_ms: 200000,
    popularity: 80,
  };

  it('renders trending tracks', () => {
    render(<MusicTrendingNow />);
    
    expect(screen.getByText('Tendencias Ahora')).toBeInTheDocument();
    expect(screen.getByText('Bohemian Rhapsody')).toBeInTheDocument();
    expect(screen.getByText('Stairway to Heaven')).toBeInTheDocument();
  });

  it('displays track names and artists', () => {
    render(<MusicTrendingNow />);
    
    expect(screen.getByText('Bohemian Rhapsody')).toBeInTheDocument();
    expect(screen.getByText('Queen')).toBeInTheDocument();
    expect(screen.getByText('Led Zeppelin')).toBeInTheDocument();
  });

  it('displays trend percentages', () => {
    render(<MusicTrendingNow />);
    
    expect(screen.getByText('+12.5%')).toBeInTheDocument();
    expect(screen.getByText('+8.3%')).toBeInTheDocument();
    expect(screen.getByText('+6.7%')).toBeInTheDocument();
  });

  it('displays ranking numbers', () => {
    render(<MusicTrendingNow />);
    
    expect(screen.getByText('1')).toBeInTheDocument();
    expect(screen.getByText('2')).toBeInTheDocument();
    expect(screen.getByText('3')).toBeInTheDocument();
  });

  it('calls onTrackSelect when play button is clicked', () => {
    const handleTrackSelect = jest.fn();
    render(<MusicTrendingNow onTrackSelect={handleTrackSelect} />);
    
    const playButtons = screen.getAllByRole('button');
    const playButton = playButtons.find(btn => 
      btn.querySelector('svg')?.getAttribute('data-testid') === 'play-icon' ||
      btn.textContent === ''
    );
    
    if (playButton) {
      fireEvent.click(playButton);
      // The component should call onTrackSelect with the track
      expect(handleTrackSelect).toHaveBeenCalled();
    }
  });

  it('renders all 4 trending tracks', () => {
    render(<MusicTrendingNow />);
    
    const tracks = [
      'Bohemian Rhapsody',
      'Stairway to Heaven',
      'Hotel California',
      "Sweet Child O Mine",
    ];
    
    tracks.forEach(track => {
      expect(screen.getByText(track)).toBeInTheDocument();
    });
  });
});

