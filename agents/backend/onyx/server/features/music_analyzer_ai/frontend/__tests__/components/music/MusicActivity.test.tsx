import { render, screen } from '@testing-library/react';
import { MusicActivity } from '@/components/music/MusicActivity';

describe('MusicActivity', () => {
  it('renders activity feed', () => {
    render(<MusicActivity />);
    
    expect(screen.getByText('Actividad Reciente')).toBeInTheDocument();
  });

  it('displays user activities', () => {
    render(<MusicActivity />);
    
    expect(screen.getByText('MusicLover23')).toBeInTheDocument();
    expect(screen.getByText('BeatHunter')).toBeInTheDocument();
    expect(screen.getByText('SoundExplorer')).toBeInTheDocument();
  });

  it('displays activity types', () => {
    render(<MusicActivity />);
    
    expect(screen.getByText(/analizó/)).toBeInTheDocument();
    expect(screen.getByText(/agregó a favoritos/)).toBeInTheDocument();
    expect(screen.getByText(/comparó/)).toBeInTheDocument();
    expect(screen.getByText(/descubrió/)).toBeInTheDocument();
  });

  it('displays track names in activities', () => {
    render(<MusicActivity />);
    
    expect(screen.getByText('Bohemian Rhapsody')).toBeInTheDocument();
    expect(screen.getByText('Stairway to Heaven')).toBeInTheDocument();
    expect(screen.getByText('3 canciones')).toBeInTheDocument();
  });

  it('displays timestamps', () => {
    render(<MusicActivity />);
    
    expect(screen.getByText('Hace 5 minutos')).toBeInTheDocument();
    expect(screen.getByText('Hace 15 minutos')).toBeInTheDocument();
    expect(screen.getByText('Hace 1 hora')).toBeInTheDocument();
  });

  it('renders all 4 activities', () => {
    render(<MusicActivity />);
    
    const activities = screen.getAllByText(/MusicLover23|BeatHunter|SoundExplorer|MelodySeeker/);
    expect(activities.length).toBeGreaterThanOrEqual(4);
  });
});

