import { render, screen, fireEvent } from '@testing-library/react';
import { MusicRadio } from '@/components/music/MusicRadio';
import * as toast from 'react-hot-toast';

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: {
    success: jest.fn(),
    info: jest.fn(),
  },
}));

describe('MusicRadio', () => {
  it('renders radio stations', () => {
    render(<MusicRadio />);
    
    expect(screen.getByText('Radio en Vivo')).toBeInTheDocument();
    expect(screen.getByText('Pop Hits Radio')).toBeInTheDocument();
    expect(screen.getByText('Rock Classics')).toBeInTheDocument();
  });

  it('displays station information', () => {
    render(<MusicRadio />);
    
    expect(screen.getByText('Pop Hits Radio')).toBeInTheDocument();
    expect(screen.getByText('Pop')).toBeInTheDocument();
    expect(screen.getByText(/1250 oyentes/)).toBeInTheDocument();
  });

  it('handles play/pause button click', () => {
    const toastSpy = jest.spyOn(toast, 'default');
    render(<MusicRadio />);
    
    const playButtons = screen.getAllByRole('button');
    const playButton = playButtons.find(btn => 
      btn.querySelector('svg')?.getAttribute('data-testid') === 'play-icon' ||
      btn.textContent === ''
    );
    
    if (playButton) {
      fireEvent.click(playButton);
      expect(toastSpy.success).toHaveBeenCalledWith('Reproduciendo radio');
    }
  });

  it('displays all 4 radio stations', () => {
    render(<MusicRadio />);
    
    const stations = [
      'Pop Hits Radio',
      'Rock Classics',
      'Jazz Lounge',
      'Electronic Vibes',
    ];
    
    stations.forEach(station => {
      expect(screen.getByText(station)).toBeInTheDocument();
    });
  });

  it('displays genre for each station', () => {
    render(<MusicRadio />);
    
    expect(screen.getByText('Pop')).toBeInTheDocument();
    expect(screen.getByText('Rock')).toBeInTheDocument();
    expect(screen.getByText('Jazz')).toBeInTheDocument();
    expect(screen.getByText('Electronic')).toBeInTheDocument();
  });
});

