import { render, screen } from '@testing-library/react';
import { MusicStreak } from '@/components/music/MusicStreak';

describe('MusicStreak', () => {
  it('renders streak section', () => {
    render(<MusicStreak />);
    
    expect(screen.getByText(/Racha|Streak/i)).toBeInTheDocument();
  });

  it('displays streak information', () => {
    render(<MusicStreak />);
    
    // Check if streak elements are rendered
    const streakSection = screen.getByText(/Racha|Streak/i);
    expect(streakSection).toBeInTheDocument();
  });
});

