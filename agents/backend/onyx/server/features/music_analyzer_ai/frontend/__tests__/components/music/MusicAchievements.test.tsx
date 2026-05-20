import { render, screen } from '@testing-library/react';
import { MusicAchievements } from '@/components/music/MusicAchievements';

describe('MusicAchievements', () => {
  it('renders achievements section', () => {
    render(<MusicAchievements />);
    
    expect(screen.getByText(/Logros|Achievements/i)).toBeInTheDocument();
  });

  it('displays achievement progress', () => {
    render(<MusicAchievements />);
    
    // Check if achievement elements are rendered
    const achievementSection = screen.getByText(/Logros|Achievements/i);
    expect(achievementSection).toBeInTheDocument();
  });
});

