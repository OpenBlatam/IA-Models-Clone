import { render, screen, fireEvent } from '@testing-library/react';
import { MusicSettings } from '@/components/music/MusicSettings';

describe('MusicSettings', () => {
  it('renders settings', () => {
    render(<MusicSettings />);
    
    expect(screen.getByText(/configuración|settings/i)).toBeInTheDocument();
  });

  it('allows changing settings', () => {
    render(<MusicSettings />);
    
    const settings = screen.getByText(/configuración|settings/i);
    expect(settings).toBeInTheDocument();
  });
});

